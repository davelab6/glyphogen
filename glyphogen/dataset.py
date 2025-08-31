from glyphogen.glyph import (
    NODE_COMMAND_WIDTH,
    COORDINATE_WIDTH,
    Glyph,
)
from glyphogen.hyperparameters import (
    NUM_GLYPHS,
    GEN_IMAGE_SIZE,
    BASE_DIR,
    ALPHABET,
    LIMIT,
    NUM_GLYPHS,
)
import glob
import os
import tqdm
from pathlib import Path
from glyphogen.rendering import get_style_image
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

font_files = list(glob.glob(BASE_DIR + "/*/*.ttf"))
# Filter out certain bad fonts
BANNED = ["noto", "bitcount", "nabla", "jersey", "rubik", "winky"]
font_files = [
    font_file
    for font_file in font_files
    if not any(ban in font_file.lower() for ban in BANNED)
]
if LIMIT > 0:
    font_files = font_files[:LIMIT]
if not font_files:
    raise ValueError(f"No suitable font files found in {BASE_DIR}")


def _load_and_process_pretrain_glyph(font_file_path_tensor, char_ord_tensor):
    # This function is designed to be wrapped by tf.py_function.
    # It takes tensors, converts them to python types, does processing, and returns numpy arrays.
    font_file_path = font_file_path_tensor.numpy().decode("utf-8")
    char_ord = char_ord_tensor.numpy()

    # Dummy data to return on error
    dummy_raster = np.zeros(
        (GEN_IMAGE_SIZE[0], GEN_IMAGE_SIZE[1], 1), dtype=np.float32
    )
    dummy_seq = np.zeros(
        (0, NODE_COMMAND_WIDTH + COORDINATE_WIDTH), dtype=np.int32
    )
    dummy_cmd = np.zeros((0, NODE_COMMAND_WIDTH), dtype=np.int32)
    dummy_coord = np.zeros((0, COORDINATE_WIDTH), dtype=np.float32)

    try:
        glyph = Glyph(Path(font_file_path), int(char_ord), location={})
        raster = glyph.rasterize(GEN_IMAGE_SIZE[0])
        svg_glyph = glyph.vectorize()
        node_glyph = svg_glyph.to_node_glyph()

        if not node_glyph.commands or len(node_glyph.commands) <= 1:
            print(f"Couldn't process glyph {char_ord} from {font_file_path} (no commands)")
            return dummy_raster, dummy_seq, dummy_cmd, dummy_coord

        encoded_svg = node_glyph.encode()
        if encoded_svg is None or len(encoded_svg) <= 1:
            # print(f"Couldn't process glyph {char_ord} from {font_file_path} (bad encoding)")
            return dummy_raster, dummy_seq, dummy_cmd, dummy_coord

        target_sequence = encoded_svg[:-1]
        ground_truth = encoded_svg[1:]
        true_command = ground_truth[:, :NODE_COMMAND_WIDTH]
        true_coord = ground_truth[:, NODE_COMMAND_WIDTH:].astype(np.float32)
        # print(f"Processed glyph {char_ord} from {font_file_path}")

        return raster, target_sequence, true_command, true_coord

    except Exception as e:
        print(f"Couldn't process glyph {char_ord} from {font_file_path}: {e}")
        return dummy_raster, dummy_seq, dummy_cmd, dummy_coord


def _is_not_dummy(inputs, outputs):
    # Filter based on sequence length. If it's 0, it's a dummy.
    # The `target_sequence` is at inputs[1]
    return tf.shape(inputs[1])[0] > 0


def create_real_dataset():
    style_images = []
    glyph_ids = []
    encoded_svgs = []
    true_rasters = []

    for font_file_path in tqdm.tqdm(font_files):
        font_file = Path(font_file_path)
        try:
            style_image = get_style_image(
                font_file, variation={}
            )  # Assuming no variation for now
        except Exception as e:
            print(f"Error getting style image for {font_file}: {e}")
            continue

        for i, char in enumerate(ALPHABET):
            glyph = Glyph(font_file, ord(char), location={})

            # Rasterize
            try:
                raster = glyph.rasterize(GEN_IMAGE_SIZE[0])
            except Exception as e:
                print(f"Error rasterizing glyph {char} from {font_file}: {e}")
                continue

            # Vectorize
            svg_glyph = glyph.vectorize()
            node_glyph = svg_glyph.to_node_glyph()
            if not node_glyph.commands:
                continue
            encoded_svg = node_glyph.encode()
            if encoded_svg is None:
                continue

            # One-hot encode glyph ID
            glyph_id_one_hot = tf.one_hot(i, NUM_GLYPHS).numpy()

            style_images.append(style_image)
            glyph_ids.append(glyph_id_one_hot)
            encoded_svgs.append(encoded_svg)
            true_rasters.append(raster)

    # Convert lists to numpy arrays
    style_images = np.array(style_images)
    glyph_ids = np.array(glyph_ids)
    encoded_svgs = np.array(encoded_svgs)
    true_rasters = np.array(true_rasters)

    # Split data into training and testing sets
    (
        style_images_train,
        style_images_test,
        glyph_ids_train,
        glyph_ids_test,
        encoded_svgs_train,
        encoded_svgs_test,
        true_rasters_train,
        true_rasters_test,
    ) = train_test_split(
        style_images,
        glyph_ids,
        encoded_svgs,
        true_rasters,
        test_size=0.2,  # 20% for testing
        random_state=42,
    )

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (style_images_train, glyph_ids_train, encoded_svgs_train, true_rasters_train)
    )

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (style_images_test, glyph_ids_test, encoded_svgs_test, true_rasters_test)
    )

    return train_dataset, test_dataset


def prepare_data(style_image, glyph_id, encoded_svg, true_raster):
    target_sequence = encoded_svg[:-1]
    ground_truth = encoded_svg[1:]
    true_command = ground_truth[:, :NODE_COMMAND_WIDTH]
    true_coord = ground_truth[:, NODE_COMMAND_WIDTH:]
    true_coord = tf.cast(true_coord, tf.float32)
    return (style_image, glyph_id, target_sequence), {
        "raster": true_raster,
        "command": true_command,
        "coord": true_coord,
    }


def get_full_model_data():
    # Create real dataset
    if not os.path.exists("train.tfds"):
        train_dataset, test_dataset = create_real_dataset()
        train_dataset.save("train.tfds")
        test_dataset.save("test.tfds")

    # Load 'em anyway, because using an on-disk version saves memory
    train_dataset = tf.data.Dataset.load("train.tfds")
    test_dataset = tf.data.Dataset.load("test.tfds")
    return train_dataset, test_dataset


def get_pretrain_data():
    # Split list of fonts into train/test
    train_fonts, test_fonts = train_test_split(
        font_files, test_size=0.2, random_state=42
    )

    def build_dataset(font_list, name):
        print(f"Building pre-train dataset '{name}' with {len(font_list)} fonts.")

        def generator():
            for font_file_path in font_list:
                for char in ALPHABET:
                    yield (font_file_path, ord(char))

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            ),
        )

        def map_py_function(path, char_ord):
            raster, seq, cmd, coord = tf.py_function(
                _load_and_process_pretrain_glyph,
                [path, char_ord],
                [tf.float32, tf.int32, tf.int32, tf.float32],
            )
            # Restore shape information lost by tf.py_function
            raster.set_shape([GEN_IMAGE_SIZE[0], GEN_IMAGE_SIZE[1], 1])
            seq.set_shape([None, NODE_COMMAND_WIDTH + COORDINATE_WIDTH])
            cmd.set_shape([None, NODE_COMMAND_WIDTH])
            coord.set_shape([None, COORDINATE_WIDTH])
            return ((raster, seq), (cmd, coord))

        dataset = dataset.map(map_py_function, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.filter(_is_not_dummy)
        return dataset

    train_dataset = build_dataset(train_fonts, "train")
    test_dataset = build_dataset(test_fonts, "test")

    return train_dataset, test_dataset