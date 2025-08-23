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
BANNED = ["noto", "bitcount", "nabla", "jersey"]
font_files = [
    font_file
    for font_file in font_files
    if not any(ban in font_file.lower() for ban in BANNED)
]
if LIMIT > 0:
    font_files = font_files[:LIMIT]
if not font_files:
    raise ValueError(f"No suitable font files found in {BASE_DIR}")


def pretrain_dataset_generator(font_file_list, name=""):
    print(f"Running {name} on {len(font_file_list)} x {len(ALPHABET)} = {len(font_file_list) * len(ALPHABET)} glyphs")

    for font_file_path in font_file_list:
        font_file = Path(font_file_path)
        for i, char in enumerate(ALPHABET):
            glyph = Glyph(font_file, ord(char), location={})
            try:
                raster = glyph.rasterize(GEN_IMAGE_SIZE[0])
            except Exception as e:
                print(f"Error rasterizing glyph {char} from {font_file}: {e}")
                continue

            svg_glyph = glyph.vectorize()
            node_glyph = svg_glyph.to_node_glyph()
            if not node_glyph.commands:
                continue
            encoded_svg = node_glyph.encode()
            if encoded_svg is None:
                continue

            target_sequence = encoded_svg[:-1]
            ground_truth = encoded_svg[1:]
            true_command = ground_truth[:, :NODE_COMMAND_WIDTH]
            true_coord = ground_truth[:, NODE_COMMAND_WIDTH:].astype(np.float32)
            yield ((raster, target_sequence), (true_command, true_coord))


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
    output_signature=(
        (
            tf.TensorSpec(shape=(GEN_IMAGE_SIZE[0], GEN_IMAGE_SIZE[1], 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, NODE_COMMAND_WIDTH + COORDINATE_WIDTH), dtype=tf.int32),
        ),
        (
            tf.TensorSpec(shape=(None, NODE_COMMAND_WIDTH), dtype=tf.int32),
            tf.TensorSpec(shape=(None, COORDINATE_WIDTH), dtype=tf.float32),
        ),
    )
    train_dataset = tf.data.Dataset.from_generator(
        lambda: pretrain_dataset_generator(train_fonts, "train"),
        output_signature=output_signature
    )
    test_dataset = tf.data.Dataset.from_generator(
        lambda: pretrain_dataset_generator(test_fonts, "test"),
        output_signature=output_signature
    )
    return train_dataset, test_dataset
