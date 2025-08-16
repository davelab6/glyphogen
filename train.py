#!/usr/bin/env python
import os
import tensorflow as tf
import keras
import tqdm
import datetime

# If the dataset doesn't exist, we turn off the GPU because it's too big
# to generate and move about on the GPU.
if not os.path.exists("train.tfds"):
    tf.config.set_visible_devices([], "GPU")


from fontTools.ttLib import TTFont
from deepvecfont3.model import GlyphGenerator, VectorizationGenerator
from deepvecfont3.glyph import NODE_COMMAND_WIDTH, Glyph, NodeGlyph, SVGGlyph
from deepvecfont3.hyperparameters import (
    BATCH_SIZE,
    NUM_GLYPHS,
    MAX_COMMANDS,
    GEN_IMAGE_SIZE,
    LATENT_DIM,
    D_MODEL,
    RASTER_LOSS_WEIGHT,
    VECTOR_LOSS_WEIGHT_COMMAND,
    VECTOR_LOSS_WEIGHT_COORD,
    RATE,
    EPOCHS,
    BASE_DIR,
    ALPHABET,
    LIMIT,
    LEARNING_RATE,
)
from deepvecfont3.rendering import get_style_image


import glob
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, alpha, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.alpha = alpha
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)
        return self.alpha * tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {"d_model": self.d_model, "warmup_steps": self.warmup_steps, "alpha": self.alpha}


def create_real_dataset():
    style_images = []
    glyph_ids = []
    target_sequences = []
    true_rasters = []
    true_commands = []
    true_coords = []

    font_files = list(glob.glob(BASE_DIR + "/*/*.ttf"))
    if LIMIT > 0:
        font_files = font_files[:LIMIT]
    if not font_files:
        raise ValueError(f"No font files found in {BASE_DIR}")

    MAX_SEQUENCE_LENGTH = MAX_COMMANDS + 1  # for EOS token

    for font_file_path in tqdm.tqdm(font_files):
        font_file = Path(font_file_path)
        if "noto" in font_file.name.lower():
            continue

        # Ensure all have same upem
        if TTFont(font_file)["head"].unitsPerEm != 1000:
            continue

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

            # Pad vector representation or skip
            if encoded_svg.shape[0] > MAX_SEQUENCE_LENGTH:
                continue
            elif encoded_svg.shape[0] < MAX_SEQUENCE_LENGTH:
                padding = np.zeros(
                    (MAX_SEQUENCE_LENGTH - encoded_svg.shape[0], encoded_svg.shape[1])
                )
                encoded_svg = np.vstack((encoded_svg, padding))

            decoder_input = encoded_svg[:-1]
            ground_truth = encoded_svg[1:]

            # Split into command and coordinate parts
            command_part = ground_truth[:, :NODE_COMMAND_WIDTH]
            coord_part = ground_truth[:, NODE_COMMAND_WIDTH:]

            # One-hot encode glyph ID
            glyph_id_one_hot = tf.one_hot(i, NUM_GLYPHS).numpy()

            style_images.append(style_image)
            glyph_ids.append(glyph_id_one_hot)
            target_sequences.append(decoder_input)
            true_rasters.append(raster)
            true_commands.append(command_part)
            true_coords.append(coord_part)

    # Convert lists to numpy arrays
    style_images = np.array(style_images)
    glyph_ids = np.array(glyph_ids)
    target_sequences = np.array(target_sequences)
    true_rasters = np.array(true_rasters)
    true_commands = np.array(true_commands)
    true_coords = np.array(true_coords)

    # Split data into training and testing sets
    (
        style_images_train,
        style_images_test,
        glyph_ids_train,
        glyph_ids_test,
        target_sequences_train,
        target_sequences_test,
        true_rasters_train,
        true_rasters_test,
        true_commands_train,
        true_commands_test,
        true_coords_train,
        true_coords_test,
    ) = train_test_split(
        style_images,
        glyph_ids,
        target_sequences,
        true_rasters,
        true_commands,
        true_coords,
        test_size=0.2,  # 20% for testing
        random_state=42,
    )

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (
            (style_images_train, glyph_ids_train, target_sequences_train),
            {
                "raster": true_rasters_train,
                "command": true_commands_train,
                "coord": true_coords_train,
            },
        )
    )
    train_dataset = (
        train_dataset.shuffle(buffer_size=len(style_images_train))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (
            (style_images_test, glyph_ids_test, target_sequences_test),
            {
                "raster": true_rasters_test,
                "command": true_commands_test,
                "coord": true_coords_test,
            },
        )
    )
    test_dataset = (
        test_dataset.shuffle(buffer_size=len(style_images_test))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_dataset, test_dataset





class ImageGenerationCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, test_dataset, num_images=3, pre_train=False):
        super().__init__()
        self.file_writer = tf.summary.create_file_writer(log_dir + "/images")
        self.test_dataset = test_dataset.unbatch().take(num_images)
        self.pre_train = pre_train

    def on_epoch_end(self, epoch, logs=None):
        if self.pre_train:
            # Don't generate images during pre-training
            return

        for i, (
            (style_image, glyph_id, target_sequence),
            (true_raster, true_command, true_coord),
        ) in enumerate(self.test_dataset):
            # Add batch dimension
            style_image = tf.expand_dims(style_image, axis=0)
            glyph_id = tf.expand_dims(glyph_id, axis=0)
            target_sequence = tf.expand_dims(target_sequence, axis=0)

            # Predict generated raster
            generated_raster = self.model((style_image, glyph_id, target_sequence))[
                "raster"
            ]
            true_raster = tf.expand_dims(true_raster, axis=0)

            assert true_raster.shape == generated_raster.shape, (
                f"Shape mismatch: true_raster {true_raster.shape}, "
                f"generated_raster {generated_raster.shape}"
            )

            with self.file_writer.as_default():
                tf.summary.image(
                    f"True Glyph Raster - Sample {i}",
                    true_raster,
                    step=epoch,
                )
                tf.summary.image(
                    f"Generated Glyph Raster - Sample {i}", generated_raster, step=epoch
                )


class SVGGenerationCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, test_dataset, num_samples=3, pre_train=False):
        super().__init__()
        self.file_writer = tf.summary.create_file_writer(log_dir + "/svgs")
        self.test_dataset = test_dataset.unbatch().take(num_samples)
        self.pre_train = pre_train

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5:
            return
        model = self.model

        for i, (inputs, outputs) in enumerate(self.test_dataset):
            if self.pre_train:
                raster_image_input, target_sequence_input = inputs

                # Add batch dimension
                raster_image_input = tf.expand_dims(raster_image_input, axis=0)
                target_sequence_input = tf.expand_dims(target_sequence_input, axis=0)

                output = model(
                    (raster_image_input, target_sequence_input), training=False
                )
            else:
                (style_image, glyph_id, target_sequence_input) = inputs

                # Add batch dimension
                style_image = tf.expand_dims(style_image, axis=0)
                glyph_id = tf.expand_dims(glyph_id, axis=0)
                target_sequence_input = tf.expand_dims(target_sequence_input, axis=0)

                output = model(
                    (style_image, glyph_id, target_sequence_input), training=False
                )

            command_output = output["command"]
            coord_output = output["coord"]

            # a single batch
            command_tensor = command_output[0]
            coord_tensor = coord_output[0]

            decoded_glyph = NodeGlyph.from_numpy(
                command_tensor.numpy(), coord_tensor.numpy()
            )
            try:
                svg_string = decoded_glyph.to_svg_glyph().to_svg_string()
            except Exception:
                svg_string = "DEBUG " + decoded_glyph.to_debug_string()

            with self.file_writer.as_default():
                tf.summary.text(f"Generated SVG - Sample {i}", svg_string, step=epoch)


def get_data():
    # Create real dataset
    if not os.path.exists("train.tfds"):
        train_dataset, test_dataset = create_real_dataset()
        train_dataset.save("train.tfds")
        test_dataset.save("test.tfds")

    # Load 'em anyway, because using an on-disk version saves memory
    train_dataset = tf.data.Dataset.load("train.tfds")
    test_dataset = tf.data.Dataset.load("test.tfds")
    return train_dataset, test_dataset


def _preprocess_for_vectorizer_dataset(x, y):
    return ((y["raster"], x[2]), (y["command"], y["coord"]))


def main(
    model_name="deepvecfont3.keras",
    pre_train=False,
    epochs=EPOCHS,
    vectorizer_model_name=None,
):
    # Load the model if it exists
    if os.path.exists(f"{model_name}") and not vectorizer_model_name:
        model = tf.keras.models.load_model(
            model_name,
            custom_objects={
                "GlyphGenerator": GlyphGenerator,
                "VectorizationGenerator": VectorizationGenerator,
            },
        )
        print(f"Loaded model from {model_name}")
    else:
        # Create the model
        model = GlyphGenerator(
            num_glyphs=NUM_GLYPHS,
            d_model=D_MODEL,
            latent_dim=LATENT_DIM,
            rate=RATE,
        )

    if vectorizer_model_name:
        vectorizer = tf.keras.models.load_model(
            vectorizer_model_name,
            custom_objects={
                "VectorizationGenerator": VectorizationGenerator,
            },
        )
        model.vectorizer = vectorizer
        print(f"Loaded vectorizer from {vectorizer_model_name}")

    train_dataset, test_dataset = get_data()
    print("Reducing dataset to a single batch for overfitting test")
    test_dataset = train_dataset.take(1)
    train_dataset = train_dataset.take(1).repeat(32)  # Amortize end-of-epoch overhead

    if pre_train:
        model_to_train = model.vectorizer
        model_save_name = model_name.replace(".keras", ".vectorizer.keras")

        # we need to modify the dataset to only feed the vectorizer
        train_dataset = train_dataset.map(_preprocess_for_vectorizer_dataset)
        test_dataset = test_dataset.map(_preprocess_for_vectorizer_dataset)
    else:
        model_to_train = model
        model_save_name = model_name

    learning_rate = CustomSchedule(D_MODEL, 1.0)
    # learning_rate = LEARNING_RATE
    # optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9, clipvalue=1.0)
    optimizer = keras.optimizers.Adam(learning_rate)

    # Compile the model
    if pre_train:
        model_to_train.compile(
            optimizer=optimizer,
            loss={
                "command": keras.losses.CategoricalCrossentropy(),
                "coord": keras.losses.Huber(delta=10.0),
            },
            loss_weights={
                "command": VECTOR_LOSS_WEIGHT_COMMAND,
                "coord": VECTOR_LOSS_WEIGHT_COORD,
            },
        )
    else:
        model_to_train.compile(
            optimizer=optimizer,
            loss={
                "raster": keras.losses.MeanSquaredError(),
                "command": keras.losses.CategoricalCrossentropy(),
                "coord": keras.losses.MeanSquaredError(),
            },
            loss_weights={
                "raster": RASTER_LOSS_WEIGHT,
                "command": VECTOR_LOSS_WEIGHT_COMMAND,
                "coord": VECTOR_LOSS_WEIGHT_COORD,
            },
        )

    # Setup TensorBoard callback
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, write_images=True, update_freq="batch"
    )
    image_generation_callback = ImageGenerationCallback(
        log_dir, test_dataset, pre_train=pre_train
    )
    svg_generation_callback = SVGGenerationCallback(
        log_dir, test_dataset, pre_train=pre_train
    )
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_save_name,
        save_weights_only=False,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )

    # Train the model
    model_to_train.fit(
        train_dataset,
        epochs=epochs,
        # XXX while checking we can overfit, don't bother with val step
        # validation_data=test_dataset,
        callbacks=[
            tensorboard_callback,
            image_generation_callback,
            svg_generation_callback,
            checkpoint_callback,
        ],
    )
    # Save the model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the Glyph Generator model.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepvecfont3.keras",
        help="Name of the model to save.",
    )
    parser.add_argument(
        "--pre-train",
        action="store_true",
        help="Whether to pre-train the vectorizer.",
    )
    parser.add_argument(
        "--vectorizer_model_name",
        type=str,
        default=None,
        help="Name of the pre-trained vectorizer model to load.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of epochs to train the model.",
    )
    args = parser.parse_args()
    main(
        model_name=args.model_name,
        pre_train=args.pre_train,
        epochs=args.epochs,
        vectorizer_model_name=args.vectorizer_model_name,
    )