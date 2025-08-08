#!/usr/bin/env python
import os
import tensorflow as tf
import keras
import tqdm
import datetime


from fontTools.ttLib import TTFont
from deepvecfont3.model import GlyphGenerator, VectorizationGenerator
from deepvecfont3.glyph import EXTENDED_COMMAND_WIDTH, Glyph
from deepvecfont3.hyperparameters import (
    BATCH_SIZE,
    NUM_GLYPHS,
    MAX_COMMANDS,
    GEN_IMAGE_SIZE,
    LATENT_DIM,
    NUM_TRANSFORMER_LAYERS,
    D_MODEL,
    NUM_HEADS,
    RASTER_LOSS_WEIGHT,
    VECTOR_LOSS_WEIGHT,
    DFF,
    RATE,
    EPOCHS,
    BASE_DIR,
    ALPHABET,
)
from deepvecfont3.rendering import get_style_image


import glob
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def create_real_dataset():
    style_images = []
    glyph_ids = []
    target_sequences = []
    true_rasters = []
    true_commands = []
    true_coords = []

    font_files = list(glob.glob(BASE_DIR + "/*/*.ttf"))
    if not font_files:
        raise ValueError(f"No font files found in {BASE_DIR}")

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
            unrelaxed_svg = glyph.vectorize()
            relaxed_svg = unrelaxed_svg.relax()
            if not relaxed_svg.commands:
                continue
            encoded_svg = relaxed_svg.encode()

            # Pad or truncate vector representation
            if encoded_svg.shape[0] > MAX_COMMANDS:
                encoded_svg = encoded_svg[:MAX_COMMANDS, :]
            elif encoded_svg.shape[0] < MAX_COMMANDS:
                padding = np.zeros(
                    (MAX_COMMANDS - encoded_svg.shape[0], encoded_svg.shape[1])
                )
                encoded_svg = np.vstack((encoded_svg, padding))

            # Split into command and coordinate parts
            command_part = encoded_svg[:, :EXTENDED_COMMAND_WIDTH]
            coord_part = encoded_svg[:, EXTENDED_COMMAND_WIDTH:]

            # One-hot encode glyph ID
            glyph_id_one_hot = tf.one_hot(i, NUM_GLYPHS).numpy()

            style_images.append(style_image)
            glyph_ids.append(glyph_id_one_hot)
            target_sequences.append(encoded_svg)
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
            (true_rasters_train, true_commands_train, true_coords_train),
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
            (true_rasters_test, true_commands_test, true_coords_test),
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
            generated_raster, _, _ = self.model(
                (style_image, glyph_id, target_sequence)
            )
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
                "raster_loss_fn": keras.losses.MeanSquaredError(),
                "vector_command_loss_fn": keras.losses.CategoricalCrossentropy(),
                "vector_coord_loss_fn": keras.losses.MeanSquaredError(),
            },
        )
        print(f"Loaded model from {model_name}")
    else:
        # Create the model
        model = GlyphGenerator(
            num_glyphs=NUM_GLYPHS,
            num_transformer_layers=NUM_TRANSFORMER_LAYERS,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            dff=DFF,
            latent_dim=LATENT_DIM,
            rate=RATE,
        )

    if vectorizer_model_name:
        vectorizer = tf.keras.models.load_model(
            vectorizer_model_name,
            custom_objects={
                "VectorizationGenerator": VectorizationGenerator,
                "vector_command_loss_fn": keras.losses.CategoricalCrossentropy(),
                "vector_coord_loss_fn": keras.losses.MeanSquaredError(),
            },
        )
        model.vectorizer = vectorizer
        print(f"Loaded vectorizer from {vectorizer_model_name}")

    train_dataset, test_dataset = get_data()

    if pre_train:
        model_to_train = model.vectorizer
        model_save_name = model_name.replace(".keras", ".vectorizer.keras")

        # we need to modify the dataset to only feed the vectorizer
        train_dataset = train_dataset.map(lambda x, y: ((y[0], x[2]), (y[1], y[2])))
        test_dataset = test_dataset.map(lambda x, y: ((y[0], x[2]), (y[1], y[2])))
    else:
        model_to_train = model
        model_save_name = model_name

    # Compile the model
    if pre_train:
        model_to_train.compile(
            optimizer=keras.optimizers.Adam(),
            vector_command_loss_fn=keras.losses.CategoricalCrossentropy(),
            vector_coord_loss_fn=keras.losses.MeanSquaredError(),
        )
    else:
        model_to_train.compile(
            optimizer=keras.optimizers.Adam(),
            raster_loss_fn=keras.losses.MeanSquaredError(),
            vector_command_loss_fn=keras.losses.CategoricalCrossentropy(),
            vector_coord_loss_fn=keras.losses.MeanSquaredError(),
            raster_loss_weight=RASTER_LOSS_WEIGHT,
            vector_loss_weight=VECTOR_LOSS_WEIGHT,
        )

    # Setup TensorBoard callback
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, write_images=True, update_freq="batch"
    )
    image_generation_callback = ImageGenerationCallback(
        log_dir, test_dataset, pre_train=pre_train
    )

    # Train the model
    model_to_train.fit(
        train_dataset,
        epochs=epochs,
        # steps_per_epoch=train_steps_per_epoch,
        validation_data=test_dataset,
        # validation_steps=validation_steps,
        callbacks=[tensorboard_callback, image_generation_callback],
    )
    # Save the model
    model_to_train.save(model_save_name)
    print(f"Model saved to {model_save_name}")


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
