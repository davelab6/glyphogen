#!/usr/bin/env python
import tensorflow as tf
import keras
import tqdm
import datetime

from deepvecfont3.model import GlyphGenerator
from deepvecfont3.glyph import EXTENDED_COMMAND_WIDTH, COORDINATE_WIDTH, Glyph
from deepvecfont3.hyperparameters import (
    BATCH_SIZE,
    NUM_GLYPHS,
    MAX_COMMANDS,
    GEN_IMAGE_SIZE,
    STYLE_IMAGE_SIZE,
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

    font_files = list(glob.glob(BASE_DIR + "/*/*.ttf"))[0:50]
    if not font_files:
        raise ValueError(f"No font files found in {BASE_DIR}")

    for font_file_path in tqdm.tqdm(font_files):
        font_file = Path(font_file_path)
        style_image = get_style_image(
            font_file, variation={}
        )  # Assuming no variation for now

        for i, char in enumerate(ALPHABET):
            glyph = Glyph(font_file, ord(char), location={})

            # Rasterize
            raster = glyph.rasterize(GEN_IMAGE_SIZE[0])
            raster = np.expand_dims(raster, axis=-1)  # Add channel dimension

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
    def __init__(self, log_dir, test_dataset, num_images=1):
        super().__init__()
        self.file_writer = tf.summary.create_file_writer(log_dir + "/images")
        self.test_dataset = test_dataset.unbatch().take(num_images)

    def on_epoch_end(self, epoch, logs=None):
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

            with self.file_writer.as_default():
                tf.summary.image(
                    f"True Glyph Raster - Sample {i}",
                    true_raster,
                    step=epoch,
                )
                tf.summary.image(
                    f"Generated Glyph Raster - Sample {i}", generated_raster, step=epoch
                )


def main():
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

    # Create real dataset
    train_dataset, test_dataset = create_real_dataset()

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(),
        raster_loss_fn=keras.losses.MeanSquaredError(),
        vector_command_loss_fn=keras.losses.CategoricalCrossentropy(),
        vector_coord_loss_fn=keras.losses.MeanSquaredError(),
        raster_loss_weight=RASTER_LOSS_WEIGHT,
        vector_loss_weight=VECTOR_LOSS_WEIGHT,
    )

    # Calculate steps per epoch
    num_font_files = len(glob.glob(BASE_DIR + "/*/*.ttf"))
    num_samples = num_font_files * len(ALPHABET)

    train_steps_per_epoch = int(num_samples * 0.8) // BATCH_SIZE  # 80% for training
    validation_steps = int(num_samples * 0.2) // BATCH_SIZE  # 20% for validation

    # Setup TensorBoard callback
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )
    image_generation_callback = ImageGenerationCallback(log_dir, test_dataset)

    # Train the model
    model.fit(
        train_dataset,
        epochs=EPOCHS,
        # steps_per_epoch=train_steps_per_epoch,
        validation_data=test_dataset,
        # validation_steps=validation_steps,
        callbacks=[tensorboard_callback, image_generation_callback],
    )


if __name__ == "__main__":
    main()
