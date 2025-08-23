#!/usr/bin/env python
import os
import tensorflow as tf
import keras
import datetime
from glyphogen.dataset import get_full_model_data, get_pretrain_data, prepare_data
from glyphogen.callbacks import SVGGenerationCallback, ImageGenerationCallback
from glyphogen.model import GlyphGenerator, VectorizationGenerator
from glyphogen.hyperparameters import (
    BATCH_SIZE,
    NUM_GLYPHS,
    LATENT_DIM,
    D_MODEL,
    RASTER_LOSS_WEIGHT,
    VECTOR_LOSS_WEIGHT_COMMAND,
    VECTOR_LOSS_WEIGHT_COORD,
    RATE,
    EPOCHS,
    LEARNING_RATE,
)


import glob
import numpy as np
from pathlib import Path


def main(
    model_name="glyphogen.keras",
    pre_train=False,
    epochs=EPOCHS,
    vectorizer_model_name=None,
    single_batch=False,
):
    # Load the model if it exists
    if os.path.exists(f"{model_name}") and not vectorizer_model_name:
        model = keras.models.load_model(
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
        vectorizer = keras.models.load_model(
            vectorizer_model_name,
            custom_objects={
                "VectorizationGenerator": VectorizationGenerator,
            },
        )
        model.vectorizer = vectorizer
        print(f"Loaded vectorizer from {vectorizer_model_name}")

    if pre_train:
        train_dataset, test_dataset = get_pretrain_data()
    else:
        train_dataset, test_dataset = get_full_model_data()

    if single_batch:
        print("Reducing dataset to a single batch for overfitting test")
        test_dataset = train_dataset.take(1)
        train_dataset = train_dataset.take(1).repeat(
            32
        )  # Amortize end-of-epoch overhead

    if pre_train:
        model_to_train = model.vectorizer
        model_save_name = model_name.replace(".keras", ".vectorizer.keras")
    else:
        model_to_train = model
        model_save_name = model_name
        train_dataset = train_dataset.map(prepare_data)
        test_dataset = test_dataset.map(prepare_data)

    train_dataset = (
        train_dataset.shuffle(buffer_size=1000)
        .prefetch(tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
    )
    test_dataset = (
        test_dataset.shuffle(buffer_size=1000)
        .prefetch(tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
    )

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LEARNING_RATE, decay_steps=10000, decay_rate=0.9
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    # Compile the model
    if pre_train:
        model_to_train.compile(
            optimizer=optimizer,
            loss={
                "command": keras.losses.CategoricalCrossentropy(),
                "coord": keras.losses.Huber(delta=5.0),
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
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, write_images=True, update_freq="batch"
    )
    image_generation_callback = ImageGenerationCallback(
        log_dir, test_dataset, pre_train=pre_train
    )
    svg_generation_callback = SVGGenerationCallback(
        log_dir, test_dataset, pre_train=pre_train
    )
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=model_save_name,
        save_weights_only=False,
        monitor="val_total_loss",
        mode="min",
        save_best_only=True,
    )

    # Train the model
    model_to_train.fit(
        train_dataset,
        epochs=epochs,
        callbacks=[
            tensorboard_callback,
            image_generation_callback,
            svg_generation_callback,
            checkpoint_callback,
        ],
        **({"validation_data": test_dataset} if not single_batch else {}),
        # Maybe not needed? Keras works it out after epoch 1
        # **({"steps_per_epoch": len(font_files) * len(ALPHABET) // BATCH_SIZE} if pre_train else {})
    )
    # Save the model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the Glyph Generator model.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="glyphogen.keras",
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
    parser.add_argument(
        "--single-batch",
        action="store_true",
        help="Whether to use a single batch for training.",
    )
    args = parser.parse_args()
    main(
        model_name=args.model_name,
        pre_train=args.pre_train,
        epochs=args.epochs,
        single_batch=args.single_batch,
        vectorizer_model_name=args.vectorizer_model_name,
    )
