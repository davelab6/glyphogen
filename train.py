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
from deepvecfont3.glyph import Glyph, NodeGlyph, SVGGlyph, TOKEN_VOCAB_SIZE
from deepvecfont3.hyperparameters import (
    BATCH_SIZE,
    NUM_GLYPHS,
    MAX_COMMANDS,
    MAX_SEQUENCE_LENGTH,
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
    sequences = []
    true_rasters = []

    font_files = list(glob.glob(BASE_DIR + "/*/*.ttf"))
    if LIMIT > 0:
        font_files = font_files[:LIMIT]
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
            svg_glyph = glyph.vectorize()
            node_glyph = svg_glyph.to_node_glyph()
            if not node_glyph.commands:
                continue
            encoded_svg = node_glyph.encode()

            # Pad vector representation or skip
            if encoded_svg.shape[0] > MAX_SEQUENCE_LENGTH:
                continue
            elif encoded_svg.shape[0] < MAX_SEQUENCE_LENGTH:
                padding = np.zeros(MAX_SEQUENCE_LENGTH - encoded_svg.shape[0])
                encoded_svg = np.concatenate((encoded_svg, padding))

            # One-hot encode glyph ID
            glyph_id_one_hot = tf.one_hot(i, NUM_GLYPHS).numpy()

            style_images.append(style_image)
            glyph_ids.append(glyph_id_one_hot)
            sequences.append(encoded_svg)
            true_rasters.append(raster)

    # Convert lists to numpy arrays
    style_images = np.array(style_images)
    glyph_ids = np.array(glyph_ids)
    sequences = np.array(sequences)
    true_rasters = np.array(true_rasters)

    # Split data into training and testing sets
    (
        style_images_train,
        style_images_test,
        glyph_ids_train,
        glyph_ids_test,
        sequences_train,
        sequences_test,
        true_rasters_train,
        true_rasters_test,
    ) = train_test_split(
        style_images,
        glyph_ids,
        sequences,
        true_rasters,
        test_size=0.2,  # 20% for testing
        random_state=42,
    )

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (
            (style_images_train, glyph_ids_train, sequences_train[:, :-1]),
            {
                "raster": true_rasters_train,
                "vector": sequences_train[:, 1:],
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
            (style_images_test, glyph_ids_test, sequences_test[:, :-1]),
            {
                "raster": true_rasters_test,
                "vector": sequences_test[:, 1:],
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

        for i, (inputs, outputs) in enumerate(self.test_dataset):
            style_image, glyph_id, target_sequence = inputs
            true_raster = outputs["raster"]

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
                raster_image_input = tf.expand_dims(raster_image_input, axis=0)
                target_sequence_input = tf.expand_dims(target_sequence_input, axis=0)
                output = model(
                    (raster_image_input, target_sequence_input), training=False
                )
            else:
                (style_image, glyph_id, target_sequence_input) = inputs
                output = model(
                    (style_image, glyph_id, target_sequence_input), training=False
                )["vector"]

            # a single batch
            token_sequence = np.argmax(output[0], axis=-1)

            try:
                decoded_glyph = NodeGlyph.from_numpy(token_sequence)
                try:
                    svg_string = decoded_glyph.to_svg_glyph().to_svg_string()
                except Exception:
                    svg_string = "DEBUG " + decoded_glyph.to_debug_string()
            except Exception:
                svg_string = "ERROR " + " ".join(map(str, token_sequence))

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
    return ((y["raster"], x[2]), y["vector"])


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
    optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9, clipvalue=1.0)

    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Compile the model
    if pre_train:
        model_to_train.compile(
            optimizer=optimizer,
            loss=loss,
        )
    else:
        model_to_train.compile(
            optimizer=optimizer,
            loss={
                "raster": keras.losses.MeanSquaredError(),
                "vector": loss,
            },
            loss_weights={
                "raster": RASTER_LOSS_WEIGHT,
                "vector": VECTOR_LOSS_WEIGHT,
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