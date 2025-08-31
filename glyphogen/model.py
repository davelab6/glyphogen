#!/usr/bin/env python
import keras
import tensorflow as tf
from keras import layers, ops

from glyphogen.embedding import StyleEmbedding
from glyphogen.hyperparameters import (
    VECTOR_LOSS_WEIGHT_COORD,
    VECTOR_LOSS_WEIGHT_COMMAND,
    RASTER_LOSS_WEIGHT,
)
from glyphogen.glyph import NODE_GLYPH_COMMANDS, COORDINATE_WIDTH
from glyphogen.transformers import LSTMDecoder
from glyphogen.rasterizer import rasterize_batch
import datetime

# Set up summary writers
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = "logs/gradient_tape/" + current_time + "/train"
test_log_dir = "logs/gradient_tape/" + current_time + "/test"
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


def _calculate_masked_coordinate_loss(
    y_true_command, y_true_coord, y_pred_coord, arg_counts, delta=10.0
):
    """Calculates the masked coordinate loss."""
    true_command_indices = tf.argmax(y_true_command, axis=-1)
    args_needed = tf.gather(arg_counts, true_command_indices)
    # Create a mask for the coordinates
    # For each item in the batch, for each command in the sequence, this is
    # a vector of COORDINATE_WIDTH length, with 1s for the arguments
    # that are used and 0s for those that are not.
    coord_mask = tf.sequence_mask(args_needed, COORDINATE_WIDTH, dtype=tf.float32)

    # Now compute the Huber loss
    y_true_coord = tf.cast(y_true_coord, tf.float32)
    residuals = y_true_coord - y_pred_coord
    abs_error = ops.abs(residuals)
    half = ops.convert_to_tensor(0.5, dtype=abs_error.dtype)
    huber_error = ops.where(
        abs_error <= delta,
        half * ops.square(residuals),
        delta * abs_error - half * ops.square(delta),
    )
    masked_error = huber_error * coord_mask
    return tf.reduce_sum(masked_error) / tf.reduce_sum(coord_mask)


@keras.saving.register_keras_serializable()
class Decoder(layers.Layer):
    """Decodes a latent vector into a raster image."""

    def __init__(self, latent_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim

        # Dense layer to project latent vector to a suitable shape for deconvolution
        self.dense = layers.Dense(16 * 16 * 256, activation="relu")
        self.reshape = layers.Reshape((16, 16, 256))

        # Deconvolutional layers
        self.deconv1 = layers.Conv2DTranspose(
            256, 3, strides=2, padding="same"
        )  # 16 -> 32
        self.norm1 = layers.LayerNormalization()
        self.relu1 = layers.ReLU()

        self.deconv2 = layers.Conv2DTranspose(
            128, 3, strides=2, padding="same"
        )  # 32 -> 64
        self.norm2 = layers.LayerNormalization()
        self.relu2 = layers.ReLU()

        self.deconv3 = layers.Conv2DTranspose(
            64, 5, strides=2, padding="same"
        )  # 64 -> 128
        self.norm3 = layers.LayerNormalization()
        self.relu3 = layers.ReLU()

        self.deconv4 = layers.Conv2DTranspose(
            32, 5, strides=2, padding="same"
        )  # 128 -> 256
        self.norm4 = layers.LayerNormalization()
        self.relu4 = layers.ReLU()

        self.deconv5 = layers.Conv2DTranspose(
            16, 7, strides=2, padding="same"
        )  # 256 -> 512
        self.norm5 = layers.LayerNormalization()
        self.relu5 = layers.ReLU()

        # Output layer
        self.output_conv = layers.Conv2D(1, 7, activation="sigmoid", padding="same")

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.reshape(x)

        x = self.deconv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        x = self.deconv3(x)
        x = self.norm3(x)
        x = self.relu3(x)

        x = self.deconv4(x)
        x = self.norm4(x)
        x = self.relu4(x)

        x = self.deconv5(x)
        x = self.norm5(x)
        x = self.relu5(x)

        return self.output_conv(x)

    def get_config(self):
        config = super().get_config()
        config.update({"latent_dim": self.latent_dim})
        return config


# This model is used for pre-training the vectorizer independently.
@keras.saving.register_keras_serializable()
class VectorizationGenerator(keras.Model):
    def __init__(
        self,
        d_model,
        latent_dim=32,
        rate=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.rate = rate

        # Convolutional Encoder for raster images
        self.conv1 = layers.Conv2D(16, 7, padding="same", strides=2)  # 512 -> 256
        self.norm1 = layers.LayerNormalization()
        self.relu1 = layers.ReLU()
        self.conv2 = layers.Conv2D(32, 5, padding="same", strides=2)  # 256 -> 128
        self.norm2 = layers.LayerNormalization()
        self.relu2 = layers.ReLU()
        self.conv3 = layers.Conv2D(64, 5, padding="same", strides=2)  # 128 -> 64
        self.norm3 = layers.LayerNormalization()
        self.relu3 = layers.ReLU()
        self.conv4 = layers.Conv2D(128, 3, padding="same", strides=2)  # 64 -> 32
        self.norm4 = layers.LayerNormalization()
        self.relu4 = layers.ReLU()
        self.conv5 = layers.Conv2D(256, 3, padding="same", strides=2)  # 32 -> 16
        self.norm5 = layers.LayerNormalization()
        self.relu5 = layers.ReLU()
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(latent_dim)
        self.norm_dense = layers.LayerNormalization()
        self.sigmoid = layers.Activation("sigmoid")
        self.output_dense = layers.Dense(latent_dim)

        self.decoder = LSTMDecoder(
            d_model=d_model,
            rate=rate,
        )
        self.arg_counts = tf.constant(
            list(NODE_GLYPH_COMMANDS.values()), dtype=tf.int32
        )

    def encode(self, inputs):
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.norm4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.norm5(x)
        x = self.relu5(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.norm_dense(x)
        x = self.sigmoid(x)
        z = self.output_dense(x)
        return z

    def compile(
        self,
        optimizer,
        loss,
        **kwargs,
    ):
        super().compile(optimizer=optimizer, **kwargs)
        self.loss = loss
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.raster_loss_tracker = keras.metrics.Mean(name="raster_loss")
        self.vector_command_loss_tracker = keras.metrics.Mean(
            name="vector_command_loss"
        )
        self.vector_coord_loss_tracker = keras.metrics.Mean(name="vector_coord_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.raster_loss_tracker,
            self.vector_command_loss_tracker,
            self.vector_coord_loss_tracker,
        ]

    def call(self, inputs, training=False):
        raster_image_input, target_sequence_input = inputs
        z = self.encode(raster_image_input)
        z = ops.expand_dims(z, 1)

        command_output, coord_output = self.decoder(
            target_sequence_input,
            context=z,
            training=training,
        )
        return {"command": command_output, "coord": coord_output}

    def train_step(self, data):
        (
            raster_image_input,
            target_sequence_input,
        ), (
            true_command,
            true_coord,
        ) = data

        with tf.GradientTape() as tape:
            outputs = self(
                (raster_image_input, target_sequence_input),
                training=True,
            )

            # Rasterize the predicted vectors
            vector_rendered_images = rasterize_batch(
                outputs["command"], outputs["coord"]
            )

            raster_loss = self.loss["raster"](
                raster_image_input, vector_rendered_images
            )
            vector_command_loss = self.loss["command"](true_command, outputs["command"])
            vector_coord_loss = _calculate_masked_coordinate_loss(
                true_command, true_coord, outputs["coord"], self.arg_counts
            )

            total_loss = (
                vector_command_loss * VECTOR_LOSS_WEIGHT_COMMAND
                + vector_coord_loss * VECTOR_LOSS_WEIGHT_COORD
                + raster_loss * RASTER_LOSS_WEIGHT
            )

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.raster_loss_tracker.update_state(raster_loss)
        self.vector_command_loss_tracker.update_state(vector_command_loss)
        self.vector_coord_loss_tracker.update_state(vector_coord_loss)

        with train_summary_writer.as_default():
            tf.summary.scalar(
                "total_loss",
                self.total_loss_tracker.result(),
                step=self.optimizer.iterations,
            )
            tf.summary.scalar(
                "raster_loss",
                self.raster_loss_tracker.result(),
                step=self.optimizer.iterations,
            )
            tf.summary.scalar(
                "vector_command_loss",
                self.vector_command_loss_tracker.result(),
                step=self.optimizer.iterations,
            )
            tf.summary.scalar(
                "vector_coord_loss",
                self.vector_coord_loss_tracker.result(),
                step=self.optimizer.iterations,
            )
            if self.optimizer.iterations % 100 == 0:
                tf.summary.image(
                    "True Rasters",
                    raster_image_input,
                    max_outputs=3,
                    step=self.optimizer.iterations,
                )
                tf.summary.image(
                    "Vector-Rendered Rasters",
                    vector_rendered_images,
                    max_outputs=3,
                    step=self.optimizer.iterations,
                )

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        (
            raster_image_input,
            target_sequence_input,
        ), (
            true_command,
            true_coord,
        ) = data

        outputs = self(
            (raster_image_input, target_sequence_input),
            training=False,
        )

        vector_command_loss = self.loss["command"](true_command, outputs["command"])
        vector_coord_loss = _calculate_masked_coordinate_loss(
            true_command, true_coord, outputs["coord"], self.arg_counts
        )

        total_loss = (
            vector_command_loss * VECTOR_LOSS_WEIGHT_COMMAND
            + vector_coord_loss * VECTOR_LOSS_WEIGHT_COORD
        )

        self.total_loss_tracker.update_state(total_loss)
        self.vector_command_loss_tracker.update_state(vector_command_loss)
        self.vector_coord_loss_tracker.update_state(vector_coord_loss)

        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "latent_dim": self.latent_dim,
                "rate": self.rate,
            }
        )
        return config


class GlyphGenerator(keras.Model):
    """Generates a glyph raster image from a style reference and a glyph ID."""

    def __init__(
        self,
        num_glyphs,
        d_model,
        latent_dim=32,
        rate=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_glyphs = num_glyphs
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.rate = rate

        self.style_embedding = StyleEmbedding(latent_dim)
        self.glyph_id_embedding = layers.Dense(latent_dim, activation="relu")
        self.raster_decoder = Decoder(latent_dim * 2)
        self.vectorizer = VectorizationGenerator(
            d_model=d_model,
            latent_dim=latent_dim,
            rate=rate,
        )
        self.arg_counts = tf.constant(
            list(NODE_GLYPH_COMMANDS.values()), dtype=tf.int32
        )

    def call(self, inputs, training=False):
        style_image_input, glyph_id_input, target_sequence_input = inputs

        # Get style embedding
        _, _, z = self.style_embedding(style_image_input)

        # Raster generation
        glyph_id_embedded = self.glyph_id_embedding(glyph_id_input)
        combined = ops.concatenate([z, glyph_id_embedded], axis=-1)
        generated_glyph_raster = self.raster_decoder(combined)

        # Vector generation
        vectorizer_output = self.vectorizer(
            (generated_glyph_raster, target_sequence_input), training=training
        )
        command_output = vectorizer_output["command"]
        coord_output = vectorizer_output["coord"]

        return {
            "raster": generated_glyph_raster,
            "command": command_output,
            "coord": coord_output,
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_glyphs": self.num_glyphs,
                "latent_dim": self.latent_dim,
                "d_model": self.d_model,
                "rate": self.rate,
            }
        )
        return config

    def compile(
        self,
        optimizer,
        loss,
        loss_weights,
        **kwargs,
    ):
        super().compile(optimizer=optimizer, **kwargs)
        self.loss = loss
        self.loss_weights = loss_weights
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.raster_loss_tracker = keras.metrics.Mean(name="raster_loss")
        self.vector_command_loss_tracker = keras.metrics.Mean(
            name="vector_command_loss"
        )
        self.vector_coord_loss_tracker = keras.metrics.Mean(name="vector_coord_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.raster_loss_tracker,
            self.vector_command_loss_tracker,
            self.vector_coord_loss_tracker,
        ]

    def train_step(self, data):
        (
            style_image_input,
            glyph_id_input,
            target_sequence_input,
        ), y = data

        with tf.GradientTape() as tape:
            outputs = self(
                (style_image_input, glyph_id_input, target_sequence_input),
                training=True,
            )
            generated_glyph_raster = outputs["raster"]
            command_output = outputs["command"]
            coord_output = outputs["coord"]

            raster_loss = self.loss["raster"](y["raster"], generated_glyph_raster)
            vector_command_loss = self.loss["command"](y["command"], command_output)

            vector_coord_loss = _calculate_masked_coordinate_loss(
                y["command"], y["coord"], coord_output, self.arg_counts
            )

            total_loss = (
                self.loss_weights["raster"] * raster_loss
                + self.loss_weights["command"] * vector_command_loss
                + self.loss_weights["coord"] * vector_coord_loss
            )

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.raster_loss_tracker.update_state(raster_loss)
        self.vector_command_loss_tracker.update_state(vector_command_loss)
        self.vector_coord_loss_tracker.update_state(vector_coord_loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        (
            style_image_input,
            glyph_id_input,
            target_sequence_input,
        ), y = data

        outputs = self(
            (style_image_input, glyph_id_input, target_sequence_input),
            training=False,
        )
        generated_glyph_raster = outputs["raster"]
        command_output = outputs["command"]
        coord_output = outputs["coord"]

        raster_loss = self.loss["raster"](y["raster"], generated_glyph_raster)
        vector_command_loss = self.loss["command"](y["command"], command_output)

        vector_coord_loss = _calculate_masked_coordinate_loss(
            y["command"], y["coord"], coord_output, self.arg_counts
        )

        total_loss = (
            self.loss_weights["raster"] * raster_loss
            + self.loss_weights["command"] * vector_command_loss
            + self.loss_weights["coord"] * vector_coord_loss
        )

        self.total_loss_tracker.update_state(total_loss)
        self.raster_loss_tracker.update_state(raster_loss)
        self.vector_command_loss_tracker.update_state(vector_command_loss)
        self.vector_coord_loss_tracker.update_state(vector_coord_loss)

        return {m.name: m.result() for m in self.metrics}


def build_model(
    num_glyphs,
    d_model,
    latent_dim=32,
    rate=0.1,
):
    return GlyphGenerator(
        num_glyphs=num_glyphs,
        d_model=d_model,
        latent_dim=latent_dim,
        rate=rate,
    )
