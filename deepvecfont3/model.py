#!/usr/bin/env python
import keras
from keras import layers, ops
import tensorflow as tf

from deepvecfont3.embedding import StyleEmbedding


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


from deepvecfont3.transformers import TransformerDecoder, TransformerEncoder


def create_look_ahead_mask(size):
    mask = 1 - ops.triu(ops.ones((size, size)))
    return mask


# This model is used for pre-training the vectorizer independently.
@keras.saving.register_keras_serializable()
class VectorizationGenerator(keras.Model):
    def __init__(
        self,
        num_transformer_layers,
        d_model,
        num_heads,
        dff,
        latent_dim=32,
        rate=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_transformer_layers = num_transformer_layers
        self.num_heads = num_heads
        self.dff = dff
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

        self.transformer_decoder = TransformerDecoder(
            num_layers=num_transformer_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            rate=rate,
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
        self.vector_command_loss_tracker = keras.metrics.Mean(
            name="vector_command_loss"
        )
        self.vector_coord_loss_tracker = keras.metrics.Mean(name="vector_coord_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.vector_command_loss_tracker,
            self.vector_coord_loss_tracker,
        ]

    def call(self, inputs, training=False):
        raster_image_input, target_sequence_input = inputs
        z = self.encode(raster_image_input)
        z = ops.expand_dims(z, 1)

        look_ahead_mask = create_look_ahead_mask(ops.shape(target_sequence_input)[1])
        command_output, coord_output = self.transformer_decoder(
            target_sequence_input,
            context=z,
            look_ahead_mask=look_ahead_mask,
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

            vector_command_loss = self.loss["command"](
                true_command, outputs["command"]
            )
            vector_coord_loss = self.loss["coord"](true_coord, outputs["coord"])

            total_loss = vector_command_loss + vector_coord_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.vector_command_loss_tracker.update_state(vector_command_loss)
        self.vector_coord_loss_tracker.update_state(vector_coord_loss)

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

        vector_command_loss = self.loss["command"](
            true_command, outputs["command"]
        )
        vector_coord_loss = self.loss["coord"](true_coord, outputs["coord"])

        total_loss = vector_command_loss + vector_coord_loss

        self.total_loss_tracker.update_state(total_loss)
        self.vector_command_loss_tracker.update_state(vector_command_loss)
        self.vector_coord_loss_tracker.update_state(vector_coord_loss)

        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_transformer_layers": self.num_transformer_layers,
                "num_heads": self.num_heads,
                "dff": self.dff,
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
        num_transformer_layers,
        d_model,
        num_heads,
        dff,
        latent_dim=32,
        rate=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_glyphs = num_glyphs
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.num_transformer_layers = num_transformer_layers
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.style_embedding = StyleEmbedding(latent_dim)
        self.glyph_id_embedding = layers.Dense(latent_dim, activation="relu")
        self.raster_decoder = Decoder(latent_dim * 2)
        self.vectorizer = VectorizationGenerator(
            num_transformer_layers=num_transformer_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            latent_dim=latent_dim,
            rate=rate,
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
                "num_transformer_layers": self.num_transformer_layers,
                "num_heads": self.num_heads,
                "dff": self.dff,
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
            vector_coord_loss = self.loss["coord"](y["coord"], coord_output)

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
        vector_coord_loss = self.loss["coord"](y["coord"], coord_output)

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
