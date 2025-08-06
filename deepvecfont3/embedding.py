#!/usr/bin/env python
import keras
from keras import layers, ops


@keras.saving.register_keras_serializable()
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a font."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon


@keras.saving.register_keras_serializable()
class StyleEmbedding(layers.Layer):
    """Encoder layer that embeds font glyphs into a latent space."""

    def __init__(self, latent_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim

        # Convolutional layers
        self.conv1 = layers.Conv2D(16, 7, padding="same")
        self.norm1 = layers.LayerNormalization()
        self.relu1 = layers.ReLU()

        self.conv2 = layers.Conv2D(32, 5, strides=2, padding="same")
        self.norm2 = layers.LayerNormalization()
        self.relu2 = layers.ReLU()

        self.conv3 = layers.Conv2D(64, 5, strides=2, padding="same")
        self.norm3 = layers.LayerNormalization()
        self.relu3 = layers.ReLU()

        # Dense layers
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(latent_dim)
        self.norm4 = layers.LayerNormalization()
        self.sigmoid = layers.Activation("sigmoid")
        self.activity_reg = keras.layers.ActivityRegularization(l1=1e-3)

        # Latent space layers
        self.z_mean_layer = layers.Dense(
            latent_dim,
            name="z_mean",
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.Zeros(),
        )
        self.z_mean_norm = layers.LayerNormalization()

        self.z_log_var_layer = layers.Dense(
            latent_dim,
            name="z_log_var",
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.Zeros(),
        )
        self.z_log_var_norm = layers.LayerNormalization()

        self.sampling = Sampling()

    def call(self, style_image):
        # style_image is (batch_size, 40, 168, 1)
        x = self.conv1(style_image)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)

        # Dense layers
        x = self.flatten(x)
        x = self.dense(x)
        x = self.norm4(x)
        x = self.sigmoid(x)
        x = self.activity_reg(x)

        # Latent space encoding
        z_mean = self.z_mean_layer(x)
        z_mean = self.z_mean_norm(z_mean)

        z_log_var = self.z_log_var_layer(x)
        z_log_var = self.z_log_var_norm(z_log_var)

        z = self.sampling([z_mean, z_log_var])  # pyrefly: ignore

        return [z_mean, z_log_var, z]

    def get_config(self):
        config = super().get_config()
        config.update({"latent_dim": self.latent_dim})
        return config
