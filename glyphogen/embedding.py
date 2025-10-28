#!/usr/bin/env python
import torch
from torch import nn


class Sampling(nn.Module):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a font."""

    def forward(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


class StyleEmbedding(nn.Module):
    """Encoder layer that embeds font glyphs into a latent space."""

    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, 7, padding="same")
        self.norm1 = nn.LayerNorm([16, 40, 168])
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.norm2 = nn.LayerNorm([32, 20, 84])
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(32, 64, 5, stride=2, padding=2)
        self.norm3 = nn.LayerNorm([64, 10, 42])
        self.relu3 = nn.ReLU()

        # Dense layers
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(64 * 10 * 42, latent_dim)
        self.norm4 = nn.LayerNorm(latent_dim)
        self.sigmoid = nn.Sigmoid()
        # ActivityRegularization is handled in the training loop

        # Latent space layers
        self.z_mean_layer = nn.Linear(latent_dim, latent_dim)
        nn.init.zeros_(self.z_mean_layer.weight)
        nn.init.zeros_(self.z_mean_layer.bias)
        self.z_mean_norm = nn.LayerNorm(latent_dim)

        self.z_log_var_layer = nn.Linear(latent_dim, latent_dim)
        nn.init.zeros_(self.z_log_var_layer.weight)
        nn.init.zeros_(self.z_log_var_layer.bias)
        self.z_log_var_norm = nn.LayerNorm(latent_dim)

        self.sampling = Sampling()

    def forward(self, style_image):
        # style_image is (batch_size, 1, 40, 168)
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
        # We will need to add L1 activity regularization in the training loop
        # on the output of the next layer.
        x = self.dense(x)
        self.last_dense_output = x
        x = self.norm4(x)
        x = self.sigmoid(x)

        # Latent space encoding
        z_mean = self.z_mean_layer(x)
        z_mean = self.z_mean_norm(z_mean)

        z_log_var = self.z_log_var_layer(x)
        z_log_var = self.z_log_var_norm(z_log_var)

        z = self.sampling((z_mean, z_log_var))

        return z_mean, z_log_var, z
