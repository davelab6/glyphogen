#!/usr/bin/env python
import torch
from torch import nn
import torch.nn.functional as F

from glyphogen_torch.embedding import StyleEmbedding
from glyphogen_torch.lstm import LSTMDecoder

# from glyphogen_torch.rasterizer import rasterize_batch # TODO: Port this

from glyphogen.glyph import NODE_GLYPH_COMMANDS, COORDINATE_WIDTH


def calculate_masked_coordinate_loss(
    y_true_command, y_true_coord, y_pred_coord, arg_counts, delta=10.0
):
    """Calculates the masked coordinate loss."""
    true_command_indices = torch.argmax(y_true_command, axis=-1)
    args_needed = arg_counts[true_command_indices]

    max_args = arg_counts.max()
    coord_mask = (
        torch.arange(max_args, device=args_needed.device)[None, None, :]
        < args_needed[..., None]
    )
    coord_mask = coord_mask.to(y_pred_coord.dtype)

    huber_loss = F.huber_loss(y_pred_coord, y_true_coord, reduction="none", delta=delta)
    masked_loss = huber_loss * coord_mask

    return torch.sum(masked_loss) / torch.sum(coord_mask)


class Decoder(nn.Module):
    """Decodes a latent vector into a raster image."""

    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim

        self.dense = nn.Linear(latent_dim, 16 * 16 * 256)

        self.deconv1 = nn.ConvTranspose2d(
            256, 256, 3, stride=2, padding=1, output_padding=1
        )
        self.norm1 = nn.LayerNorm([256, 32, 32])
        self.relu1 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(
            256, 128, 3, stride=2, padding=1, output_padding=1
        )
        self.norm2 = nn.LayerNorm([128, 64, 64])
        self.relu2 = nn.ReLU()

        self.deconv3 = nn.ConvTranspose2d(
            128, 64, 5, stride=2, padding=2, output_padding=1
        )
        self.norm3 = nn.LayerNorm([64, 128, 128])
        self.relu3 = nn.ReLU()

        self.deconv4 = nn.ConvTranspose2d(
            64, 32, 5, stride=2, padding=2, output_padding=1
        )
        self.norm4 = nn.LayerNorm([32, 256, 256])
        self.relu4 = nn.ReLU()

        self.deconv5 = nn.ConvTranspose2d(
            32, 16, 7, stride=2, padding=3, output_padding=1
        )
        self.norm5 = nn.LayerNorm([16, 512, 512])
        self.relu5 = nn.ReLU()

        self.output_conv = nn.Conv2d(16, 1, 7, padding="same")
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.dense(inputs)
        x = x.view(-1, 256, 16, 16)

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

        x = self.output_conv(x)
        return self.sigmoid(x)


class VectorizationGenerator(nn.Module):
    def __init__(self, d_model, latent_dim=32, rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.rate = rate

        self.conv1 = nn.Conv2d(1, 16, 7, padding=3, stride=2)
        self.norm1 = nn.LayerNorm([16, 256, 256])
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2, stride=2)
        self.norm2 = nn.LayerNorm([32, 128, 128])
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2, stride=2)
        self.norm3 = nn.LayerNorm([64, 64, 64])
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.norm4 = nn.LayerNorm([128, 32, 32])
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1, stride=2)
        self.norm5 = nn.LayerNorm([256, 16, 16])
        self.relu5 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(256 * 16 * 16, latent_dim)
        self.norm_dense = nn.LayerNorm(latent_dim)
        self.sigmoid = nn.Sigmoid()
        self.output_dense = nn.Linear(latent_dim, latent_dim)

        self.decoder = torch.compile(
            LSTMDecoder(d_model=d_model, latent_dim=latent_dim, rate=rate)
        )
        self.arg_counts = torch.tensor(
            list(NODE_GLYPH_COMMANDS.values()), dtype=torch.long
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

    def forward(self, inputs):
        raster_image_input, target_sequence_input = inputs
        z = self.encode(raster_image_input)
        z = z.unsqueeze(1)

        command_output, coord_output = self.decoder(target_sequence_input, context=z)
        return {"command": command_output, "coord": coord_output}


class GlyphGenerator(nn.Module):
    """Generates a glyph raster image from a style reference and a glyph ID."""

    def __init__(self, num_glyphs, d_model, latent_dim=32, rate=0.1):
        super().__init__()
        self.num_glyphs = num_glyphs
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.rate = rate

        self.style_embedding = StyleEmbedding(latent_dim)
        self.glyph_id_embedding = nn.Linear(num_glyphs, latent_dim)
        self.raster_decoder = Decoder(latent_dim * 2)
        self.vectorizer = VectorizationGenerator(
            d_model=d_model, latent_dim=latent_dim, rate=rate
        )
        self.arg_counts = torch.tensor(
            list(NODE_GLYPH_COMMANDS.values()), dtype=torch.long
        )

    def forward(self, inputs):
        style_image_input, glyph_id_input, target_sequence_input = inputs

        _, _, z = self.style_embedding(style_image_input)

        glyph_id_embedded = self.glyph_id_embedding(glyph_id_input)
        combined = torch.cat([z, glyph_id_embedded], dim=-1)
        generated_glyph_raster = self.raster_decoder(combined)

        vectorizer_output = self.vectorizer(
            (generated_glyph_raster, target_sequence_input)
        )
        command_output = vectorizer_output["command"]
        coord_output = vectorizer_output["coord"]

        return {
            "raster": generated_glyph_raster,
            "command": command_output,
            "coord": coord_output,
        }


def build_model(num_glyphs, d_model, latent_dim=32, rate=0.1):
    return GlyphGenerator(
        num_glyphs=num_glyphs,
        d_model=d_model,
        latent_dim=latent_dim,
        rate=rate,
    )
