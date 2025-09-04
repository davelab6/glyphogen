#!/usr/bin/env python
from glyphogen_torch.hyperparameters import VECTOR_RASTERIZATION_LOSS_WEIGHT
import torch
from torch import nn
import torch.nn.functional as F

from glyphogen_torch.embedding import StyleEmbedding
from glyphogen_torch.lstm import LSTMDecoder
from glyphogen_torch.rasterizer import rasterize_batch

from glyphogen.glyph import NODE_GLYPH_COMMANDS, COORDINATE_WIDTH

from glyphogen_torch.hyperparameters import (
    RASTER_LOSS_WEIGHT,
    VECTOR_LOSS_WEIGHT_COMMAND,
    VECTOR_LOSS_WEIGHT_COORD,
    CONTOUR_COUNT_WEIGHT,
    NODE_COUNT_WEIGHT,
    HANDLE_SMOOTHNESS_WEIGHT,
)

SKIP_RASTERIZATION = True


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


def point_placement_loss(y_true_command, y_pred_command):
    """
    Calculates a loss based on drawing practices.
    """
    command_keys = list(NODE_GLYPH_COMMANDS.keys())
    z_index = command_keys.index("Z")
    eos_index = command_keys.index("EOS")
    n_index = command_keys.index("N")
    nh_index = command_keys.index("NH")
    nv_index = command_keys.index("NV")

    y_true_command_indices = torch.argmax(y_true_command, axis=-1)
    y_pred_command_indices = torch.argmax(y_pred_command, axis=-1)

    # Find EOS indices
    true_eos_mask = y_true_command_indices == eos_index
    pred_eos_mask = y_pred_command_indices == eos_index

    # If no EOS, use sequence length
    true_eos_idx = torch.where(
        torch.any(true_eos_mask, dim=1),
        torch.argmax(true_eos_mask.float(), dim=1),
        torch.full(
            (y_true_command.shape[0],),
            y_true_command.shape[1],
            device=y_true_command.device,
            dtype=torch.long,
        ),
    )
    pred_eos_idx = torch.where(
        torch.any(pred_eos_mask, dim=1),
        torch.argmax(pred_eos_mask.float(), dim=1),
        torch.full(
            (y_pred_command.shape[0],),
            y_pred_command.shape[1],
            device=y_pred_command.device,
            dtype=torch.long,
        ),
    )

    # Contour count loss
    batch_size = y_true_command.shape[0]

    true_z_counts = []
    for i in range(batch_size):
        true_seq = y_true_command_indices[i, : true_eos_idx[i]]
        true_z_counts.append(torch.sum(true_seq == z_index))
    true_z_counts = torch.stack(true_z_counts).float()

    pred_z_counts = []
    for i in range(batch_size):
        pred_seq = y_pred_command_indices[i, : pred_eos_idx[i]]
        pred_z_counts.append(torch.sum(pred_seq == z_index))
    pred_z_counts = torch.stack(pred_z_counts).float()

    contour_loss = torch.mean(torch.abs(true_z_counts - pred_z_counts))

    # Sequence length loss; try to do it in same number of nodes as the designer
    eos_loss = torch.mean(torch.abs(pred_eos_idx.float() - true_eos_idx.float()))

    # Handle orientation loss
    pred_n_prob = torch.mean(y_pred_command[:, :, n_index])
    pred_nh_prob = torch.mean(y_pred_command[:, :, nh_index])
    pred_nv_prob = torch.mean(y_pred_command[:, :, nv_index])
    handle_loss = pred_n_prob - pred_nh_prob - pred_nv_prob

    return contour_loss, eos_loss, handle_loss


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

        self.raster_loss_fn = torch.nn.MSELoss()
        self.command_loss_fn = torch.nn.CrossEntropyLoss()

    @torch.compile(backend="aot_eager")
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

    def step(self, batch, step):
        device = next(self.parameters()).device
        (inputs, y) = batch
        (raster_image_input, target_sequence_input) = inputs
        raster_image_input, target_sequence_input = raster_image_input.to(
            device
        ), target_sequence_input.to(device)
        outputs = self((raster_image_input, target_sequence_input))
        losses = self.losses(y, raster_image_input, outputs, step)
        return losses

    def losses(self, y, raster_image_input, outputs, step):
        device = next(self.parameters()).device
        (true_command, true_coord) = y
        true_command, true_coord = true_command.to(device), true_coord.to(device)
        if VECTOR_RASTERIZATION_LOSS_WEIGHT == 0.0:
            raster_loss = self.raster_loss_fn(raster_image_input, raster_image_input)
        else:
            vector_rendered_images = rasterize_batch(
                outputs["command"], outputs["coord"], seed=step
            ).to(device)
            raster_loss = self.raster_loss_fn(
                raster_image_input, vector_rendered_images
            )

        command_loss = self.command_loss_fn(outputs["command"], true_command)
        coord_loss = calculate_masked_coordinate_loss(
            true_command, true_coord, outputs["coord"], self.arg_counts.to(device)
        )
        (
            point_placement_contour_loss,
            point_placement_eos_loss,
            point_placement_handle_loss,
        ) = point_placement_loss(true_command, outputs["command"])

        total_loss = (
            VECTOR_RASTERIZATION_LOSS_WEIGHT * raster_loss
            + VECTOR_LOSS_WEIGHT_COMMAND * command_loss
            + VECTOR_LOSS_WEIGHT_COORD * coord_loss
            + CONTOUR_COUNT_WEIGHT * point_placement_contour_loss
            + NODE_COUNT_WEIGHT * point_placement_eos_loss
            + HANDLE_SMOOTHNESS_WEIGHT * point_placement_handle_loss
        )
        return {
            "total_loss": total_loss,
            "raster_loss": VECTOR_RASTERIZATION_LOSS_WEIGHT * raster_loss,
            "command_loss": VECTOR_LOSS_WEIGHT_COMMAND * command_loss,
            "coord_loss": VECTOR_LOSS_WEIGHT_COORD * coord_loss,
            "point_placement_contour_loss": CONTOUR_COUNT_WEIGHT
            * point_placement_contour_loss,
            "point_placement_eos_loss": NODE_COUNT_WEIGHT * point_placement_eos_loss,
            "point_placement_handle_loss": HANDLE_SMOOTHNESS_WEIGHT
            * point_placement_handle_loss,
        }

    @torch.compile(backend="aot_eager")
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

    def training_step(self, batch, raster_loss_fn, command_loss_fn):
        device = next(self.parameters()).device
        (style_image_input, glyph_id_input, target_sequence_input), y = batch
        style_image_input, glyph_id_input, target_sequence_input = (
            style_image_input.to(device),
            glyph_id_input.to(device),
            target_sequence_input.to(device),
        )
        y = {k: v.to(device) for k, v in y.items()}

        outputs = self((style_image_input, glyph_id_input, target_sequence_input))
        raster_loss = raster_loss_fn(outputs["raster"], y["raster"])
        true_command, true_coord = y["command"], y["coord"]

        command_loss = command_loss_fn(outputs["command"], true_command)
        coord_loss = calculate_masked_coordinate_loss(
            true_command, true_coord, outputs["coord"], self.arg_counts.to(device)
        )

        total_loss = (
            RASTER_LOSS_WEIGHT * raster_loss
            + VECTOR_LOSS_WEIGHT_COMMAND * command_loss
            + VECTOR_LOSS_WEIGHT_COORD * coord_loss
        )
        return {
            "total_loss": total_loss,
            "raster_loss": raster_loss,
            "command_loss": command_loss,
            "coord_loss": coord_loss,
        }

    def validation_step(self, batch, raster_loss_fn, command_loss_fn):
        device = next(self.parameters()).device
        (style_image_input, glyph_id_input, target_sequence_input), y = batch
        style_image_input, glyph_id_input, target_sequence_input = (
            style_image_input.to(device),
            glyph_id_input.to(device),
            target_sequence_input.to(device),
        )
        y = {k: v.to(device) for k, v in y.items()}

        outputs = self((style_image_input, glyph_id_input, target_sequence_input))
        raster_loss = raster_loss_fn(outputs["raster"], y["raster"])
        true_command, true_coord = y["command"], y["coord"]

        command_loss = command_loss_fn(outputs["command"], true_command)
        coord_loss = calculate_masked_coordinate_loss(
            true_command,
            true_coord,
            outputs["coord"],
            self.arg_counts.to(device),
        )

        total_loss = (
            RASTER_LOSS_WEIGHT * raster_loss
            + VECTOR_LOSS_WEIGHT_COMMAND * command_loss
            + VECTOR_LOSS_WEIGHT_COORD * coord_loss
        )
        return {
            "total_loss": total_loss,
            "raster_loss": raster_loss,
            "command_loss": command_loss,
            "coord_loss": coord_loss,
        }

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


if __name__ == "__main__":
    # Example usage:
    num_glyphs = 42
    d_model = 512
    model = build_model(num_glyphs, d_model)

    # Create some dummy input tensors
    style_image = torch.randn(1, 1, 40, 168)
    glyph_id = torch.randn(1, num_glyphs)
    target_sequence = torch.randn(1, 150, 12)

    # Forward pass
    output = model((style_image, glyph_id, target_sequence))

    # Print output shapes
    print("Raster output shape:", output["raster"].shape)
    print("Command output shape:", output["command"].shape)
    print("Coord output shape:", output["coord"].shape)
