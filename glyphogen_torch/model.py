#!/usr/bin/env python
import torch
import torch.nn.functional as F
from torch import nn

from glyphogen_torch.embedding import StyleEmbedding
from glyphogen_torch.command_defs import (
    COORDINATE_WIDTH,
    NODE_GLYPH_COMMANDS,
    MAX_COORDINATE,
)
from glyphogen_torch.hyperparameters import (
    CONTOUR_COUNT_WEIGHT,
    EOS_SOFTMAX_TEMPERATURE,
    HANDLE_SMOOTHNESS_WEIGHT,
    NODE_COUNT_WEIGHT,
    RASTER_LOSS_WEIGHT,
    VECTOR_LOSS_WEIGHT_COMMAND,
    VECTOR_LOSS_WEIGHT_COORD,
    VECTOR_RASTERIZATION_LOSS_WEIGHT,
    HUBER_DELTA,
    LOSS_IMAGE_SIZE,
    RASTER_LOSS_CUTOFF,
)
from glyphogen_torch.lstm import LSTMDecoder
from glyphogen_torch.rasterizer import rasterize_batch

SKIP_RASTERIZATION = VECTOR_RASTERIZATION_LOSS_WEIGHT == 0.0


@torch.compile
def unroll_relative_coords(command_tensor, coord_tensor_relative):
    """Converts a tensor of relative coordinates to absolute coordinates."""
    batch_size, seq_len, _ = command_tensor.shape
    device = command_tensor.device

    command_keys = list(NODE_GLYPH_COMMANDS.keys())
    soc_index = command_keys.index("SOC")
    command_indices = torch.argmax(command_tensor, dim=-1)
    is_soc = command_indices == soc_index

    is_first_node_in_contour = torch.zeros_like(is_soc)
    is_first_node_in_contour[:, 1:] = is_soc[:, :-1]

    current_pos = torch.zeros(
        batch_size, 2, device=device, dtype=coord_tensor_relative.dtype
    )
    coord_tensor_absolute = torch.zeros_like(coord_tensor_relative)

    for i in range(seq_len):
        is_first = is_first_node_in_contour[:, i]
        relative_coords_step = coord_tensor_relative[:, i, 0:2]

        current_pos = torch.where(
            is_first.unsqueeze(-1).expand_as(current_pos),
            relative_coords_step,
            current_pos + relative_coords_step,
        )

        absolute_coords_step = coord_tensor_relative[:, i, :].clone()
        absolute_coords_step[:, 0:2] = current_pos
        coord_tensor_absolute[:, i, :] = absolute_coords_step

    return coord_tensor_absolute


def calculate_masked_coordinate_loss(
    y_true_command, y_true_coord, y_pred_coord, arg_counts, delta=HUBER_DELTA
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


def find_eos(command_batch):
    command_keys = list(NODE_GLYPH_COMMANDS.keys())
    eos_index = command_keys.index("EOS")
    command_indices = torch.argmax(command_batch, axis=-1)
    eos_mask = command_indices == eos_index
    eos_idx = torch.where(
        torch.any(eos_mask, dim=1),
        torch.argmax(eos_mask.float(), dim=1),
        torch.full(
            (command_batch.shape[0],),
            command_batch.shape[1],
            device=command_batch.device,
            dtype=torch.long,
        ),
    )
    return eos_idx


@torch.compile
def point_placement_loss(y_true_command, y_pred_command):
    """
    Calculates a loss based on drawing practices.
    """
    command_keys = list(NODE_GLYPH_COMMANDS.keys())
    soc_index = command_keys.index("SOC")
    n_index = command_keys.index("N")
    nh_index = command_keys.index("NH")
    nv_index = command_keys.index("NV")
    eos_index = command_keys.index("EOS")

    y_true_command_indices = torch.argmax(y_true_command, axis=-1)
    true_eos_idx = find_eos(y_true_command)

    # Contour count loss
    batch_size = y_true_command.shape[0]

    true_soc_counts = []
    for i in range(batch_size):
        true_seq = y_true_command_indices[i, : true_eos_idx[i]]
        true_soc_counts.append(torch.sum(true_seq == soc_index))
    true_soc_counts = torch.stack(true_soc_counts).float()

    pred_probs = F.softmax(y_pred_command, dim=-1)
    pred_soc_probs = pred_probs[:, :, soc_index]
    eos_probs = pred_probs[:, :, eos_index]
    continue_probs = 1.0 - eos_probs
    cumprod_continue = torch.cumprod(continue_probs, dim=1)
    mask = torch.cat(
        [
            torch.ones(batch_size, 1, device=y_pred_command.device),
            cumprod_continue[:, :-1],
        ],
        dim=1,
    )
    soft_pred_soc_counts = torch.sum(pred_soc_probs * mask, dim=1)
    contour_loss = torch.mean(torch.abs(true_soc_counts - soft_pred_soc_counts))

    # Sequence length loss; try to do it in as many nodes as the designer or fewer
    soft_argmax_probs = F.softmax(eos_probs / EOS_SOFTMAX_TEMPERATURE, dim=1)
    indices = torch.arange(
        y_pred_command.shape[1], device=y_pred_command.device, dtype=torch.float32
    )
    soft_pred_eos_idx = torch.sum(soft_argmax_probs * indices, dim=1)
    eos_loss = torch.mean(
        torch.max(
            soft_pred_eos_idx - true_eos_idx.float(),
            torch.zeros_like(soft_pred_eos_idx),
        )
    )

    # Handle orientation loss
    pred_n_prob = torch.mean(pred_probs[:, :, n_index])
    pred_nh_prob = torch.mean(pred_probs[:, :, nh_index])
    pred_nv_prob = torch.mean(pred_probs[:, :, nv_index])
    handle_loss = pred_n_prob - pred_nh_prob - pred_nv_prob

    return contour_loss, eos_loss, handle_loss


def sequence_command_loss(y_true_command, y_pred_command):
    """
    Calculates the command loss only up to the longest EOS between true and pred.
    """
    true_eos_idx = find_eos(y_true_command)
    pred_eos_idx = find_eos(y_pred_command)

    max_len = torch.max(true_eos_idx, pred_eos_idx)

    batch_size, seq_len, num_classes = y_true_command.shape

    # Create a mask for the sequences
    indices = torch.arange(seq_len, device=y_true_command.device).expand(batch_size, -1)
    mask = indices < max_len.unsqueeze(1)  # shape (N, L), boolean

    # Get true class indices
    y_true_indices = torch.argmax(y_true_command, dim=-1)  # shape (N, L)

    # Apply log_softmax to predictions over the class dimension
    log_probs = F.log_softmax(y_pred_command, dim=-1)  # shape (N, L, C)

    # Gather the log probabilities of the true classes
    true_log_probs = torch.gather(log_probs, 2, y_true_indices.unsqueeze(-1)).squeeze(
        -1
    )  # shape (N, L)

    # Calculate negative log likelihood loss
    nll_loss = -true_log_probs

    # Apply mask
    masked_loss = nll_loss * mask.float()

    # Normalize by the number of unmasked elements
    return torch.sum(masked_loss) / (torch.sum(mask) + 1e-8)


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
        self.use_raster = False

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
        inputs = {k: v.to(device) for k, v in inputs.items()}
        y = tuple(v.to(device) for v in y)

        outputs = self(inputs)
        losses = self.losses(y, inputs["raster_image"], outputs, step)
        return losses

    def losses(self, y, raster_image_input, outputs, step):
        device = next(self.parameters()).device
        (true_command, true_coord_relative) = y

        # Unroll true coordinates to get absolute for loss calculation
        true_coord_absolute = unroll_relative_coords(
            true_command, true_coord_relative * MAX_COORDINATE
        )

        command_loss = sequence_command_loss(true_command, outputs["command"])
        coord_loss = calculate_masked_coordinate_loss(
            true_command,
            true_coord_absolute,
            outputs["coord_absolute"],
            self.arg_counts.to(device),
        )
        (
            contour_count_loss,
            node_count_loss,
            handle_smoothness_loss,
        ) = point_placement_loss(true_command, outputs["command"])

        if SKIP_RASTERIZATION:
            raster_loss = torch.tensor(0.0, device=device)
        else:
            vector_rendered_images = rasterize_batch(
                outputs["command"],
                outputs["coord_absolute"] / MAX_COORDINATE,
                seed=step,
                img_size=LOSS_IMAGE_SIZE,
            ).to(device)
            # Resize raster_image_input to match vector_rendered_images if needed
            if raster_image_input.shape != vector_rendered_images.shape:
                raster_image_input = F.interpolate(
                    raster_image_input,
                    size=vector_rendered_images.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
            raster_loss = self.raster_loss_fn(
                raster_image_input, vector_rendered_images
            )
            # Only apply the loss if we are "close enough" to tweak, otherwise it
            # makes things worse.
            if (
                raster_loss < RASTER_LOSS_CUTOFF and command_loss < 0.25
            ) or self.use_raster:
                # Flip over to using raster loss exclusively from now on
                self.use_raster = True
            else:
                raster_loss = torch.tensor(RASTER_LOSS_CUTOFF, device=device)
            # if self.use_raster:
            #     coord_loss = coord_loss.detach()  # It's a metric now

        total_loss = (
            VECTOR_LOSS_WEIGHT_COMMAND * command_loss
            + VECTOR_LOSS_WEIGHT_COORD * coord_loss
            + CONTOUR_COUNT_WEIGHT * contour_count_loss
            + NODE_COUNT_WEIGHT * node_count_loss
            + HANDLE_SMOOTHNESS_WEIGHT * handle_smoothness_loss
            + VECTOR_RASTERIZATION_LOSS_WEIGHT * raster_loss
        )
        return {
            "total_loss": total_loss,
            "command_loss": VECTOR_LOSS_WEIGHT_COMMAND * command_loss,
            "coord_loss": VECTOR_LOSS_WEIGHT_COORD * coord_loss,
            "contour_count_loss": CONTOUR_COUNT_WEIGHT * contour_count_loss,
            "node_count_loss": NODE_COUNT_WEIGHT * node_count_loss,
            "handle_smoothness_loss": HANDLE_SMOOTHNESS_WEIGHT * handle_smoothness_loss,
            "raster_loss": VECTOR_RASTERIZATION_LOSS_WEIGHT * raster_loss,
        }

    @torch.compile(backend="aot_eager")
    def forward(self, inputs):
        raster_image_input = inputs["raster_image"]
        target_sequence_input = inputs["target_sequence"]
        z = self.encode(raster_image_input)
        z = z.unsqueeze(1)

        command_output, coord_output_relative = self.decoder(
            target_sequence_input, context=z
        )

        # Scale up relative coordinates
        coord_output_relative_scaled = coord_output_relative * MAX_COORDINATE

        coord_output_absolute = unroll_relative_coords(
            command_output, coord_output_relative_scaled
        )
        return {
            "command": command_output,
            "coord_relative": coord_output_relative,
            "coord_absolute": coord_output_absolute,
        }


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
        inputs, y = batch
        inputs = {k: v.to(device) for k, v in inputs.items()}
        y = {k: v.to(device) for k, v in y.items()}

        outputs = self(inputs)
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
        inputs, y = batch
        inputs = {k: v.to(device) for k, v in inputs.items()}
        y = {k: v.to(device) for k, v in y.items()}

        outputs = self(inputs)
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
        style_image_input = inputs["style_image"]
        glyph_id_input = inputs["glyph_id"]
        target_sequence_input = inputs["target_sequence"]

        _, _, z = self.style_embedding(style_image_input)

        glyph_id_embedded = self.glyph_id_embedding(glyph_id_input)
        combined = torch.cat([z, glyph_id_embedded], dim=-1)
        generated_glyph_raster = self.raster_decoder(combined)

        vectorizer_inputs = {
            "raster_image": generated_glyph_raster,
            "target_sequence": target_sequence_input,
        }
        vectorizer_output = self.vectorizer(vectorizer_inputs)
        command_output = vectorizer_output["command"]
        coord_output = vectorizer_output["coord_absolute"]

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
