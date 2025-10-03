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
    SIGNED_AREA_WEIGHT,
    VECTOR_LOSS_WEIGHT_COMMAND,
    VECTOR_LOSS_WEIGHT_COORD,
    VECTOR_RASTERIZATION_LOSS_WEIGHT,
    HUBER_DELTA,
    LOSS_IMAGE_SIZE,
    RASTER_LOSS_CUTOFF,
    RASTER_BLACK_PIXEL_WEIGHT,
)
from glyphogen_torch.lstm import LSTMDecoder
from glyphogen_torch.rasterizer import rasterize_batch

SKIP_RASTERIZATION = VECTOR_RASTERIZATION_LOSS_WEIGHT == 0.0


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

        current_command_indices = command_indices[:, i]
        is_lh = current_command_indices == command_keys.index("LH")
        is_lv = current_command_indices == command_keys.index("LV")

        # For relative line commands (LH, LV), one of the deltas is zero.
        # For normal commands, both deltas are used.
        # The value for LH/LV is always taken from the first coordinate.
        dx = coord_tensor_relative[:, i, 0]
        dy = coord_tensor_relative[:, i, 1]

        step_dx = torch.where(is_lv, torch.zeros_like(dx), dx)
        step_dy = torch.where(
            is_lh,
            torch.zeros_like(dy),
            torch.where(is_lv, dx, dy),  # For LV, dy is taken from the first coord
        )
        relative_coords_step = torch.stack([step_dx, step_dy], dim=-1)

        current_pos = torch.where(
            is_first.unsqueeze(-1).expand_as(current_pos),
            relative_coords_step,
            current_pos + relative_coords_step,
        )

        absolute_coords_step = coord_tensor_relative[:, i, :].clone()
        absolute_coords_step[:, 0:2] = current_pos
        coord_tensor_absolute[:, i, :] = absolute_coords_step

    return coord_tensor_absolute


@torch.compile(backend="aot_eager")
def calculate_masked_coordinate_loss(
    y_true_command,
    y_true_coord,
    y_pred_coord,
    arg_counts,
    sequence_mask,
    delta=HUBER_DELTA,
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

    # Combine the argument mask with the sequence length mask
    combined_mask = coord_mask * sequence_mask.unsqueeze(-1)
    masked_loss = huber_loss * combined_mask

    return torch.sum(masked_loss) / torch.sum(combined_mask)


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


@torch.compile(dynamic=True)
def signed_area(points):
    """
    Calculates the signed area of a polygon using the Shoelace formula.
    The input should be a tensor of shape (N, 2) where N is the number of vertices.
    """
    if points.shape[0] < 3:
        return torch.tensor(0.0, device=points.device, dtype=points.dtype)
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * (
        torch.sum(x[:-1] * y[1:])
        + x[-1] * y[0]
        - torch.sum(y[:-1] * x[1:])
        - y[-1] * x[0]
    )


@torch.compiler.disable
def point_placement_loss(
    y_true_command,
    y_true_coord,
    y_pred_command,
    y_pred_coord,
    true_eos_idx,
    sequence_mask,
    step,
):
    """
    Calculates losses based on drawing practices, ensuring calculations are
    masked to the length of the true command sequence.
    """
    command_keys = list(NODE_GLYPH_COMMANDS.keys())
    soc_index = command_keys.index("SOC")
    n_index = command_keys.index("N")
    nh_index = command_keys.index("NH")
    nv_index = command_keys.index("NV")
    eos_index = command_keys.index("EOS")
    batch_size, seq_len, _ = y_true_command.shape
    y_true_command_indices = torch.argmax(y_true_command, axis=-1)
    indices = torch.arange(seq_len, device=y_true_command.device).expand(batch_size, -1)

    # --- Predicted Probabilities ---
    pred_probs = F.softmax(y_pred_command, dim=-1)

    # --- Contour Count Loss ---
    true_soc_counts = torch.sum(
        (y_true_command_indices == soc_index) * sequence_mask, dim=1
    )
    pred_soc_probs = pred_probs[:, :, soc_index]
    soft_pred_soc_counts = torch.sum(pred_soc_probs * sequence_mask, dim=1)
    contour_loss = F.l1_loss(soft_pred_soc_counts, true_soc_counts)

    # --- Node Count Loss (Sequence Length Loss) ---
    # A more direct 'soft length' formulation.
    eos_probs = pred_probs[:, :, eos_index]
    continue_probs = 1.0 - eos_probs
    soft_len = torch.sum(continue_probs * sequence_mask, dim=1)
    eos_loss = F.l1_loss(soft_len, true_eos_idx.float())

    # --- Handle Orientation Loss ---
    pred_n_prob = torch.mean(pred_probs[:, :, n_index])
    pred_nh_prob = torch.mean(pred_probs[:, :, nh_index])
    pred_nv_prob = torch.mean(pred_probs[:, :, nv_index])
    handle_loss = pred_n_prob - pred_nh_prob - pred_nv_prob

    # --- Signed Area Loss ---
    signed_area_loss_total = torch.tensor(0.0, device=y_true_command.device)
    num_contours_in_batch = 0

    for i in range(batch_size):
        item_mask = sequence_mask[i]
        # Find all explicit SOCs within the valid sequence for this item.
        explicit_soc_starts = torch.where(
            (y_true_command_indices[i] == soc_index) * item_mask
        )[0]

        # The first contour always starts at index 0.
        # All contour starts are [0] concatenated with the explicit SOCs.
        all_contour_starts = torch.cat(
            [torch.tensor([0], device=y_true_command.device), explicit_soc_starts]
        )

        # The end of a contour is the start of the next, or the final EOS.
        eos_pos = true_eos_idx[i]
        all_contour_ends = torch.cat((all_contour_starts[1:], eos_pos.unsqueeze(0)))
        contour_count = len(all_contour_starts)

        for countour in range(contour_count):
            start = all_contour_starts[countour]
            end = all_contour_ends[countour]

            # Skip empty contours that might be created by this logic.
            if end <= start:
                continue

            true_points = y_true_coord[i, start:end, :2]
            pred_points = y_pred_coord[i, start:end, :2]

            true_area = signed_area(true_points)
            pred_area = signed_area(pred_points)

            signed_area_loss_total += torch.abs(true_area - pred_area)
            num_contours_in_batch += 1

    if num_contours_in_batch > 0:
        signed_area_loss = signed_area_loss_total / num_contours_in_batch
    else:
        signed_area_loss = torch.tensor(0.0, device=y_true_command.device)

    return contour_loss, eos_loss, handle_loss, signed_area_loss


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


def weighted_raster_loss(y_true, y_pred, black_pixel_weight=10.0):
    """
    Calculates a weighted Mean Squared Error (MSE) loss for raster images.

    This loss function addresses the "imbalanced class" problem inherent in
    glyph raster images, where the vast majority of pixels are white (background).
    A standard MSE would allow the model to achieve a deceptively low loss by
    simply outputting a blank image, as the error from the few black (foreground)
    pixels would be averaged out over the entire image.

    To counteract this, we apply a higher weight to the errors on pixels that
    should be black. This forces the model to pay significantly more attention to
    correctly forming the glyph's shape, rather than just getting the background
    right.

    Args:
        y_true: The ground truth raster image (tensor).
        y_pred: The predicted raster image (tensor).
        black_pixel_weight: The factor by which to multiply the loss for black pixels.

    Returns:
        A single scalar tensor representing the weighted loss.
    """
    # Element-wise squared error
    error = F.mse_loss(y_pred, y_true, reduction="none")

    # Create a weight map. Black pixels (value closer to 1) get higher weight.
    # We use y_true to determine where the black pixels *should* be.
    weight_map = torch.ones_like(y_true)
    weight_map[y_true >= 0.5] = black_pixel_weight

    # Apply weights and calculate the mean
    weighted_error = error * weight_map
    return torch.mean(weighted_error)


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


@torch.compile(fullgraph=True)
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

    def step(self, batch, step, val=False):
        with torch.profiler.record_function("h2d copy"):
            device = next(self.parameters()).device
            (inputs, y) = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            y = tuple(v.to(device) for v in y)

        with torch.profiler.record_function("forward step"):
            outputs = self(inputs)
        with torch.profiler.record_function("losses"):
            losses = self.losses(y, inputs["raster_image"], outputs, step, val=val)
        return losses

    @torch.compile()
    def losses(self, y, raster_image_input, outputs, step, val=False):
        device = next(self.parameters()).device
        (true_command, true_coord_relative) = y
        batch_size, seq_len, _ = true_command.shape

        # --- Common Masking ---
        # Establish a single, authoritative mask based on the ground truth sequence length.
        true_eos_idx = find_eos(true_command)
        indices = torch.arange(seq_len, device=device).expand(batch_size, -1)
        sequence_mask = (indices < true_eos_idx.unsqueeze(1)).float()

        # Unroll true coordinates to get absolute for loss calculation
        true_coord_absolute = unroll_relative_coords(
            true_command, true_coord_relative * MAX_COORDINATE
        )

        metrics = {}

        command_loss = sequence_command_loss(true_command, outputs["command"])
        metrics["command_raw"] = command_loss.detach()

        coord_loss = calculate_masked_coordinate_loss(
            true_command,
            true_coord_absolute,
            outputs["coord_absolute"],
            self.arg_counts.to(device),
            sequence_mask,
        )
        metrics["coord_raw"] = coord_loss.detach()

        (
            contour_count_loss,
            node_count_loss,
            handle_smoothness_loss,
            signed_area_loss,
        ) = point_placement_loss(
            true_command,
            true_coord_absolute,
            outputs["command"],
            outputs["coord_absolute"],
            true_eos_idx,
            sequence_mask,
            step,
        )
        metrics["contour_count_raw"] = contour_count_loss.detach()
        metrics["node_count_raw"] = node_count_loss.detach()
        metrics["handle_smoothness_raw"] = handle_smoothness_loss.detach()
        metrics["signed_area_raw"] = signed_area_loss.detach()

        if SKIP_RASTERIZATION and not val:
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
            raster_loss = weighted_raster_loss(
                raster_image_input,
                vector_rendered_images,
                black_pixel_weight=RASTER_BLACK_PIXEL_WEIGHT,
            )
            if val:
                metrics["raster_metric"] = (
                    1.0
                    - torch.nn.MSELoss()(
                        raster_image_input, vector_rendered_images
                    ).detach()
                )
            metrics["raster_raw"] = raster_loss.detach()

        total_loss = (
            VECTOR_LOSS_WEIGHT_COMMAND * command_loss
            + VECTOR_LOSS_WEIGHT_COORD * coord_loss
            + CONTOUR_COUNT_WEIGHT * contour_count_loss
            + NODE_COUNT_WEIGHT * node_count_loss
            + HANDLE_SMOOTHNESS_WEIGHT * handle_smoothness_loss
            + SIGNED_AREA_WEIGHT * signed_area_loss
            + VECTOR_RASTERIZATION_LOSS_WEIGHT * raster_loss
        )
        losses = {
            "total_loss": total_loss,
            "command_loss": VECTOR_LOSS_WEIGHT_COMMAND * command_loss,
            "coord_loss": VECTOR_LOSS_WEIGHT_COORD * coord_loss,
        }
        if CONTOUR_COUNT_WEIGHT > 0.0:
            losses["contour_count_loss"] = CONTOUR_COUNT_WEIGHT * contour_count_loss
        if NODE_COUNT_WEIGHT > 0.0:
            losses["node_count_loss"] = NODE_COUNT_WEIGHT * node_count_loss
        if HANDLE_SMOOTHNESS_WEIGHT > 0.0:
            losses["handle_smoothness_loss"] = (
                HANDLE_SMOOTHNESS_WEIGHT * handle_smoothness_loss
            )
        if SIGNED_AREA_WEIGHT > 0.0:
            losses["signed_area_loss"] = SIGNED_AREA_WEIGHT * signed_area_loss
        if not SKIP_RASTERIZATION:
            losses["raster_loss"] = VECTOR_RASTERIZATION_LOSS_WEIGHT * raster_loss
        return {**losses, **metrics}

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
