import torch
import torch.nn.functional as F
from torchmetrics.classification import BinaryJaccardIndex

from glyphogen.command_defs import (
    NODE_GLYPH_COMMANDS,
    MAX_COORDINATE,
)
from glyphogen.hyperparameters import (
    HUBER_DELTA,
    CONTOUR_COUNT_WEIGHT,
    HANDLE_SMOOTHNESS_WEIGHT,
    NODE_COUNT_WEIGHT,
    SIGNED_AREA_WEIGHT,
    VECTOR_LOSS_WEIGHT_COMMAND,
    VECTOR_LOSS_WEIGHT_COORD,
    VECTOR_RASTERIZATION_LOSS_WEIGHT,
    LOSS_IMAGE_SIZE,
)
from glyphogen.rasterizer import rasterize_batch

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
    # Initialize with relative coordinates; we'll update the first two columns.
    coord_tensor_absolute = coord_tensor_relative.clone()

    for i in range(seq_len):
        is_first = is_first_node_in_contour[:, i]

        relative_coords_step = coord_tensor_relative[:, i, 0:2]

        current_pos = torch.where(
            is_first.unsqueeze(-1).expand_as(current_pos),
            relative_coords_step,
            current_pos + relative_coords_step,
        )

        coord_tensor_absolute[:, i, 0:2] = current_pos

    return coord_tensor_absolute


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
    pred_coord_width = y_pred_coord.shape[-1]
    combined_mask = (
        coord_mask[:, :, :pred_coord_width] * sequence_mask.unsqueeze(-1)
    )
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
def abs_signed_area_loss(true_points, pred_points):
    """
    Calculates the signed area of a polygon using the Shoelace formula.
    The input should be a tensor of shape (N, 2) where N is the number of vertices.

    We do this inefficiently in two steps because the pred points will have
    gradients and the true points will not, which confuses the compiler
    """
    # Compute true signed area
    if true_points.shape[0] < 3:
        true_signed_area = torch.tensor(0.0, device=true_points.device, dtype=true_points.dtype)
    else:
        x = true_points[:, 0]
        y = true_points[:, 1]
        true_signed_area = 0.5 * (
            torch.sum(x[:-1] * y[1:])
            + x[-1] * y[0]
            - torch.sum(y[:-1] * x[1:])
            - y[-1] * x[0]
        )
    
    # Compute predicted signed area
    if pred_points.shape[0] < 3:
        pred_signed_area = torch.tensor(0.0, device=pred_points.device, dtype=pred_points.dtype)
    else:
        x = pred_points[:, 0]
        y = pred_points[:, 1]
        pred_signed_area = 0.5 * (
            torch.sum(x[:-1] * y[1:])
            + x[-1] * y[0]
            - torch.sum(y[:-1] * x[1:])
            - y[-1] * x[0]
        )
    
    return torch.abs(true_signed_area - pred_signed_area)


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
    eos_index = command_keys.index("EOS")
    batch_size, seq_len, _ = y_true_command.shape
    y_true_command_indices = torch.argmax(y_true_command, axis=-1)

    # --- Predicted Probabilities ---
    pred_probs = F.softmax(y_pred_command, dim=-1)

    # --- Node Count Loss (Sequence Length Loss) ---
    # A more direct 'soft length' formulation.
    eos_probs = pred_probs[:, :, eos_index]
    continue_probs = 1.0 - eos_probs
    soft_len = torch.sum(continue_probs * sequence_mask, dim=1)
    eos_loss = F.l1_loss(soft_len, true_eos_idx.float())

    # --- Handle Orientation Loss ---
    handle_loss = torch.tensor(0.0, device=y_true_command.device)

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

            area_diff = abs_signed_area_loss(true_points, pred_points)
            signed_area_loss_total += area_diff
            num_contours_in_batch += 1

    if num_contours_in_batch > 0:
        signed_area_loss = signed_area_loss_total / num_contours_in_batch
    else:
        signed_area_loss = torch.tensor(0.0, device=y_true_command.device)

    return eos_loss, handle_loss, signed_area_loss


def sequence_command_loss(y_true_command, y_pred_command):
    """
    Calculates the command loss and accuracy.
    """
    # Standard loss calculation setup
    true_eos_idx = find_eos(y_true_command)
    pred_eos_idx = find_eos(y_pred_command)
    max_len = torch.max(true_eos_idx, pred_eos_idx)

    batch_size, seq_len, num_classes = y_true_command.shape
    device = y_true_command.device

    indices = torch.arange(seq_len, device=device).expand(batch_size, -1)
    mask = (indices < max_len.unsqueeze(1)).float()

    y_true_indices = torch.argmax(y_true_command, dim=-1)
    y_pred_indices = torch.argmax(y_pred_command, dim=-1)

    # Calculate per-token cross-entropy loss
    TEMP = 1
    nll_loss = F.cross_entropy(
        y_pred_command.permute(0, 2, 1) / TEMP,
        y_true_indices,
        reduction="none",
        label_smoothing=0.15,
    )

    # Apply sequence mask
    masked_loss = nll_loss * mask

    # Normalize by the number of unmasked elements
    loss = torch.sum(masked_loss) / (torch.sum(mask) + 1e-8)

    # --- Accuracy Calculation ---
    correct_predictions = (y_true_indices == y_pred_indices).float()
    masked_correct = correct_predictions * mask
    accuracy = torch.sum(masked_correct) / (torch.sum(mask) + 1e-8)

    return loss, accuracy



def losses(y, inputs, outputs, step, device, arg_counts, val=False):
    jaccard = BinaryJaccardIndex().to(device)
    (true_command, true_coord_relative) = (y["command"], y["coord"])
    raster_image_input = inputs["raster_image"]
    true_contour_count = inputs["contour_count"]
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

    command_loss, command_accuracy = sequence_command_loss(
        true_command, outputs["command"]
    )
    metrics["command_raw"] = command_loss.detach()
    metrics["command_accuracy"] = command_accuracy.detach()

    coord_loss = calculate_masked_coordinate_loss(
        true_command,
        true_coord_absolute,
        outputs["coord_absolute"],
        arg_counts.to(device),
        sequence_mask,
    )
    metrics["coord_raw"] = coord_loss.detach()

    (
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
    metrics["node_count_raw"] = node_count_loss.detach()
    metrics["handle_smoothness_raw"] = handle_smoothness_loss.detach()
    metrics["signed_area_raw"] = signed_area_loss.detach()

    # Contour count loss from regression head
    pred_contour_count = outputs["pred_contour_count"].squeeze(-1)
    contour_count_loss = F.l1_loss(pred_contour_count, true_contour_count)
    metrics["contour_count_raw"] = contour_count_loss.detach()

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
        raster_loss = jaccard(
            vector_rendered_images < 0.2,  # Match black pixels
            raster_image_input < 0.2,
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
