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

    return eos_loss, handle_loss, signed_area_loss


def sequence_command_loss(y_true_command, y_pred_command):
    """
    Calculates a synonym-aware command loss.

    This loss function understands that some commands are functionally similar
    (e.g., a generic Node 'N' vs. a specialized horizontal Node 'NH'). It applies
    a smaller penalty when the model predicts a valid synonym, and the full
    penalty for more significant errors.
    """
    # Define synonym sets based on the confusion matrix analysis
    command_keys = list(NODE_GLYPH_COMMANDS.keys())
    nh_set = {command_keys.index(cmd) for cmd in ["N", "NH"]}
    nv_set = {command_keys.index(cmd) for cmd in ["N", "NV"]}
    lh_set = {command_keys.index(cmd) for cmd in ["L", "LH"]}
    lv_set = {command_keys.index(cmd) for cmd in ["L", "LV"]}
    nco_set = {command_keys.index(cmd) for cmd in ["NCO", "L"]}
    nci_set = {command_keys.index(cmd) for cmd in ["NCI", "N"]}
    synonym_sets = [
        # Using unoptimized forms instead of optimized forms is kind of OK
        (nh_set, 0.5),
        (nv_set, 0.5),
        (lh_set, 0.5),
        (lv_set, 0.5),
        # Confusions between line and node are more serious but still somewhat understandable
        (nco_set, 0.75),
        (nci_set, 0.75),
    ]

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

    # --- Synonym-aware scaling ---
    # Start with full penalty for all tokens
    scales = torch.ones_like(nll_loss)

    # Check for errors where both true and pred are in the same synonym set
    is_error = y_true_indices != y_pred_indices

    for syn_set, penalty in synonym_sets:
        is_true_in_set = torch.zeros_like(is_error)
        is_pred_in_set = torch.zeros_like(is_error)
        for idx in syn_set:
            is_true_in_set |= y_true_indices == idx
            is_pred_in_set |= y_pred_indices == idx

        is_synonym_error = is_error & is_true_in_set & is_pred_in_set
        scales[is_synonym_error] = penalty

    # Apply scaling and sequence mask
    scaled_loss = nll_loss * scales
    masked_loss = scaled_loss * mask

    # Normalize by the number of unmasked elements
    return torch.sum(masked_loss) / (torch.sum(mask) + 1e-8)


jaccard = BinaryJaccardIndex()


def losses(y, inputs, outputs, step, device, arg_counts, val=False):
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

    command_loss = sequence_command_loss(true_command, outputs["command"])
    metrics["command_raw"] = command_loss.detach()

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
