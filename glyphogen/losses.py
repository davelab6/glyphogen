from glyphogen.coordinate import to_image_space
import torch
import torch.nn.functional as F

from glyphogen.command_defs import NODE_GLYPH_COMMANDS
from glyphogen.hyperparameters import (
    HUBER_DELTA,
    VECTOR_LOSS_WEIGHT_COMMAND,
    VECTOR_LOSS_WEIGHT_COORD,
)


def losses(y, outputs, device, validation=False):
    """
    Calculates losses for the hierarchical vectorization model.
    This function iterates through contours and calculates loss for each one,
    comparing absolute coordinates directly.
    """
    gt_contours = y["gt_contours"]
    pred_commands_list = outputs["pred_commands"]
    pred_coords_abs_list = outputs["pred_coords_img_space"]

    num_contours_to_compare = min(len(gt_contours), len(pred_commands_list))
    total_command_loss = torch.tensor(0.0, device=device)
    total_coord_loss = torch.tensor(0.0, device=device)

    if num_contours_to_compare == 0:
        return {
            "total_loss": torch.tensor(0.0, device=device, requires_grad=True),
            "command_loss": torch.tensor(0.0, device=device),
            "coord_loss": torch.tensor(0.0, device=device),
        }

    for i in range(num_contours_to_compare):
        pred_command = pred_commands_list[i]
        pred_coords = pred_coords_abs_list[i]

        gt_contour = gt_contours[i]
        gt_sequence = gt_contour["sequence"]
        command_width = len(NODE_GLYPH_COMMANDS)
        gt_command = gt_sequence[:, :command_width]
        gt_coords = gt_sequence[:, command_width:]

        used_teacher_forcing = outputs.get("used_teacher_forcing", True)

        # CRITICAL ALIGNMENT FIX:
        # GT sequence structure: [SOS, N1, N2, ..., Nn, EOS]
        #
        # Teacher forcing:
        #   - Input to decoder: [SOS, N1, N2, ..., Nn] (everything except EOS)
        #   - Decoder outputs predictions: [pred_N1, pred_N2, ..., pred_Nn, pred_EOS]
        #   - We want to compare: predictions vs [N1, N2, ..., Nn, EOS]
        #   - So: pred vs gt[1:]
        #
        # Autoregressive:
        #   - Start with SOS, generate predictions
        #   - Decoder outputs: [pred_N1, pred_N2, ..., pred_Nn, pred_EOS]
        #   - We want to compare: predictions vs [N1, N2, ..., Nn, EOS]
        #   - So: pred vs gt[1:]
        #
        # Both cases are the same! Compare predictions with gt[1:]

        gt_command_for_loss = gt_command[1:]  # Skip SOS
        gt_coords_for_loss = gt_coords[1:]  # Skip SOS coordinates

        # Predictions should align with gt[1:] in both cases
        pred_command_for_loss = pred_command
        pred_coords_for_loss = pred_coords

        # Pad sequences if lengths differ (can happen in validation)
        if pred_command_for_loss.shape[0] != gt_command_for_loss.shape[0]:
            max_len = max(pred_command_for_loss.shape[0], gt_command_for_loss.shape[0])
            # (Padding logic remains the same as before)
            pred_len, gt_len = (
                pred_command_for_loss.shape[0],
                gt_command_for_loss.shape[0],
            )
            if pred_len < max_len:
                pad_len = max_len - pred_len
                pred_command_pad = torch.zeros(
                    pad_len,
                    pred_command_for_loss.shape[1],
                    device=device,
                    dtype=pred_command_for_loss.dtype,
                )
                pred_coords_pad = torch.zeros(
                    pad_len,
                    pred_coords_for_loss.shape[1],
                    device=device,
                    dtype=pred_coords_for_loss.dtype,
                )
                pred_command_for_loss = torch.cat(
                    [pred_command_for_loss, pred_command_pad], dim=0
                )
                pred_coords_for_loss = torch.cat(
                    [pred_coords_for_loss, pred_coords_pad], dim=0
                )
            if gt_len < max_len:
                pad_len = max_len - gt_len
                eos_index = list(NODE_GLYPH_COMMANDS.keys()).index("EOS")
                gt_command_pad = torch.zeros(
                    pad_len,
                    gt_command_for_loss.shape[1],
                    device=device,
                    dtype=gt_command_for_loss.dtype,
                )
                gt_command_pad[:, eos_index] = 1.0
                gt_coords_pad = torch.zeros(
                    pad_len,
                    gt_coords_for_loss.shape[1],
                    device=device,
                    dtype=gt_coords_for_loss.dtype,
                )
                gt_command_for_loss = torch.cat(
                    [gt_command_for_loss, gt_command_pad], dim=0
                )
                gt_coords_for_loss = torch.cat(
                    [gt_coords_for_loss, gt_coords_pad], dim=0
                )

        if pred_command_for_loss.shape[0] == 0:
            continue

        # For now we only care about the node coordinates, not the handles
        # Slice ground truth.
        gt_coords_for_loss = gt_coords_for_loss[:, :2]
        gt_coords_image_space = to_image_space(gt_coords_for_loss)

        # 1. Command Loss (Cross-Entropy)
        command_loss = F.cross_entropy(
            pred_command_for_loss.unsqueeze(0).permute(0, 2, 1),
            gt_command_for_loss.unsqueeze(0).argmax(dim=-1),
            label_smoothing=0.1,
        )

        # 2. Coordinate Loss (Huber) on absolute coordinates
        coord_loss = F.huber_loss(
            pred_coords_for_loss,
            gt_coords_image_space,
            delta=HUBER_DELTA,
        )

        total_command_loss += command_loss
        total_coord_loss += coord_loss

    avg_command_loss = (
        total_command_loss / num_contours_to_compare
        if num_contours_to_compare > 0
        else torch.tensor(0.0)
    )
    avg_coord_loss = (
        total_coord_loss / num_contours_to_compare
        if num_contours_to_compare > 0
        else torch.tensor(0.0)
    )

    total_loss = (
        VECTOR_LOSS_WEIGHT_COMMAND * avg_command_loss
        + VECTOR_LOSS_WEIGHT_COORD * avg_coord_loss
    )

    return {
        "total_loss": total_loss,
        "command_loss": avg_command_loss.detach(),
        "coord_loss": avg_coord_loss.detach(),
    }
