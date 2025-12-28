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
    comparing image-space coordinates directly.
    """
    gt_contours = y["gt_contours"]
    pred_commands_list = outputs["pred_commands"]
    pred_coords_img_space_list = outputs["pred_coords_img_space"]

    num_contours_to_compare = min(len(gt_contours), len(pred_commands_list))
    total_command_loss = torch.tensor(0.0, device=device)
    total_coord_loss = torch.tensor(0.0, device=device)
    total_correct_cmds = 0
    total_cmds = 0

    if num_contours_to_compare == 0:
        return {
            "total_loss": torch.tensor(0.0, device=device, requires_grad=True),
            "command_loss": torch.tensor(0.0, device=device),
            "coord_loss": torch.tensor(0.0, device=device),
            "command_accuracy": 0.0,
        }

    for i in range(num_contours_to_compare):
        pred_command = pred_commands_list[i]
        pred_coords_img_space = pred_coords_img_space_list[i]

        gt_sequence = gt_contours[i]["sequence"]

        (
            gt_command_for_loss,
            gt_coords_for_loss,
            pred_command_for_loss,
            pred_coords_for_loss,
        ) = align_sequences(
            device,
            gt_sequence,
            pred_command,
            pred_coords_img_space,
        )

        if pred_command_for_loss.shape[0] == 0:
            continue

        # 1. Command Loss (Cross-Entropy)
        command_loss = F.cross_entropy(
            pred_command_for_loss.unsqueeze(0).permute(0, 2, 1),
            gt_command_for_loss.unsqueeze(0).argmax(dim=-1),
            label_smoothing=0.1,
        )

        # 2. Coordinate Loss (Huber) on image-space coordinates, masked by command
        coord_loss = masked_coordinate_loss(
            device, gt_command_for_loss, gt_coords_for_loss, pred_coords_for_loss
        )

        total_command_loss += command_loss
        total_coord_loss += coord_loss

        # Accumulate accuracy stats
        pred_indices = torch.argmax(pred_command_for_loss, dim=-1)
        gt_indices = torch.argmax(gt_command_for_loss, dim=-1)
        total_correct_cmds += (pred_indices == gt_indices).sum().item()
        total_cmds += len(pred_indices)

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
    command_accuracy = (
        torch.tensor(total_correct_cmds / total_cmds)
        if total_cmds > 0
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
        "command_accuracy_metric": command_accuracy,
    }


def masked_coordinate_loss(
    device, gt_command_for_loss, gt_coords_for_loss, pred_coords_for_loss
):
    command_indices = torch.argmax(gt_command_for_loss, dim=-1)

    # Create a lookup tensor for coordinate counts
    arg_counts_list = [NODE_GLYPH_COMMANDS[cmd] for cmd in NODE_GLYPH_COMMANDS]
    arg_counts = torch.tensor(arg_counts_list, device=device)

    # Get the number of relevant coordinates for each timestep
    num_relevant_coords = arg_counts[command_indices]

    # Create the mask
    coord_mask = torch.arange(
        gt_coords_for_loss.shape[1], device=device
    ) < num_relevant_coords.unsqueeze(1)

    elementwise_coord_loss = F.huber_loss(
        pred_coords_for_loss,
        gt_coords_for_loss,
        delta=HUBER_DELTA,
        reduction="none",
    )

    masked_coord_loss = elementwise_coord_loss * coord_mask

    # Average the loss over the number of relevant coordinates
    num_coords_in_loss = coord_mask.sum()
    if num_coords_in_loss > 0:
        coord_loss = masked_coord_loss.sum() / num_coords_in_loss
    else:
        coord_loss = torch.tensor(0.0, device=device)
    return coord_loss


def align_sequences(
    device,
    gt_sequence,
    pred_command,
    pred_coords_img_space,
):
    command_width = len(NODE_GLYPH_COMMANDS)
    gt_command = gt_sequence[:, :command_width]
    gt_coords_img_space = gt_sequence[:, command_width:]
    # Since both training and validation are now teacher-forced, the logic is the same.
    # The model was fed gt_sequence[:-1], so its output corresponds to gt_sequence[1:].
    gt_command_for_loss = gt_command[1:]
    gt_coords_for_loss = gt_coords_img_space[1:]

    pred_command_for_loss = pred_command
    pred_coords_for_loss = pred_coords_img_space

    # Pad sequences if lengths differ (should not happen in teacher forcing)
    # But it will happen when we use the model autoregressively for validation
    if pred_command_for_loss.shape[0] != gt_command_for_loss.shape[0]:
        max_len = max(pred_command_for_loss.shape[0], gt_command_for_loss.shape[0])
        pred_len, gt_len = pred_command_for_loss.shape[0], gt_command_for_loss.shape[0]
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
            gt_coords_for_loss = torch.cat([gt_coords_for_loss, gt_coords_pad], dim=0)
    return (
        gt_command_for_loss,
        gt_coords_for_loss,
        pred_command_for_loss,
        pred_coords_for_loss,
    )
