from typing import List
from glyphogen.nodeglyph import NodeGlyph
from glyphogen.svgglyph import SVGGlyph
import torch
import torch.nn.functional as F
import sys

from glyphogen.command_defs import NodeCommand
from glyphogen.hyperparameters import (
    ALIGNMENT_LOSS_WEIGHT,
    HUBER_DELTA,
    SIGNED_AREA_WEIGHT,
    VECTOR_LOSS_WEIGHT_COMMAND,
    VECTOR_LOSS_WEIGHT_COORD,
)
from glyphogen.typing import (
    CollatedGlyphData,
    GroundTruthContour,
    LossDictionary,
    ModelResults,
)


@torch.compile
def losses(
    collated_batch: CollatedGlyphData,
    outputs: ModelResults,
    device,
    validation=False,
) -> LossDictionary:
    """
    Calculates losses for the hierarchical vectorization model.
    This function now iterates through all contours in a collated batch.
    """
    pred_commands_list = outputs.pred_commands
    pred_coords_norm_list = outputs.pred_coords_norm

    # Data from the collated batch
    gt_target_sequences = collated_batch["target_sequences"]
    contour_boxes = collated_batch["contour_boxes"]
    x_aligned_point_indices = collated_batch["x_aligned_point_indices"]
    y_aligned_point_indices = collated_batch["y_aligned_point_indices"]

    num_contours_to_compare = len(gt_target_sequences)
    total_command_loss = torch.tensor(0.0, device=device)
    total_coord_loss = torch.tensor(0.0, device=device)
    total_signed_area_loss = torch.tensor(0.0, device=device)
    total_alignment_loss = torch.tensor(0.0, device=device)
    total_coord_mae_metric = torch.tensor(0.0, device=device)
    total_correct_cmds = 0
    total_cmds = 0

    if num_contours_to_compare == 0:
        return {
            "total_loss": torch.tensor(0.0, device=device, requires_grad=True),
            "command_loss": torch.tensor(0.0, device=device),
            "coord_loss": torch.tensor(0.0, device=device),
            "signed_area_loss": torch.tensor(0.0, device=device),
            "alignment_loss": torch.tensor(0.0, device=device),
            "command_accuracy_metric": torch.tensor(0.0, device=device),
            "coordinate_mae_metric": torch.tensor(0.0, device=device),
        }

    for i in range(num_contours_to_compare):
        pred_command = pred_commands_list[i]
        pred_coords_norm = pred_coords_norm_list[i]
        box = contour_boxes[i]

        # Convert GT sequence from image space to normalized mask space
        # Move GT tensor to device here, as it's not handled in the main loop anymore
        gt_sequence_img_space = gt_target_sequences[i].to(device)
        gt_sequence_norm = NodeCommand.image_space_to_mask_space(
            gt_sequence_img_space, box
        )

        (
            gt_command_for_loss,
            gt_coords_for_loss,
            pred_command_for_loss,
            pred_coords_for_loss,
        ) = align_sequences(
            device,
            gt_sequence_norm,
            pred_command,
            pred_coords_norm,
        )

        if pred_command_for_loss.shape[0] == 0:
            continue

        # 1. Command Loss (Cross-Entropy)
        command_loss = F.cross_entropy(
            pred_command_for_loss.unsqueeze(0).permute(0, 2, 1),
            gt_command_for_loss.unsqueeze(0).argmax(dim=-1),
            label_smoothing=0.1,
        )

        # 2. Coordinate Loss (Huber) on absolute normalized mask-space coordinates
        coord_loss = masked_coordinate_loss(
            device, gt_command_for_loss, gt_coords_for_loss, pred_coords_for_loss
        )

        # 3. Signed Area Loss
        # A vertex is any command that is not SOS or EOS.
        eos_idx = NodeCommand.encode_command("EOS")
        sos_idx = NodeCommand.encode_command("SOS")
        gt_vertex_mask = (gt_command_for_loss.argmax(dim=-1) != eos_idx) & (
            gt_command_for_loss.argmax(dim=-1) != sos_idx
        )
        pred_vertex_mask = (pred_command_for_loss.argmax(dim=-1) != eos_idx) & (
            pred_command_for_loss.argmax(dim=-1) != sos_idx
        )

        # The on-curve points are always the first two coordinates.
        gt_on_curve_points = gt_coords_for_loss[gt_vertex_mask, 0:2]
        pred_on_curve_points = pred_coords_for_loss[pred_vertex_mask, 0:2]

        signed_area_loss = abs_signed_area_loss(
            gt_on_curve_points, pred_on_curve_points
        )

        # 4. Alignment Loss
        align_loss = alignment_loss(
            pred_coords_for_loss,
            x_aligned_point_indices[i],
            y_aligned_point_indices[i],
            device,
            gt_command_for_loss,
        )

        total_command_loss += command_loss
        total_coord_loss += coord_loss
        total_signed_area_loss += signed_area_loss
        total_alignment_loss += align_loss

        # Accumulate accuracy stats
        pred_indices = torch.argmax(pred_command_for_loss, dim=-1)
        gt_indices = torch.argmax(gt_command_for_loss, dim=-1)
        correct_commands = (pred_indices == gt_indices).sum().item()
        total_correct_cmds += correct_commands
        total_cmds += len(pred_indices)
        # And metric
        coord_mae_metric = masked_coordinate_mae_metric(
            device, gt_command_for_loss, gt_coords_for_loss, pred_coords_for_loss
        )
        total_coord_mae_metric += coord_mae_metric

    def average_a_loss(loss):
        return (
            loss / num_contours_to_compare
            if num_contours_to_compare > 0
            else torch.tensor(0.0)
        )

    avg_command_loss = average_a_loss(total_command_loss)
    avg_coord_loss = average_a_loss(total_coord_loss)
    avg_signed_area_loss = average_a_loss(total_signed_area_loss)
    avg_alignment_loss = average_a_loss(total_alignment_loss)
    avg_coord_mae_metric = average_a_loss(total_coord_mae_metric)
    command_accuracy = (
        torch.tensor(total_correct_cmds / total_cmds)
        if total_cmds > 0
        else torch.tensor(0.0)
    )

    total_loss = (
        VECTOR_LOSS_WEIGHT_COMMAND * avg_command_loss
        + VECTOR_LOSS_WEIGHT_COORD * avg_coord_loss
        + SIGNED_AREA_WEIGHT * avg_signed_area_loss
        + ALIGNMENT_LOSS_WEIGHT * avg_alignment_loss
    )

    return {
        "total_loss": total_loss,
        "command_loss": avg_command_loss.detach(),
        "coord_loss": avg_coord_loss.detach(),
        "signed_area_loss": avg_signed_area_loss.detach(),
        "alignment_loss": avg_alignment_loss.detach(),
        "command_accuracy_metric": command_accuracy,
        "coordinate_mae_metric": avg_coord_mae_metric.detach(),
    }


def abs_signed_area_loss(true_points, pred_points):
    """
    Calculates the signed area of a polygon using the Shoelace formula.
    The input should be a tensor of shape (N, 2) where N is the number of vertices.
    This used to contain code to handle the case of contours with less than three
    points, but we don't need to do that now as such contours are filtered out
    in svgglyph.py and are not in the dataset.
    """

    def compute_signed_area(points):
        # Compute signed area without branching on shape
        # Slices naturally handle small tensors: x[:-1] is empty if len(x) <= 1
        x = points[:, 0]
        y = points[:, 1]

        signed_area = 0.5 * (
            torch.sum(x[:-1] * y[1:])
            + x[-1:] @ y[:1]  # Last x times first y (becomes empty if n=0)
            - torch.sum(y[:-1] * x[1:])
            - y[-1:] @ x[:1]  # Last y times first x
        )

        return signed_area

    true_signed_area = compute_signed_area(true_points)
    pred_signed_area = compute_signed_area(pred_points)

    return torch.abs(true_signed_area - pred_signed_area)


def alignment_loss(
    pred_coords, x_alignment_sets, y_alignment_sets, device, gt_command_for_loss
):
    """
    Calculates a loss based on the variance of coordinates that should be aligned.
    """
    total_x_variance = torch.tensor(0.0, device=device)
    total_y_variance = torch.tensor(0.0, device=device)

    # On-curve points are in the first two columns
    on_curve_points = pred_coords[:, 0:2]

    # We need to filter out points that are not part of the sequence, e.g. EOS padding
    valid_nodes_mask = torch.argmax(
        gt_command_for_loss, dim=-1
    ) != NodeCommand.encode_command("EOS")
    num_valid_nodes = valid_nodes_mask.sum().item()

    # X-alignments
    for alignment_set in x_alignment_sets:
        # Map node indices to row indices in the `_for_loss` tensors.
        # The sequence (after SOS, which filter out in align_sequences)
        # is [M, Cmd for node 0, Cmd for node 1, ...],
        # so the command for node `i` is at row `i+1`.
        row_indices = [idx + 1 for idx in alignment_set]
        valid_set = [idx for idx in row_indices if idx < num_valid_nodes]
        if len(valid_set) > 1:
            # Gather the x-coordinates for the aligned points
            aligned_x_coords = on_curve_points[valid_set, 0]
            total_x_variance += torch.var(aligned_x_coords)

    # Y-alignments
    for alignment_set in y_alignment_sets:
        row_indices = [idx + 1 for idx in alignment_set]
        valid_set = [idx for idx in row_indices if idx < num_valid_nodes]
        if len(valid_set) > 1:
            # Gather the y-coordinates for the aligned points
            aligned_y_coords = on_curve_points[valid_set, 1]
            total_y_variance += torch.var(aligned_y_coords)

    return total_x_variance + total_y_variance


def coordinate_width_mask(commands: torch.Tensor, coords: torch.Tensor):
    device = commands.device
    command_indices = torch.argmax(commands, dim=-1)

    # Create a lookup tensor for coordinate counts
    arg_counts_list = [NodeCommand.grammar[cmd] for cmd in NodeCommand.grammar]
    arg_counts = torch.tensor(arg_counts_list, device=device)

    # Get the number of relevant coordinates for each timestep
    num_relevant_coords = arg_counts[command_indices]

    # Create the mask
    coord_mask = torch.arange(
        coords.shape[1], device=device
    ) < num_relevant_coords.unsqueeze(1)
    return coord_mask


def masked_coordinate_loss(
    device, gt_command_for_loss, gt_coords_for_loss, pred_coords_for_loss
):
    coord_mask = coordinate_width_mask(gt_command_for_loss, gt_coords_for_loss)

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


def masked_coordinate_mae_metric(
    device, gt_command_for_loss, gt_coords_for_loss, pred_coords_for_loss
):
    """
    Calculate Mean Absolute Error (MAE) for coordinates, masked by command type.
    This function computes the MAE only for the coordinates that are relevant
    based on the command type at each timestep. It is scaled by 512 as the
    coordinates are normalized in the range [-1, 1], and we want to understand
    how much they differ in "mask space" space of size 512x512.
    """
    coord_mask = coordinate_width_mask(gt_command_for_loss, gt_coords_for_loss)
    elementwise_coord_error = torch.abs(
        pred_coords_for_loss * 256.0 - gt_coords_for_loss * 256.0
    )
    masked_coord_error = elementwise_coord_error * coord_mask
    num_coords_in_metric = coord_mask.sum()
    if num_coords_in_metric > 0:
        coord_mae = masked_coord_error.sum() / num_coords_in_metric
    else:
        coord_mae = torch.tensor(0.0, device=device)
    return coord_mae


def align_sequences(
    device,
    gt_sequence_norm,
    pred_command,
    pred_coords_norm,
):
    # Step 1: Align sequences in relative space (the original, simple way)
    gt_command_all, _ = NodeCommand.split_tensor(gt_sequence_norm)

    gt_command_for_loss = gt_command_all[1:]
    pred_command_for_loss = pred_command

    # Step 2: Unroll GT and Predictions to get absolute coordinates for loss

    # Unroll the full GT sequence and take the part needed for loss ([1:])
    abs_gt_sequence = NodeCommand.unroll_relative_coordinates(gt_sequence_norm)
    _, abs_gt_coords_all = NodeCommand.split_tensor(abs_gt_sequence)
    gt_coords_for_loss = abs_gt_coords_all[1:]

    # To unroll predictions, we need a full sequence.
    # It starts with SOS (from GT) and then the model's predictions.
    sos_token = gt_sequence_norm[0:1, :]
    predicted_relative_sequence = torch.cat([pred_command, pred_coords_norm], dim=-1)
    full_predicted_relative_sequence = torch.cat(
        [sos_token, predicted_relative_sequence], dim=0
    )

    abs_pred_sequence = NodeCommand.unroll_relative_coordinates(
        full_predicted_relative_sequence
    )
    _, abs_pred_coords_all = NodeCommand.split_tensor(abs_pred_sequence)

    # The part of the unrolled prediction that corresponds to the model's output
    # is from index 1 onwards (to match the GT slice).
    pred_coords_for_loss = abs_pred_coords_all[1:]

    # Step 3: Pad if necessary (e.g., for autoregressive validation)
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
            eos_index = NodeCommand.encode_command("EOS")
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


def predictions_to_image_space(
    outputs: ModelResults, gt_contours: List[GroundTruthContour]
):
    # For visualization, we need to convert the predicted normalized coordinates
    # back to image space.
    pred_commands_list = outputs.pred_commands
    pred_coords_norm_list = outputs.pred_coords_norm
    contour_boxes = outputs.contour_boxes

    pred_commands_and_coords_img_space = []
    for i in range(len(pred_commands_list)):
        pred_cmd = pred_commands_list[i].detach().cpu()
        pred_coords_norm = pred_coords_norm_list[i].detach().cpu()
        box = contour_boxes[i].detach().cpu()

        pred_sequence_norm = torch.cat([pred_cmd, pred_coords_norm], dim=-1)
        # We need a full sequence to convert back to image space, so prepend SOS
        sos_token = NodeCommand.image_space_to_mask_space(
            gt_contours[i]["sequence"].cpu(), box
        )[0:1, :]
        full_pred_sequence_norm = torch.cat([sos_token, pred_sequence_norm], dim=0)

        # Convert back to image space for visualization
        pred_sequence_img_space = NodeCommand.mask_space_to_image_space(
            full_pred_sequence_norm, box
        )
        # We only need the parts that were actually predicted
        pred_commands_and_coords_img_space.append(pred_sequence_img_space[1:])
    return pred_commands_and_coords_img_space


def dump_debug_sequences(
    writer, global_step, batch_idx, gt_contours, outputs: ModelResults, loss_values
):
    """For debugging, dump ground truth and predicted sequences"""
    pred_commands_and_coords_img_space = predictions_to_image_space(
        outputs, gt_contours
    )
    pred_glyph = NodeGlyph.decode(pred_commands_and_coords_img_space, NodeCommand)
    debug_string = SVGGlyph.from_node_glyph(pred_glyph).to_svg_string()
    gt_glyph = NodeGlyph.decode([x["sequence"].cpu() for x in gt_contours], NodeCommand)
    gt_nodes = gt_glyph.to_debug_string()
    gt_debug_string = SVGGlyph.from_node_glyph(gt_glyph).to_svg_string()
    # very_debug(outputs, gt_contours, writer, global_step, batch_idx)

    writer.add_text(
        f"SVG/Debug_{batch_idx}",
        f"""
GT: {gt_debug_string}
Pred: {debug_string}
GT Nodes: {gt_nodes}
Pred Nodes: {pred_glyph.to_debug_string()}
Command loss: {loss_values['command_loss'].item():.4f}
Coord loss: {loss_values['coord_loss'].item():.4f}
""",
        global_step,
    )


def very_debug(
    outputs: ModelResults,
    gt_contours: List[GroundTruthContour],
    writer,
    global_step,
    batch_idx,
):
    # Keep this aligned with the losses function
    pred_commands_list = outputs.pred_commands
    pred_coords_norm_list = outputs.pred_coords_norm
    contour_boxes = outputs.contour_boxes

    num_contours_to_compare = min(len(gt_contours), len(pred_commands_list))
    device = outputs.pred_commands[0].device
    for i in range(num_contours_to_compare):
        pred_command = pred_commands_list[i]
        pred_coords_norm = pred_coords_norm_list[i]
        box = contour_boxes[i]

        # Convert GT sequence from image space to normalized mask space
        gt_sequence_img_space = gt_contours[i]["sequence"]
        gt_sequence_norm = NodeCommand.image_space_to_mask_space(
            gt_sequence_img_space, box
        )

        (
            gt_command_for_loss,
            gt_coords_for_loss,
            pred_command_for_loss,
            pred_coords_for_loss,
        ) = align_sequences(
            device,
            gt_sequence_norm,
            pred_command,
            pred_coords_norm,
        )

        # Denormalize from [-1, 1] to [0, 512] for debugging
        gt_coords_denorm = (gt_coords_for_loss.detach().cpu() + 1) / 2 * 512
        pred_coords_denorm = (pred_coords_for_loss.detach().cpu() + 1) / 2 * 512

        gt_contour_dump = ""
        commands = (
            gt_command_for_loss[:, : NodeCommand.command_width].argmax(dim=-1).tolist()
        )
        gt_commands_debug = [NodeCommand.decode_command(cmd) for cmd in commands]
        # Dump the masked coordinates as well
        gt_coords_debug = gt_coords_denorm.tolist()
        for cmd, coord in zip(gt_commands_debug, gt_coords_debug):
            num_coords = NodeCommand.grammar[cmd]
            gt_contour_dump += (
                f"{cmd} "
                + " ".join([f"{coord[j]:.2f}, " for j in range(0, num_coords)])
                + "\n"
            )
        # Now do the same for the predicted
        pred_contour_dump = ""
        commands = (
            pred_command_for_loss[:, : NodeCommand.command_width]
            .argmax(dim=-1)
            .tolist()
        )
        pred_commands_debug = [NodeCommand.decode_command(cmd) for cmd in commands]
        pred_coords_debug = pred_coords_denorm.tolist()
        command_loss = F.cross_entropy(
            pred_command_for_loss.unsqueeze(0).permute(0, 2, 1),
            gt_command_for_loss.unsqueeze(0).argmax(dim=-1),
            label_smoothing=0.1,
        )
        coord_loss = masked_coordinate_loss(
            device, gt_command_for_loss, gt_coords_for_loss, pred_coords_for_loss
        )
        for cmd, coord in zip(pred_commands_debug, pred_coords_debug):
            num_coords = NodeCommand.grammar[cmd]
            pred_contour_dump += (
                f"{cmd} "
                + " ".join([f"{coord[j]:.2f}, " for j in range(0, num_coords)])
                + "\n"
            )
        writer.add_text(
            f"SVG/Debug_{batch_idx}_contour_{i}",
            f"""
GT: 
{gt_contour_dump}
Pred: 
{pred_contour_dump}
Command loss: {command_loss.item():.4f}
Coord loss: {coord_loss.item():.4f}
    """,
            global_step,
        )
