from glyphogen.typing import ModelResults
import numpy as np
import torch
from torchvision.utils import draw_bounding_boxes

from glyphogen.losses import align_sequences
from glyphogen.rasterizer import rasterize_batch

from .command_defs import NodeCommand
from .nodeglyph import NodeGlyph
from .svgglyph import SVGGlyph


def log_vectorizer_outputs(
    model, data_loader, writer, epoch, num_images=4, log_svgs=True
):
    """
    Unified logging function for vectorizer outputs.

    Runs inference on images using autoregressive generation and logs:
    - Raster visualizations (overlay of predicted vs ground truth)
    - SVG outputs (generated vector graphics and debug info)

    Args:
        model: The vectorizer model
        data_loader: DataLoader for validation images
        writer: TensorBoard writer
        epoch: Current epoch number
        num_images: Number of images to log
        log_svgs: Whether to also generate and log SVG outputs
    """
    # Only log SVGs every 5 epochs to save storage
    skip_svgs = log_svgs and (epoch % 5 != 0)

    device = next(model.parameters()).device
    model.eval()
    images, _ = next(iter(data_loader))

    with torch.no_grad():
        for i in range(min(num_images, len(images))):
            img = images[i].to(device)

            # Run inference (uses autoregressive generation via gt_targets=None)
            outputs: ModelResults = model(img.unsqueeze(0), gt_targets=None)

            # --- Log Raster Visualization ---
            if outputs.pred_commands:
                # Prepare contour sequences for from_numpy
                contour_sequences = [
                    (
                        outputs.pred_commands[idx],
                        outputs.pred_coords_img_space[idx],
                    )
                    for idx in range(len(outputs.pred_commands))
                ]

                # Decode to NodeGlyph
                predicted_raster = rasterize_batch(
                    [contour_sequences], NodeCommand, device=device
                )[0]
            else:
                # If model predicts no contours, show a blank image
                predicted_raster = torch.ones_like(img[0:1])

            # Create a 3-channel overlay for visualization
            # We want predicted in red, ground truth in green, blue channel empty
            predicted_inv = 1.0 - predicted_raster
            true_inv = 1.0 - img[0:1, :, :]  # Take first channel and keep dims
            zeros = torch.zeros_like(predicted_inv)

            overlay_image = torch.cat([predicted_inv, true_inv, zeros], dim=0)
            writer.add_image(f"Vectorizer_Images/Overlay_{i}", overlay_image, epoch)
            # writer.add_image(f"Vectorizer_Images/Overlay_{i}", predicted_raster, epoch)

            # --- Log SVG Outputs (if enabled and epoch matches) ---
            if log_svgs and not skip_svgs:
                # Prepare contour sequences for from_numpy
                contour_sequences = [
                    (
                        outputs.pred_commands[idx],
                        outputs.pred_coords_img_space[idx],
                    )
                    for idx in range(len(outputs.pred_commands))
                ]

                try:
                    decoded_glyph = NodeGlyph.decode(contour_sequences, NodeCommand)
                    svg_string = SVGGlyph.from_node_glyph(decoded_glyph).to_svg_string()
                    debug_string = decoded_glyph.to_debug_string()
                except Exception as e:
                    svg_string = f"Couldn't generate SVG: {e}"
                    debug_string = "Error in decoding glyph."

                writer.add_text(f"SVG/Generated_{i}", svg_string, epoch)
                writer.add_text(f"SVG/Node_{i}", debug_string, epoch)

                # Log raw command sequence
                command_keys = {
                    k: v for k, v in enumerate(list(NodeCommand.grammar.keys()))
                }
                raw_commands_list = []
                for contour_idx in range(len(outputs.pred_commands)):
                    pred_cmds = outputs.pred_commands[contour_idx]
                    for cmd_idx in range(pred_cmds.shape[0]):
                        cmd_argmax = int(np.argmax(pred_cmds[cmd_idx].cpu().numpy()))
                        cmd_name = command_keys.get(cmd_argmax, "?")
                        raw_commands_list.append(f"{cmd_argmax} ({cmd_name})")
                raw_commands = " ".join(raw_commands_list)
                writer.add_text(f"SVG/Raw_{i}", raw_commands, epoch)

    writer.flush()


def init_confusion_matrix_state():
    """Initializes a state dictionary for collecting confusion matrix data."""

    return {"all_true_indices": [], "all_pred_indices": []}


def collect_confusion_matrix_data(
    state, outputs_list: list[ModelResults], targets_tuple
):
    """Collects prediction and ground truth data from a validation batch."""

    # Iterate over each sample in the batch
    for i in range(len(targets_tuple)):
        y = targets_tuple[i]
        outputs: ModelResults = outputs_list[i]

        gt_contours = y["gt_contours"]
        pred_commands_list = outputs.pred_commands
        pred_coords_norm_list = outputs.pred_coords_norm
        contour_boxes = outputs.contour_boxes

        num_contours_to_compare = min(len(gt_contours), len(pred_commands_list))

        for j in range(num_contours_to_compare):
            pred_command = pred_commands_list[j]
            pred_coords_norm = pred_coords_norm_list[j]
            box = contour_boxes[j]

            # Convert GT sequence from image space to normalized mask space
            gt_sequence_img_space = gt_contours[j]["sequence"]
            gt_sequence_norm = NodeCommand.image_space_to_mask_space(
                gt_sequence_img_space, box
            )

            (
                gt_command_for_loss,
                _,
                pred_command_for_loss,
                _,
            ) = align_sequences(
                gt_sequence_norm.device,
                gt_sequence_norm,
                pred_command,
                pred_coords_norm,
            )

            true_indices = torch.argmax(gt_command_for_loss, dim=-1)
            pred_indices = torch.argmax(pred_command_for_loss, dim=-1)
            state["all_true_indices"].append(true_indices.detach().cpu())
            state["all_pred_indices"].append(pred_indices.detach().cpu())


def log_confusion_matrix(state, writer, epoch):
    """Computes and logs the confusion matrix at the end of an epoch."""
    if not state["all_true_indices"]:
        return

    true_indices = torch.cat(state["all_true_indices"])
    pred_indices = torch.cat(state["all_pred_indices"])
    num_classes = len(NodeCommand.grammar)
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int32)
    for i in range(true_indices.shape[0]):
        true_label = true_indices[i]
        pred_label = pred_indices[i]
        if true_label < num_classes and pred_label < num_classes:
            matrix[true_label, pred_label] += 1

    # Format as Markdown table
    command_names = list(NodeCommand.grammar.keys())
    header = "| True \\ Pred | " + " | ".join(command_names) + " |\n"
    separator = "|--- " * (num_classes + 1) + "|\n"
    body = ""
    for i, name in enumerate(command_names):
        row = f"| **{name}** | "
        for j in range(num_classes):
            row += f"{matrix[i, j].item()} | "
        body += row + "\n"

    markdown_string = header + separator + body

    writer.add_text("Diagnostics/Confusion Matrix", markdown_string, epoch)
    writer.flush()

    # State is reset in the training loop


def log_bounding_boxes(model, data_loader, writer, epoch, num_images=4):
    """Logs images with ground truth and predicted bounding boxes."""
    device = next(model.parameters()).device
    model.eval()

    # Get a batch of data
    # The data loader returns a tuple of image tensors and a tuple of target dicts
    images, targets = next(iter(data_loader))

    # Take only num_images from the batch
    images = images[:num_images]
    targets = targets[:num_images]

    # images is a tuple of tensors. We need to stack them.
    images_for_segmenter = torch.stack(images).to(device)

    # Get predictions
    with torch.no_grad():
        # The segmenter is part of the main model
        predictions = model.segmenter(images_for_segmenter)

    for i in range(num_images):
        img_tensor = images[i]
        gt_target = targets[i]
        pred_target = predictions[i]

        # Prepare image for drawing (convert to uint8, 3 channels)
        img_to_draw = (img_tensor * 255).to(torch.uint8)
        if img_to_draw.shape[0] == 1:
            img_to_draw = img_to_draw.repeat(3, 1, 1)

        # Get GT boxes and labels
        gt_boxes = torch.stack([c["box"] for c in gt_target["gt_contours"]])
        gt_labels = [
            f"GT: {'hole' if c['label']==1 else 'outer'}"
            for c in gt_target["gt_contours"]
        ]

        # Get predicted boxes and labels
        pred_boxes = pred_target["boxes"]
        pred_labels = [
            f"Pred: {'hole' if label==2 else 'outer'} ({(s*100):.0f}%)"
            for label, s in zip(pred_target["labels"], pred_target["scores"])
        ]

        # Draw boxes on the image
        # Draw predicted first, then GT, so GT is on top in case of overlap
        img_with_boxes = draw_bounding_boxes(
            img_to_draw, boxes=pred_boxes, labels=pred_labels, colors="red"
        )
        img_with_boxes = draw_bounding_boxes(
            img_with_boxes, boxes=gt_boxes, labels=gt_labels, colors="green"
        )

        writer.add_image(f"Bounding_Boxes/Image_{i}", img_with_boxes, epoch)

    writer.flush()
