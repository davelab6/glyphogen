import random
from glyphogen.coordinate import to_image_space
import torch
from torchvision.utils import draw_bounding_boxes

from glyphogen.hyperparameters import BATCH_SIZE
from .glyph import NodeGlyph, NodeCommand
import numpy as np
from .command_defs import NODE_GLYPH_COMMANDS


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
            outputs = model(img.unsqueeze(0), gt_targets=None)

            # --- Log Raster Visualization ---
            if outputs["pred_commands"]:
                # Prepare contour sequences for from_numpy
                contour_sequences = [
                    (
                        outputs["pred_commands"][idx],
                        outputs["pred_coords_img_space"][idx],
                    )
                    for idx in range(len(outputs["pred_commands"]))
                ]

                # Decode to NodeGlyph
                decoded_glyph = NodeGlyph.from_numpy(contour_sequences)

                # Convert to SVG and rasterize
                svg_glyph = decoded_glyph.to_svg_glyph()
                kurbopy_contours = svg_glyph.to_bezpaths()

                # Flatten and transform points for rasterization
                all_points = []
                for path in kurbopy_contours:
                    font_space_points = [(pt.x, pt.y) for pt in path.flatten(1.0)]
                    all_points.extend(font_space_points)

                if all_points:
                    points_tensor = torch.tensor(all_points)
                    image_space_points = to_image_space(points_tensor)
                    # Create PIL image for rasterization
                    from PIL import Image, ImageDraw

                    pil_img = Image.new("L", (img.shape[-1], img.shape[-2]), 0)
                    draw = ImageDraw.Draw(pil_img)

                    # Draw each contour
                    start_idx = 0
                    for path in kurbopy_contours:
                        path_points = [(pt.x, pt.y) for pt in path.flatten(1.0)]
                        num_points = len(path_points)
                        contour_points = image_space_points[
                            start_idx : start_idx + num_points
                        ].tolist()
                        if len(contour_points) >= 2:
                            draw.polygon(contour_points, fill=1)
                        start_idx += num_points

                    predicted_raster = (
                        torch.from_numpy(np.array(pil_img, dtype=np.float32) / 255.0)
                        .unsqueeze(0)
                        .to(device)
                    )
                else:
                    predicted_raster = torch.ones_like(img[0:1])
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

            # --- Log SVG Outputs (if enabled and epoch matches) ---
            if log_svgs and not skip_svgs:
                # Prepare contour sequences for from_numpy
                contour_sequences = [
                    (
                        outputs["pred_commands"][idx],
                        outputs["pred_coords_img_space"][idx],
                    )
                    for idx in range(len(outputs["pred_commands"]))
                ]

                try:
                    decoded_glyph = NodeGlyph.from_numpy(contour_sequences)
                    svg_string = decoded_glyph.to_svg_glyph().to_svg_string()
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
                for contour_idx in range(len(outputs["pred_commands"])):
                    pred_cmds = outputs["pred_commands"][contour_idx]
                    for cmd_idx in range(pred_cmds.shape[0]):
                        cmd_argmax = np.argmax(pred_cmds[cmd_idx].cpu().numpy())
                        cmd_name = command_keys.get(cmd_argmax, "?")
                        raw_commands_list.append(f"{cmd_argmax} ({cmd_name})")
                raw_commands = " ".join(raw_commands_list)
                writer.add_text(f"SVG/Raw_{i}", raw_commands, epoch)

    writer.flush()


def init_confusion_matrix_state():
    """Initializes a state dictionary for collecting confusion matrix data."""

    return {"all_true_indices": [], "all_pred_indices": []}


def collect_confusion_matrix_data(state, outputs_list, targets_tuple):
    """Collects prediction and ground truth data from a validation batch."""

    # Iterate over each sample in the batch
    for i in range(len(targets_tuple)):
        y = targets_tuple[i]
        outputs = outputs_list[i]

        gt_contours = y["gt_contours"]
        pred_commands_list = outputs.get("pred_commands", [])

        num_contours_to_compare = min(len(gt_contours), len(pred_commands_list))

        for j in range(num_contours_to_compare):
            pred_command = pred_commands_list[j]
            gt_sequence = gt_contours[j]["sequence"]

            command_width = len(NODE_GLYPH_COMMANDS)
            gt_command = gt_sequence[:, :command_width]

            # Align sequence lengths for comparison
            gt_len = gt_command.shape[0]
            pred_len = pred_command.shape[0]

            if pred_len > gt_len:
                pred_command = pred_command[:gt_len]
            elif gt_len > pred_len:
                pad_len = gt_len - pred_len
                pad_command = torch.zeros(
                    pad_len, pred_command.shape[1], device=pred_command.device
                )
                pad_command[:, -1] = 1  # Pad with EOS token
                pred_command = torch.cat([pred_command, pad_command], dim=0)

            true_indices = torch.argmax(gt_command, dim=-1)
            pred_indices = torch.argmax(pred_command, dim=-1)

            state["all_true_indices"].append(true_indices.detach().cpu())
            state["all_pred_indices"].append(pred_indices.detach().cpu())


def log_confusion_matrix(state, writer, epoch):
    """Computes and logs the confusion matrix at the end of an epoch."""
    if not state["all_true_indices"]:
        return

    true_indices = torch.cat(state["all_true_indices"])
    pred_indices = torch.cat(state["all_pred_indices"])
    num_classes = len(NODE_GLYPH_COMMANDS)
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int32)
    for i in range(true_indices.shape[0]):
        true_label = true_indices[i]
        pred_label = pred_indices[i]
        if true_label < num_classes and pred_label < num_classes:
            matrix[true_label, pred_label] += 1

    # Format as Markdown table
    command_names = list(NODE_GLYPH_COMMANDS.keys())
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
            f"Pred: {'hole' if l==2 else 'outer'} ({(s*100):.0f}%)"
            for l, s in zip(pred_target["labels"], pred_target["scores"])
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
