import random
import torch

from glyphogen.hyperparameters import BATCH_SIZE
from .glyph import NodeGlyph, NodeCommand
import numpy as np
from .rasterizer import rasterize_batch
from .command_defs import MAX_COORDINATE, NODE_GLYPH_COMMANDS
from .losses import find_eos


def log_vectorizer_rasters(model, test_loader, writer, epoch, num_images=BATCH_SIZE):
    device = next(model.parameters()).device
    model.eval()
    if isinstance(test_loader, list):
        random_batch = random.sample(test_loader, 1)[0]
    else:
        random_batch = next(iter(test_loader))
    with torch.no_grad():
        (inputs, y) = random_batch
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(inputs)

        vector_rendered_images = rasterize_batch(
            outputs["command"], outputs["coord_absolute"] / MAX_COORDINATE
        ).to(device)

        for i in range(min(num_images, vector_rendered_images.shape[0])):
            true_raster = inputs["raster_image"][i]
            predicted_raster = vector_rendered_images[i]

            # Create a 3-channel image for overlay
            # true_raster and predicted_raster are (1, H, W)
            # We want (3, H, W) for add_image
            # Desired colors:
            # - Background: white (where both rasters are 1.0)
            # - True raster: green
            # - Predicted raster: red
            # - Overlap: black / blended
            red_channel = true_raster
            green_channel = predicted_raster
            blue_channel = torch.min(true_raster, predicted_raster)

            overlay_image = torch.cat([red_channel, green_channel, blue_channel], dim=0)

            writer.add_image(
                f"Vectorizer_Images/Overlay_{i}",
                overlay_image,
                epoch,
            )
    writer.flush()


def log_svgs(model, test_loader, writer, epoch, num_samples=3):
    if epoch % 5 != 0:
        return

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        batch = next(iter(test_loader))
        (inputs, y) = batch
        inputs = {k: v.to(device) for k, v in inputs.items()}
        output = model(inputs)

        command_output = output["command"]
        coord_output = output["coord_absolute"]
        for i in range(min(num_samples, command_output.shape[0])):
            command_tensor = command_output[i].detach().cpu().numpy()
            coord_tensor = coord_output[i].detach().cpu().numpy()
            command_keys = list(NodeCommand.grammar.keys())
            raw_commands = " ".join(
                [
                    command_keys[np.argmax(command_tensor[i])]
                    for i in range(command_tensor.shape[0])
                ]
            )
            try:
                decoded_glyph = NodeGlyph.from_numpy(
                    command_tensor, coord_tensor, relative=False
                )
                svg_string = decoded_glyph.to_svg_glyph().to_svg_string()
                debug_string = decoded_glyph.to_debug_string()
            except Exception as e:
                svg_string = f"Couldn't generate SVG: {e}"
                debug_string = "Error in decoding glyph."

            writer.add_text(f"SVG/Generated_{i}", svg_string, epoch)
            writer.add_text(f"SVG/Node_{i}", debug_string, epoch)
            writer.add_text(f"SVG/Raw_{i}", raw_commands, epoch)
    writer.flush()


def init_confusion_matrix_state():
    """Initializes a state dictionary for collecting confusion matrix data."""
    return {
        "all_true_indices": [],
        "all_pred_indices": [],
        "all_masks": [],
    }


def collect_confusion_matrix_data(state, outputs, y):
    """Collects prediction and ground truth data from a validation batch."""
    true_command, _ = y
    pred_command = outputs["command"]

    # Create sequence mask to ignore padding
    true_eos_idx = find_eos(true_command)
    batch_size, seq_len, _ = true_command.shape
    device = true_command.device
    indices = torch.arange(seq_len, device=device).expand(batch_size, -1)
    sequence_mask = indices < true_eos_idx.unsqueeze(1)

    true_indices = torch.argmax(true_command, dim=-1)
    pred_indices = torch.argmax(pred_command, dim=-1)

    state["all_true_indices"].append(true_indices.detach().cpu())
    state["all_pred_indices"].append(pred_indices.detach().cpu())
    state["all_masks"].append(sequence_mask.detach().cpu())


def log_confusion_matrix(state, writer, epoch):
    """Computes and logs the confusion matrix at the end of an epoch."""
    if not state["all_true_indices"]:
        return

    true_indices = torch.cat(state["all_true_indices"])
    pred_indices = torch.cat(state["all_pred_indices"])
    mask = torch.cat(state["all_masks"])

    masked_true = torch.masked_select(true_indices, mask)
    masked_pred = torch.masked_select(pred_indices, mask)

    num_classes = len(NODE_GLYPH_COMMANDS)
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int32)

    # This is a bit slow, but simple and reliable
    for i in range(masked_true.shape[0]):
        true_label = masked_true[i]
        pred_label = masked_pred[i]
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
