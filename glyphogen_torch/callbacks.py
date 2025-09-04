import torch
from .glyph import NodeGlyph, NodeCommand
import numpy as np
from .rasterizer import rasterize_batch


def log_images(model, test_loader, writer, epoch, pre_train=False, num_images=3):
    if pre_train:
        return

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_images:
                break

            (style_image, glyph_id, target_sequence), y = batch
            style_image, glyph_id, target_sequence = (
                style_image.to(device),
                glyph_id.to(device),
                target_sequence.to(device),
            )
            true_raster = y["raster"].to(device)

            generated_raster = model((style_image, glyph_id, target_sequence))["raster"]

            writer.add_image(f"Images/True_{i}", true_raster.squeeze(0), epoch)
            writer.add_image(
                f"Images/Generated_{i}", generated_raster.squeeze(0), epoch
            )
    writer.flush()


def log_pretrain_rasters(model, test_loader, writer, epoch, num_images=3):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        (inputs, y) = next(iter(test_loader))
        (raster_image_input, target_sequence_input) = inputs
        raster_image_input, target_sequence_input = raster_image_input.to(
            device
        ), target_sequence_input.to(device)

        outputs = model((raster_image_input, target_sequence_input))
        vector_rendered_images = rasterize_batch(
            outputs["command"], outputs["coord"]
        ).to(device)

        for i in range(num_images):
            writer.add_image(f"Pretrain_Images/True_{i}", raster_image_input[i], epoch)
            writer.add_image(
                f"Pretrain_Images/Generated_{i}",
                vector_rendered_images[i],
                epoch,
            )
    writer.flush()


def log_svgs(model, test_loader, writer, epoch, pre_train=False, num_samples=3):
    if epoch % 5 != 0:
        return

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        batch = next(iter(test_loader))
        if pre_train:
            (inputs, y) = batch
            (raster_image_input, target_sequence_input) = inputs
            raster_image_input, target_sequence_input = raster_image_input.to(
                device
            ), target_sequence_input.to(device)
            output = model((raster_image_input, target_sequence_input))
        else:
            (style_image, glyph_id, target_sequence_input), y = batch
            style_image, glyph_id, target_sequence_input = (
                style_image.to(device),
                glyph_id.to(device),
                target_sequence_input.to(device),
            )
            output = model((style_image, glyph_id, target_sequence_input))

    
        command_output = output["command"]
        coord_output = output["coord"]
        for i in range(min(num_samples, command_output.shape[0])):
            command_tensor = command_output[i].detach().cpu().numpy()
            coord_tensor = coord_output[i].detach().cpu().numpy()

            try:
                decoded_glyph = NodeGlyph.from_numpy(command_tensor, coord_tensor)
                svg_string = decoded_glyph.to_svg_glyph().to_svg_string()
            except Exception as e:
                svg_string = f"Couldn't generate SVG: {e}"

            writer.add_text(f"SVG/Generated_{i}", svg_string, epoch)
    writer.flush()
