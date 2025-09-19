import random
import torch
from .glyph import NodeGlyph, NodeCommand
import numpy as np
from .rasterizer import rasterize_batch
from .command_defs import MAX_COORDINATE


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
    if isinstance(test_loader, list):
        random_batch = random.sample(test_loader, 1)[0]
    else:
        random_batch = next(iter(test_loader))
    with torch.no_grad():
        (inputs, y) = random_batch
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(inputs)
        raster_loss_fn = torch.nn.MSELoss()

        vector_rendered_images = rasterize_batch(
            outputs["command"], outputs["coord_absolute"] / MAX_COORDINATE
        ).to(device)

        raster_loss = raster_loss_fn(inputs["raster_image"], vector_rendered_images)
        writer.add_scalar("Metrics/raster_metric", 1.0 - raster_loss.item(), epoch)

        for i in range(min(num_images, vector_rendered_images.shape[0])):
            true_raster = inputs["raster_image"][i]
            predicted_raster = vector_rendered_images[i]

            # Create a 3-channel image for overlay
            # true_raster and predicted_raster are (1, H, W)
            # We want (3, H, W) for add_image
            
            # Red channel for true raster (alpha 0.5)
            red_channel = true_raster * 0.5
            # Green channel for predicted raster (alpha 0.5)
            green_channel = predicted_raster * 0.5
            
            # Combine into an RGB image
            # Overlapping areas will be red + green = yellow
            overlay_image = torch.cat([red_channel, green_channel, torch.zeros_like(red_channel)], dim=0)
            
            writer.add_image(
                f"Pretrain_Images/Overlay_{i}",
                overlay_image,
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
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output = model(inputs)
        else:
            (style_image, glyph_id, target_sequence_input), y = batch
            style_image, glyph_id, target_sequence_input = (
                style_image.to(device),
                glyph_id.to(device),
                target_sequence_input.to(device),
            )
            output = model((style_image, glyph_id, target_sequence_input))

        command_output = output["command"]
        coord_output = output["coord_absolute"]
        for i in range(min(num_samples, command_output.shape[0])):
            command_tensor = command_output[i].detach().cpu().numpy()
            coord_tensor = coord_output[i].detach().cpu().numpy()
            # command_keys = list(NodeCommand.grammar.keys())
            # svg_string = " ".join(
            #     [
            #         command_keys[np.argmax(command_tensor[i])]
            #         for i in range(command_tensor.shape[0])
            #     ]
            # )
            try:
                decoded_glyph = NodeGlyph.from_numpy(
                    command_tensor, coord_tensor, relative=False
                )
                svg_string = decoded_glyph.to_svg_glyph().to_svg_string()
            except Exception as e:
                svg_string = f"Couldn't generate SVG: {e}"

            writer.add_text(f"SVG/Generated_{i}", svg_string, epoch)
    writer.flush()
