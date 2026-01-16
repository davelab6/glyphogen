import torch
from torch.utils.data import DataLoader
from glyphogen.dataset import get_vectorizer_data, collate_fn
from glyphogen.rasterizer import rasterize_batch
import pydiffvg
import torch.nn.functional as F


def main():
    """
    Test script for loading a batch from the pre-train dataset and preparing it for rasterization.
    """
    train_dataset, _ = get_vectorizer_data()
    train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=collate_fn)
    RENDER_SIZE = 64

    # Take a single batch from the dataset
    for batch in train_loader:
        (rasters, seq), (cmds, coords) = batch
        break

    rasters_small = F.interpolate(rasters, size=(RENDER_SIZE, RENDER_SIZE), mode="area")
    rendered_images = rasterize_batch(cmds, coords, img_size=RENDER_SIZE)

    # imwrite wants 3 channels, so we convert our single channel image to RGB
    img_to_save = rasters_small[0].repeat(3, 1, 1).permute(1, 2, 0)
    pydiffvg.imwrite(img_to_save.cpu(), "target.png", gamma=2.2)
    img_to_save = rendered_images[0].repeat(3, 1, 1).permute(1, 2, 0)
    pydiffvg.imwrite(img_to_save.cpu(), "produced.png", gamma=2.2)

    loss = torch.nn.functional.mse_loss(rendered_images, rasters_small)
    print(f"Raster loss: {loss.item()}")

    # Create a comparison image
    for ix, (target_raster, produced_raster) in enumerate(
        zip(rasters_small, rendered_images)
    ):
        # Red channel for target, Green for produced
        red_channel = target_raster.squeeze()
        green_channel = produced_raster.squeeze()
        blue_channel = torch.zeros_like(red_channel)

        # Alpha is 50% of the intensity
        alpha_channel = torch.clamp(red_channel + green_channel, 0.0, 1.0) * 0.5

        comparison_image = torch.stack(
            [red_channel, green_channel, blue_channel, alpha_channel], dim=-1
        )

        pydiffvg.imwrite(comparison_image.cpu(), f"comparison-{ix:02d}.png")


if __name__ == "__main__":
    main()
