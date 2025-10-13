import torch
from torch.utils.data import DataLoader
import pydiffvg
import os
from glyphogen.dataset import collate_fn, PretrainGlyphDataset
from glyphogen.rasterizer import rasterize_batch
from glyphogen.model import calculate_masked_coordinate_loss
from glyphogen.glyph import NODE_GLYPH_COMMANDS
import torch.nn.functional as F


def save_image(tensor, filename, ix=0):
    if not os.path.exists("debug_gradients"):
        os.makedirs("debug_gradients")
    # check if tensor is B, C, H, W
    if len(tensor.shape) == 4:
        img_to_save = tensor[ix].repeat(3, 1, 1).permute(1, 2, 0)
    else:
        img_to_save = tensor.repeat(3, 1, 1).permute(1, 2, 0)

    pydiffvg.imwrite(img_to_save.cpu(), f"debug_gradients/{filename}", gamma=2.2)


def main():
    """
    A script to debug gradient flow in the differentiable rasterizer.
    It loads a batch, deforms the true coordinates, and then uses an
    optimizer to try to recover the original coordinates.
    """
    RENDER_SIZE = 64
    LOSS_SCALE = 1.0
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure we get a single character batch for simplicity
    train_dataset = PretrainGlyphDataset(
        ["NotoSans[wdth,wght].ttf"] * 2, ["a"] * 2, is_train=True
    )
    train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=collate_fn)

    # Take a single batch from the dataset
    (_, _), (cmds, true_coords) = next(iter(train_loader))
    cmds = cmds.to(device)
    true_coords = true_coords.to(device)

    # Generate the target raster using the same pipeline
    with torch.no_grad():
        target_rasters = rasterize_batch(cmds, true_coords, img_size=RENDER_SIZE)
    target_rasters = target_rasters.to(device)

    # Save the target image
    save_image(target_rasters, "target.png")

    # Deform the true coordinates slightly
    deformed_coords = (
        true_coords.clone() + torch.randn_like(true_coords) * 0.02
    ).detach()
    deformed_coords.requires_grad = True

    # Save the initial raster
    initial_raster = rasterize_batch(cmds, deformed_coords, img_size=RENDER_SIZE)
    save_image(initial_raster, "initial.png")

    # Optimizer
    optimizer = torch.optim.Adam([deformed_coords], lr=1e-3)
    arg_counts = torch.tensor(list(NODE_GLYPH_COMMANDS.values()), dtype=torch.long).to(
        device
    )

    print("Starting optimization...")
    for i in range(500):
        optimizer.zero_grad()

        # Render the image
        rendered_images = rasterize_batch(
            cmds, deformed_coords, seed=i, img_size=RENDER_SIZE
        )

        # Compute the loss
        loss = F.mse_loss(rendered_images.to(device), target_rasters) * LOSS_SCALE

        # Backpropagate
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Compute the mse loss of the coordinates
        coord_mse = calculate_masked_coordinate_loss(
            cmds, true_coords, deformed_coords, arg_counts
        )
        # print(deformed_coords)

        if i % 10 == 0:
            print(
                f"Iteration {i}, Raster loss: {loss.item()}, Coord MSE: {coord_mse.item()}"
            )
            save_image(rendered_images, f"iter_{i:03d}.png")

    print("Optimization finished.")

    # Save the final raster
    final_raster = rasterize_batch(cmds, deformed_coords, img_size=RENDER_SIZE)
    save_image(final_raster, "final.png")

    final_loss = F.mse_loss(final_raster.to(device), target_rasters)
    print(f"Final Loss: {final_loss.item()}")


if __name__ == "__main__":
    main()
