import pydiffvg
import torch

from .command_defs import NodeCommand
from .hyperparameters import GEN_IMAGE_SIZE

command_keys = list(NodeCommand.grammar.keys())
cmd_n_val = command_keys.index("N")
cmd_sos_val = command_keys.index("SOS")
cmd_eos_val = command_keys.index("EOS")


@torch.compiler.disable(recursive=False)
def rasterize_batch(
    contour_sequences,
    representation,
    seed=42,
    img_size=None,
    requires_grad=True,
    device=None,
):
    """Render a batch of glyphs from their per-contour node representation.

    Args:
        contour_sequences: List of contour sequences for each glyph in the batch.
                          Each element is a list of (cmd_tensor, coord_tensor) tuples.
        representation: The command representation class (e.g., NodeCommand) used.
        seed: Random seed for rendering
        img_size: Image size for rendering
        requires_grad: Whether gradients are needed
        device: Device to render on

    Returns:
        Tensor of rendered images (batch_size, 1, img_size, img_size)
    """
    if img_size is None:
        img_size = GEN_IMAGE_SIZE[0]

    # Determine device from first contour's commands
    if device is None:
        if contour_sequences and contour_sequences[0]:
            device = contour_sequences[0][0][0].device
        else:
            device = torch.device("cpu")

    pydiffvg.set_device(device)

    dead_image = torch.ones(1, img_size, img_size, dtype=torch.float32).to(device)
    images = []

    for glyph_contours in contour_sequences:
        if not glyph_contours:
            # print("No contours for this glyph, returning blank image.")
            images.append(dead_image)
            continue

        all_shapes = []

        # Process each contour
        for cmd_tensor, coord_tensor in glyph_contours:
            # Ensure gradients if needed
            if requires_grad:
                coord_tensor = coord_tensor.clone()
                coord_tensor.requires_grad_(True)

            # Pad coordinates if needed
            if coord_tensor.shape[-1] == 2:
                padding = torch.zeros(
                    *coord_tensor.shape[:-1],
                    representation.coordinate_width - 2,
                    device=device,
                    dtype=coord_tensor.dtype,
                )
                coord_tensor = torch.cat([coord_tensor, padding], dim=-1)

            # Extract segments
            points, num_control_points, num_cp_splits, point_splits = (
                representation.tensors_to_segments(
                    cmd_tensor.clone(), coord_tensor.clone()
                )
            )

            # Convert segments to paths
            num_cp_start = 0
            point_start = 0
            for num_cp_end, point_end in zip(num_cp_splits, point_splits):
                num_cp = num_control_points[num_cp_start:num_cp_end]
                path_points = points[point_start:point_end]

                path = pydiffvg.Path(
                    num_control_points=num_cp.to(torch.int32).cpu(),
                    points=path_points.cpu(),
                    is_closed=True,
                )
                all_shapes.append(path)
                num_cp_start = num_cp_end
                point_start = point_end

        # If there are no shapes, return a blank image
        if len(all_shapes) == 0:
            # print("No shapes for this glyph, returning blank image.")
            images.append(dead_image)
            continue

        path_group = pydiffvg.ShapeGroup(
            shape_ids=torch.arange(len(all_shapes)),
            fill_color=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        )
        shape_groups = [path_group]
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            img_size, img_size, all_shapes, shape_groups
        )

        render = pydiffvg.RenderFunction.apply
        try:
            img = render(img_size, img_size, 2, 2, seed, None, *scene_args).to(device)
        except Exception as e:
            print(f"Failed to rasterize: {e}")
            images.append(dead_image)
            continue

        img = torch.max(img) - img
        # Get alpha channel and add channel dimension
        img = img[:, :, 3].unsqueeze(0)
        images.append(img)

    return torch.stack(images)


def rasterize_and_save(contour_sequences, filename="produced.png"):
    """Render a single glyph and save it to a file.

    Args:
        contour_sequences: List of (cmd_tensor, coord_tensor) tuples for one glyph
        filename: Output filename
    """
    # Wrap in a batch of size 1
    images = rasterize_batch([contour_sequences])
    # imwrite expects H, W, C
    pydiffvg.imwrite(images[0].permute(1, 2, 0).cpu(), filename, gamma=1.0)
