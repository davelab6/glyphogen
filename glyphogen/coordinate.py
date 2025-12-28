from typing import List, Tuple
import torch
from .command_defs import MAX_COORDINATE
from .hyperparameters import GEN_IMAGE_SIZE


def to_image_space(pts):
    """
    Transforms a coordinate pair from font coordinate space to image space,
    matching the transformation used in rasterize_batch.
    """
    x, y = pts
    x /= MAX_COORDINATE
    y /= MAX_COORDINATE

    img_size = GEN_IMAGE_SIZE[0]
    baseline = 0.33 * img_size

    # Add 100 units to X to stop italics from being cut off
    x += 100.0 / MAX_COORDINATE

    # Apply scaling and baseline translation
    # This seems to scale the glyph to fit roughly in the top 2/3 of the image
    transformed_x = x * img_size * 2.0 / 3.0
    transformed_y = y * img_size * 2.0 / 3.0
    # Add the baseline to the Y coordinate.
    transformed_y += baseline
    # Apply vertical flip for image coordinates (Y-down)
    transformed_y = img_size - transformed_y
    return transformed_x, transformed_y


def get_bounds(points: List[Tuple[float, float]]) -> "ImageSpaceBbox":
    """
    Computes the bounding box of the points in image space.
    Returns [x_min, y_min, x_max, y_max]
    """
    if not points:
        return ImageSpaceBbox([0.0, 0.0, 0.0, 0.0])

    x_min = min(points, key=lambda p: p[0])[0]
    x_max = max(points, key=lambda p: p[0])[0]
    y_min = min(points, key=lambda p: p[1])[1]
    y_max = max(points, key=lambda p: p[1])[1]

    return ImageSpaceBbox([x_min, y_min, x_max, y_max])


class ImageSpaceBbox(list):
    pass


@torch.compile
def image_space_to_mask_space(coords_img_space, box):
    """
    Normalizes image-space coordinates to the model's internal [-1, 1] range
    relative to a given bounding box (also in image space).

    We also convert the handle coordinates from absolute to relative coordinates
    for easier learning.
    """
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1

    width = width if width > 0 else 1
    height = height if height > 0 else 1

    normalized = coords_img_space.clone()
    normalized[:, 0] = (coords_img_space[:, 0] - x1) / width
    normalized[:, 1] = (coords_img_space[:, 1] - y1) / height

    # Handles are currently absolute coordinates. Normalize them too.
    normalized[:, 2] = (coords_img_space[:, 2] - x1) / width
    normalized[:, 3] = (coords_img_space[:, 3] - y1) / height
    normalized[:, 4] = (coords_img_space[:, 4] - x1) / width
    normalized[:, 5] = (coords_img_space[:, 5] - y1) / height
    # # And now convert them to relative coordinates for easier learning
    # normalized[:, 2] -= normalized[:, 0]
    # normalized[:, 3] -= normalized[:, 1]
    # normalized[:, 4] -= normalized[:, 0]
    # normalized[:, 5] -= normalized[:, 1]

    return (normalized * 2) - 1


@torch.compile
def mask_space_to_image_space(coords_norm, box):
    """
    Denormalizes coordinates from the model's internal [-1, 1] range back
    to image space relative to a given bounding box.
    """
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1

    coords_0_1 = (coords_norm + 1) / 2

    denormalized = coords_0_1.clone()
    # # Convert the handles back from relative to absolute coordinates
    # denormalized[:, 2] = coords_0_1[:, 2] + coords_0_1[:, 0]
    # denormalized[:, 3] = coords_0_1[:, 3] + coords_0_1[:, 1]
    # denormalized[:, 4] = coords_0_1[:, 4] + coords_0_1[:, 0]
    # denormalized[:, 5] = coords_0_1[:, 5] + coords_0_1[:, 1]

    # Denormalize the position back into image space
    denormalized[:, 0] = coords_0_1[:, 0] * width + x1
    denormalized[:, 1] = coords_0_1[:, 1] * height + y1
    denormalized[:, 2] = denormalized[:, 2] * width + x1
    denormalized[:, 3] = denormalized[:, 3] * height + y1
    denormalized[:, 4] = denormalized[:, 4] * width + x1
    denormalized[:, 5] = denormalized[:, 5] * height + y1

    return denormalized
