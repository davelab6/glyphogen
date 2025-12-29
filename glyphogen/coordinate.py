from typing import List, Tuple
import torch
from .command_defs import NODE_GLYPH_COMMANDS, MAX_COORDINATE, NODE_COMMAND_WIDTH
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
def image_space_to_mask_space(sequence, box):
    """
    Normalizes a sequence's image-space coordinates to the model's internal
    [-1, 1] range relative to a given bounding box. Handles mixed
    absolute (M) and relative (other) commands.
    """
    commands = sequence[:, :NODE_COMMAND_WIDTH]
    coords_img_space = sequence[:, NODE_COMMAND_WIDTH:]

    x1, y1, x2, y2 = box
    width = x2 - x1 if x2 > x1 else 1
    height = y2 - y1 if y2 > y1 else 1

    command_indices = torch.argmax(commands, dim=-1)
    m_index = list(NODE_GLYPH_COMMANDS.keys()).index("M")
    is_m_mask = (command_indices == m_index).unsqueeze(1)

    normalized = coords_img_space.clone()

    # Scale all x-like and y-like coordinates by width and height respectively
    if coords_img_space.shape[1] > 0:
        normalized[:, 0::2] /= width
        normalized[:, 1::2] /= height

    # Additionally translate the absolute 'M' coordinates
    if torch.sum(is_m_mask) > 0:
        m_coords = normalized[is_m_mask.squeeze(1)]
        m_coords[:, 0] -= x1 / width
        m_coords[:, 1] -= y1 / height
        normalized[is_m_mask.squeeze(1)] = m_coords

    # Final scaling to [-1, 1]
    normalized = (normalized * 2) - 1
    
    return torch.cat([commands, normalized], dim=-1)


@torch.compile
def mask_space_to_image_space(sequence, box):
    """
    Denormalizes a sequence's [-1, 1] coordinates back to image space.
    Handles mixed absolute (M) and relative (other) commands.
    """
    commands = sequence[:, :NODE_COMMAND_WIDTH]
    coords_norm = sequence[:, NODE_COMMAND_WIDTH:]

    x1, y1, x2, y2 = box
    width = x2 - x1 if x2 > x1 else 1
    height = y2 - y1 if y2 > y1 else 1

    command_indices = torch.argmax(commands, dim=-1)
    m_index = list(NODE_GLYPH_COMMANDS.keys()).index("M")
    is_m_mask = (command_indices == m_index).unsqueeze(1)

    # Scale from [-1, 1] to [0, 1]
    coords_0_1 = (coords_norm + 1) / 2
    
    denormalized = coords_0_1.clone()

    # Scale all x-like and y-like coordinates
    if coords_norm.shape[1] > 0:
        denormalized[:, 0::2] *= width
        denormalized[:, 1::2] *= height

    # Additionally translate the absolute 'M' coordinates
    if torch.sum(is_m_mask) > 0:
        m_coords = denormalized[is_m_mask.squeeze(1)]
        m_coords[:, 0] += x1
        m_coords[:, 1] += y1
        denormalized[is_m_mask.squeeze(1)] = m_coords

    return torch.cat([commands, denormalized], dim=-1)