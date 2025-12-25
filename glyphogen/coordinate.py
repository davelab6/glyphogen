import torch
from .command_defs import MAX_COORDINATE
from .hyperparameters import GEN_IMAGE_SIZE


def to_image_space(self):
    """
    Transforms a tensor of points from font coordinate space to image space,
    matching the transformation used in rasterize_batch.
    """
    if self.shape[0] == 0:
        return self

    points = self / MAX_COORDINATE

    img_size = GEN_IMAGE_SIZE[0]
    baseline = 0.33 * img_size

    # Apply scaling and baseline translation
    # This seems to scale the glyph to fit roughly in the top 2/3 of the image
    transformed_points = points * img_size * 2.0 / 3.0
    # Add the baseline to every odd column (Y coordinate). There may be more than one pair.
    transformed_points[:, 1::2] += baseline
    # Apply vertical flip for image coordinates (Y-down)
    transformed_points[:, 1::2] = img_size - transformed_points[:, 1::2]

    return transformed_points


def get_bounds(self) -> "ImageSpaceBbox":
    """
    Computes the bounding box of the points in image space.
    Returns [x_min, y_min, x_max, y_max]
    """
    if self.shape[0] == 0:
        return ImageSpaceBbox([0.0, 0.0, 0.0, 0.0])

    x_min = torch.min(self[:, 0]).item()
    y_min = torch.min(self[:, 1]).item()
    x_max = torch.max(self[:, 0]).item()
    y_max = torch.max(self[:, 1]).item()

    return ImageSpaceBbox([x_min, y_min, x_max, y_max])


class ImageSpaceBbox(list):
    pass
