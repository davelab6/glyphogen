from typing import NamedTuple, List, TypedDict
import torch
from jaxtyping import Float, UInt2


class ModelResults(NamedTuple):
    """NamedTuple for model results."""

    pred_commands: List[Float[torch.Tensor, "seq_len command_dim"]]
    pred_coords_img_space: List[Float[torch.Tensor, "contour point_dim"]]
    pred_coords_norm: List[Float[torch.Tensor, "contour point_dim"]]
    used_teacher_forcing: bool
    contour_boxes: List[
        Float[torch.Tensor, "4"]
    ]  # Added field for contour bounding boxes

    @classmethod
    def empty(cls) -> "ModelResults":
        """Create an empty ModelResults instance."""
        return cls(
            pred_commands=[],
            pred_coords_img_space=[],
            pred_coords_norm=[],
            used_teacher_forcing=False,
            contour_boxes=[],
        )


class SegmenterOutput(TypedDict):
    """TypedDict for segmenter outputs."""

    boxes: Float[torch.Tensor, "num_boxes 4"]
    masks: Float[torch.Tensor, "num_boxes height width"]


class LossDictionary(TypedDict):
    """TypedDict for loss values."""

    total_loss: Float[torch.Tensor, ""]
    command_loss: Float[torch.Tensor, ""]
    coord_loss: Float[torch.Tensor, ""]
    signed_area_loss: Float[torch.Tensor, ""]
    command_accuracy_metric: Float[torch.Tensor, ""]
    coordinate_mae_metric: Float[torch.Tensor, ""]


class GroundTruthContour(TypedDict):
    box: Float[torch.Tensor, "4"]
    label: UInt2[torch.Tensor, ""]
    mask: UInt2[torch.Tensor, "height width"]
    sequence: Float[
        torch.Tensor, "seq_len 17"
    ]  # Assuming 17 command dimensions, relative NodeCommand encoding


class Target(TypedDict):
    image_id: int
    gt_contours: List[GroundTruthContour]
