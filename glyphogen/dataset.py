import glob

import torch
import torchvision.transforms.v2 as T
from fontTools.ttLib import TTFont
from torchvision.datasets import CocoDetection

from glyphogen.representations.model import MODEL_REPRESENTATION
from glyphogen.hyperparameters import BASE_DIR
from glyphogen.typing import CollatedGlyphData, GroundTruthContour, Target
from typing import List
import torch.nn.functional as F

# This file is now much simpler as it's only used by the hierarchical model.
# The original dataset logic is preserved in the history.

font_files = []
BANNED = ["noto", "bitcount", "nabla", "jersey", "rubik", "winky", "bungee", "adobe"]
for file in sorted(list(glob.glob(BASE_DIR + "/*/*.ttf"))):
    if any(ban in file.lower() for ban in BANNED):
        continue
    ttfont = TTFont(file)
    if "COLR" in ttfont:
        continue
    if ttfont["head"].unitsPerEm != 1000:  # type: ignore
        continue
    font_files.append(file)
if not font_files:
    raise ValueError(f"No suitable font files found in {BASE_DIR}")


def filter_out(targets):
    # CURVE_COMMANDS = [
    #     x for x in MODEL_REPRESENTATION.grammar.keys() if x.startswith("N")
    # ]
    # filtered_indices = [
    #     MODEL_REPRESENTATION.encode_command_one_hot(cmd) for cmd in CURVE_COMMANDS
    # ]
    # # Sum and convert to bool
    # curve_mask = torch.sum(torch.stack(filtered_indices, dim=0), dim=0).bool()
    # for sequences in targets:
    #     (commands, _) = MODEL_REPRESENTATION.split_tensor(sequences["sequence"])
    #     for command in commands:
    #         if torch.any(command.bool() & curve_mask):
    #             return False
    return True


class GlyphCocoDataset(CocoDetection):
    """
    A map-style dataset for the hierarchical vectorization model.
    It loads data from a COCO-style JSON file and provides ground truth
    for each contour (box, mask, and vector sequence).
    """

    def __init__(self, root, annFile, transforms=None):
        super().__init__(root, annFile)
        self.transforms = transforms

    def __getitem__(
        self, index
    ):  # -> DatasetItem, although the type checker hates it because you can't override __getitem__
        img, target_anns = super().__getitem__(index)
        img_id = self.ids[index]

        # For the hierarchical model, we want a list of targets, one per contour.
        targets = []
        if not target_anns:
            # Return the image and an empty list if there are no annotations
            pass
        else:
            for ann in target_anns:
                box = torch.as_tensor(ann["bbox"], dtype=torch.float32)
                # convert from [x, y, w, h] to [x1, y1, x2, y2]
                box[2:] += box[:2]

                # The sequence was saved as a list of lists, convert back to tensor
                sequence = torch.tensor(ann["sequence"], dtype=torch.float32)

                targets.append(
                    {
                        "box": box,
                        "label": torch.tensor(ann["category_id"], dtype=torch.int64),
                        "mask": torch.as_tensor(
                            self.coco.annToMask(ann), dtype=torch.uint8
                        ),
                        "sequence": sequence,
                        "x_aligned_point_indices": ann.get(
                            "x_aligned_point_indices", []
                        ),
                        "y_aligned_point_indices": ann.get(
                            "y_aligned_point_indices", []
                        ),
                    }
                )

        # Sort targets by bounding box position (top-to-bottom, left-to-right)
        # This ensures a canonical order that matches our model's normalization.
        targets.sort(key=lambda t: (t["box"][1], t["box"][0]))

        if self.transforms is not None:
            # Note: transforms will need to handle a list of targets
            img, targets = self.transforms(img, targets)

        keep = filter_out(targets)
        if not keep:
            return None
        return img, {"image_id": img_id, "gt_contours": targets}


def collate_fn(batch):
    """
    Custom collate_fn for hierarchical glyph data.

    This function takes a batch of samples (each being an (image, target) tuple)
    and collates them into a single `CollatedGlyphData` dictionary.

    It unnests the contours from each image's target dictionary and flattens
    them into batch-wide tensors. This pre-processing is moved here from the
    model's forward pass to make the model more `torch.compile` friendly by
    avoiding data-dependent control flow (variable-length loops) in the model.
    """
    # Filter out None items, which can happen if an image fails to load.
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    images, gt_targets = zip(*batch)

    all_normalized_masks = []
    all_contour_boxes = []
    all_target_sequences = []
    all_contour_image_idx = []
    all_x_alignments = []
    all_y_alignments = []
    all_labels = []

    for i, target in enumerate(gt_targets):
        img_height, img_width = images[i].shape[1], images[i].shape[2]
        gt_contours: List[GroundTruthContour] = target["gt_contours"]

        for contour in gt_contours:
            box = contour["box"].clamp(min=0, max=max(img_height, img_width) - 1)
            mask = contour["mask"]
            label = contour["label"]
            sequence = contour["sequence"]

            x1, y1, x2, y2 = box.long()
            if x1 >= x2 or y1 >= y2:
                continue

            # Crop and normalize the mask
            cropped_mask = mask[y1:y2, x1:x2].unsqueeze(0).unsqueeze(0)
            normalized_mask = F.interpolate(
                cropped_mask.to(torch.float32),
                size=(512, 512),
                mode="bilinear",
                align_corners=False,
            )

            all_normalized_masks.append(normalized_mask)
            all_contour_boxes.append(box)
            all_target_sequences.append(sequence)
            all_contour_image_idx.append(i)
            all_labels.append(label)
            all_x_alignments.append(contour["x_aligned_point_indices"])
            all_y_alignments.append(contour["y_aligned_point_indices"])

    # If a batch has no valid contours, return None. The training loop must handle this.
    if not all_normalized_masks:
        return None

    collated_data: CollatedGlyphData = {
        "images": torch.stack(images, 0),
        "gt_targets": list(gt_targets),
        "normalized_masks": torch.cat(all_normalized_masks, dim=0),
        "contour_boxes": torch.stack(all_contour_boxes, dim=0),
        "labels": torch.stack(all_labels, dim=0),
        "target_sequences": all_target_sequences,
        "contour_image_idx": torch.tensor(all_contour_image_idx, dtype=torch.long),
        "x_aligned_point_indices": all_x_alignments,
        "y_aligned_point_indices": all_y_alignments,
    }
    return collated_data


def get_transform(train):
    """
    Defines the transformations to be applied to the dataset.
    """
    transforms = []
    # The ToTensor transform is now applied inside the dataset __getitem__
    if train:
        # We can add more augmentations here later
        # transforms.append(T.RandomHorizontalFlip(0.5))
        pass
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


def get_hierarchical_data():
    """
    Creates and returns DataLoaders for the hierarchical training task.
    """
    from pathlib import Path

    DATA_DIR = Path("data")
    TRAIN_IMG_DIR = DATA_DIR / "images_hierarchical" / "train"
    TEST_IMG_DIR = DATA_DIR / "images_hierarchical" / "test"
    TRAIN_JSON = DATA_DIR / "train_hierarchical.json"
    TEST_JSON = DATA_DIR / "test_hierarchical.json"

    train_dataset = GlyphCocoDataset(
        root=TRAIN_IMG_DIR, annFile=TRAIN_JSON, transforms=get_transform(train=True)
    )
    test_dataset = GlyphCocoDataset(
        root=TEST_IMG_DIR, annFile=TEST_JSON, transforms=get_transform(train=False)
    )

    return train_dataset, test_dataset
