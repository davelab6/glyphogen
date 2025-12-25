from PIL import Image
import torch
from torchvision.datasets import CocoDetection
import glob
from fontTools.ttLib import TTFont
import torchvision.transforms.v2 as T

from glyphogen.hyperparameters import BASE_DIR

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
    if ttfont["head"].unitsPerEm != 1000:
        continue
    font_files.append(file)
if not font_files:
    raise ValueError(f"No suitable font files found in {BASE_DIR}")


class GlyphCocoDataset(CocoDetection):
    """
    A map-style dataset for the hierarchical vectorization model.
    It loads data from a COCO-style JSON file and provides ground truth
    for each contour (box, mask, and vector sequence).
    """

    def __init__(self, root, annFile, transforms=None):
        super().__init__(root, annFile)
        self.transforms = transforms

    def __getitem__(self, index):
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
                    }
                )

        # Sort targets by bounding box position (top-to-bottom, left-to-right)
        # This ensures a canonical order that matches our model's normalization.
        targets.sort(key=lambda t: (t["box"][1], t["box"][0]))

        if self.transforms is not None:
            # Note: transforms will need to handle a list of targets
            img, targets = self.transforms(img, targets)

        # The training loop's `step` function expects a tuple of (image, target_dict)
        return img, {"image_id": img_id, "gt_contours": targets}


def collate_fn(batch):
    """
    Standard collate_fn for object detection tasks.
    It does not try to stack the targets, but returns them as a list of dicts.
    """
    return tuple(zip(*batch))


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
