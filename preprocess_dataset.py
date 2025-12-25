import json
import os
from pathlib import Path
import random

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from glyphogen.dataset import font_files
from glyphogen.glyph import Glyph
from glyphogen.hyperparameters import GEN_IMAGE_SIZE, ALPHABET


def process_glyph_data(glyph_list, image_dir, start_img_id=0, start_ann_id=0):
    """
    Processes a list of glyphs to generate COCO-style dataset components.
    """
    images_json = []
    annotations_json = []

    img_id = start_img_id
    ann_id = start_ann_id

    for font_path, char in tqdm(glyph_list):
        try:
            glyph = Glyph(Path(font_path), ord(char), location={})

            # 1. Generate and save raster image
            raster_img = glyph.rasterize(GEN_IMAGE_SIZE[0])
            if np.sum(raster_img) == 0:  # Skip blank images
                continue

            # Save the image and store its info
            img_filename = f"{img_id}.png"
            img_path = image_dir / img_filename

            # Convert to PIL image to save
            from PIL import Image

            pil_img = Image.fromarray(
                (raster_img.squeeze(-1) * 255).astype(np.uint8), mode="L"
            )
            pil_img.save(img_path)

            images_json.append(
                {
                    "id": img_id,
                    "width": GEN_IMAGE_SIZE[1],
                    "height": GEN_IMAGE_SIZE[0],
                    "file_name": img_filename,
                }
            )

            # 2. Get segmentation data
            svg_glyph = glyph.vectorize()
            segmentation_data = svg_glyph.get_segmentation_data()

            if not segmentation_data:
                continue

            # 3. Create annotations for this image
            for seg_item in segmentation_data:
                bbox = seg_item["bbox"]  # [x_min, y_min, x_max, y_max]
                x, y, x2, y2 = bbox
                w = x2 - x
                h = y2 - y

                area = w * h

                # pycocotools expects segmentation in RLE format
                from pycocotools import mask as mask_util

                rle = mask_util.encode(np.asfortranarray(seg_item["mask"]))
                rle["counts"] = rle["counts"].decode("utf-8")

                annotations_json.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": seg_item["label"] + 1,  # 1 for outer, 2 for hole
                        "segmentation": rle,
                        "area": float(area),
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

            img_id += 1

        except Exception as e:
            print(f"Could not process {char} from {font_path}: {e}")

    return images_json, annotations_json, img_id, ann_id


def main():
    # --- Configuration ---
    DATA_DIR = Path("data")
    TRAIN_IMG_DIR = DATA_DIR / "images" / "train"
    TEST_IMG_DIR = DATA_DIR / "images" / "test"

    # Create directories
    TRAIN_IMG_DIR.mkdir(parents=True, exist_ok=True)
    TEST_IMG_DIR.mkdir(parents=True, exist_ok=True)

    # --- Data Preparation ---
    all_glyphs = []
    for font in font_files:
        for char in ALPHABET:
            all_glyphs.append((font, char))

    random.seed(42)
    random.shuffle(all_glyphs)

    # Split glyph list
    train_glyphs, test_glyphs = train_test_split(
        all_glyphs, test_size=0.2, random_state=42
    )

    print(f"Total glyph combinations: {len(all_glyphs)}")
    print(f"Training set size: {len(train_glyphs)}")
    print(f"Testing set size: {len(test_glyphs)}")

    # --- Categories Definition ---
    categories_json = [
        {"id": 1, "name": "outer", "supercategory": "contour"},
        {"id": 2, "name": "hole", "supercategory": "contour"},
    ]

    # --- Process Training Data ---
    print("\nProcessing training data...")
    train_images, train_annotations, _, _ = process_glyph_data(
        train_glyphs, TRAIN_IMG_DIR
    )

    train_coco_json = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": categories_json,
    }

    # --- Process Test Data ---
    print("\nProcessing test data...")
    test_images, test_annotations, _, _ = process_glyph_data(test_glyphs, TEST_IMG_DIR)

    test_coco_json = {
        "images": test_images,
        "annotations": test_annotations,
        "categories": categories_json,
    }

    # --- Save JSON files ---
    print("\nSaving COCO JSON files...")
    with open(DATA_DIR / "train.json", "w") as f:
        json.dump(train_coco_json, f)

    with open(DATA_DIR / "test.json", "w") as f:
        json.dump(test_coco_json, f)

    print("\nDone.")
    print(
        f"Training set: {len(train_images)} images, {len(train_annotations)} annotations."
    )
    print(f"Test set: {len(test_images)} images, {len(test_annotations)} annotations.")


if __name__ == "__main__":
    main()
