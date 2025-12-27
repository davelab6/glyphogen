import json
from pathlib import Path
import random
from glyphogen.coordinate import (
    to_image_space,
)
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch

from glyphogen.glyph import Glyph
from glyphogen.hyperparameters import GEN_IMAGE_SIZE, ALPHABET
from glyphogen.dataset import font_files
from glyphogen.command_defs import NodeCommand, MAX_COORDINATE, NODE_COMMAND_WIDTH


def process_glyph_data(glyph_list, image_dir, start_img_id=0, start_ann_id=0):
    images_json = []
    annotations_json = []

    img_id = start_img_id
    ann_id = start_ann_id

    for font_path, char in tqdm(glyph_list, desc="Processing glyphs"):
        try:
            glyph = Glyph(Path(font_path), ord(char), location={})

            # Generate vector data first to ensure it's valid
            node_glyph = glyph.vectorize().to_node_glyph()
            node_glyph.normalize()  # IMPORTANT: ensure canonical order
            contour_sequences = node_glyph.encode()
            if contour_sequences is None:
                continue

            # Now get segmentation data, which should be in the same order
            svg_glyph = node_glyph.to_svg_glyph()
            segmentation_data = svg_glyph.get_segmentation_data()

            if not segmentation_data or len(segmentation_data) != len(
                contour_sequences
            ):
                # Mismatch between number of contours in segmentation and vectorization
                continue

            # Generate and save raster image
            raster_img = glyph.rasterize(GEN_IMAGE_SIZE[0])
            if np.sum(raster_img) < 0.01:  # Skip blank images
                print(f"Skipping blank image for {char} in {font_path}")
                continue

            img_filename = f"{img_id}.png"
            img_path = image_dir / img_filename
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

            # Create annotations for this image
            for i, seg_item in enumerate(segmentation_data):
                bbox = seg_item["bbox"]
                x, y, x2, y2 = bbox
                w = x2 - x
                h = y2 - y

                from pycocotools import mask as mask_util

                rle = mask_util.encode(np.asfortranarray(seg_item["mask"]))
                rle["counts"] = rle["counts"].decode("utf-8")

                # Transform sequence coordinates from font units to image space
                encoded_contour = torch.from_numpy(contour_sequences[i]).float()
                commands = encoded_contour[:, :NODE_COMMAND_WIDTH]
                coords_font_space = encoded_contour[:, NODE_COMMAND_WIDTH:]
                # Normalize and transform
                coords_img_space = to_image_space(coords_font_space)

                # Reshape back and recombine
                sequence_img_space = torch.cat([commands, coords_img_space], dim=-1)
                sequence_as_list = sequence_img_space.tolist()

                annotations_json.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": seg_item["label"] + 1,
                        "segmentation": rle,
                        "area": float(w * h),
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "iscrowd": 0,
                        "sequence": sequence_as_list,  # Add the sequence data
                    }
                )
                ann_id += 1

            img_id += 1

        except Exception as e:
            # print(f"Could not process {char} from {font_path}: {e}")
            pass

    return images_json, annotations_json


def main():
    DATA_DIR = Path("data")
    TRAIN_IMG_DIR = DATA_DIR / "images_hierarchical" / "train"
    TEST_IMG_DIR = DATA_DIR / "images_hierarchical" / "test"

    TRAIN_IMG_DIR.mkdir(parents=True, exist_ok=True)
    TEST_IMG_DIR.mkdir(parents=True, exist_ok=True)

    all_glyphs = []
    for font in font_files:
        for char in ALPHABET:
            all_glyphs.append((font, char))

    random.seed(42)
    random.shuffle(all_glyphs)

    train_glyphs, test_glyphs = train_test_split(
        all_glyphs, test_size=0.2, random_state=42
    )

    print(
        f"Processing {len(train_glyphs)} training glyphs and {len(test_glyphs)} test glyphs."
    )

    categories_json = [
        {"id": 1, "name": "outer", "supercategory": "contour"},
        {"id": 2, "name": "hole", "supercategory": "contour"},
    ]

    train_images, train_annotations = process_glyph_data(
        train_glyphs, TRAIN_IMG_DIR, start_img_id=0, start_ann_id=0
    )
    train_coco_json = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": categories_json,
    }

    test_images, test_annotations = process_glyph_data(
        test_glyphs,
        TEST_IMG_DIR,
        start_img_id=len(train_images),
        start_ann_id=len(train_annotations),
    )
    test_coco_json = {
        "images": test_images,
        "annotations": test_annotations,
        "categories": categories_json,
    }

    print("\nSaving COCO JSON files...")
    with open(DATA_DIR / "train_hierarchical.json", "w") as f:
        json.dump(train_coco_json, f)

    with open(DATA_DIR / "test_hierarchical.json", "w") as f:
        json.dump(test_coco_json, f)

    print("\nDone.")


if __name__ == "__main__":
    main()
