import glob
import itertools
from pathlib import Path
import random
import os

import numpy as np
import torch
import tqdm
from fontTools.ttLib import TTFont
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, IterableDataset
from more_itertools import random_product

from glyphogen.command_defs import COORDINATE_WIDTH, NODE_COMMAND_WIDTH
from glyphogen.glyph import Glyph
from glyphogen.hyperparameters import (
    ALPHABET,
    BASE_DIR,
    GEN_IMAGE_SIZE,
    LIMIT,
    NUM_GLYPHS,
    MAX_COMMANDS,
)
from glyphogen.rendering import get_style_image

font_files = []
BANNED = ["noto", "bitcount", "nabla", "jersey", "rubik", "winky", "bungee"]
for file in sorted(list(glob.glob(BASE_DIR + "/*/*.ttf"))):
    if any(ban in file.lower() for ban in BANNED):
        continue
    ttfont = TTFont(file)
    if "COLR" in ttfont:
        continue
    if ttfont["head"].unitsPerEm != 1000:
        continue
    font_files.append(file)
    if LIMIT > 0 and len(font_files) >= LIMIT:
        break
if not font_files:
    raise ValueError(f"No suitable font files found in {BASE_DIR}")


class VectorizerGlyphDataset(IterableDataset):
    def __init__(
        self,
        font_files,
        alphabet: list[str],
        is_train=True,
        augmentations=0,
        roll_augmentations=0,
    ):
        self.alphabet = alphabet
        self.font_files = font_files
        self.is_train = is_train
        self.cache = {}
        self.augmentations = [{}]
        self.roll_augmentations = roll_augmentations if is_train else 0
        random.seed(1234)
        augs = [(199, 12), (23, 49), (29, -99)]
        if is_train:
            self.augmentations += [
                {"XAUG": augs[i][0], "YAUG": augs[i][1]} for i in range(augmentations)
            ]
        new_augs = []
        for roll in range(0, self.roll_augmentations + 1):
            for aug in self.augmentations:
                new_aug = aug.copy()
                new_aug["ROLL"] = roll
                new_augs.append(new_aug)
        self.augmentations = new_augs
        self.true_length = None
        self.choices = None

    def __len__(self):
        if self.true_length is not None:
            return self.true_length
        # Guess
        print(
            f"Generating dataset with {len(self.font_files)} fonts, {len(self.alphabet)} chars, {len(self.augmentations)} augmentations"
        )
        return len(self.font_files) * len(self.alphabet) * len(self.augmentations)
        # print("Calculating length of dataset...")
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        #     for font_file_path in tqdm.tqdm(iter(self), desc="Loading dataset"):
        #         pass
        # return self.true_length

    def __iter__(self):
        if torch.utils.data.get_worker_info():
            worker_total_num = torch.utils.data.get_worker_info().num_workers
            worker_id = torch.utils.data.get_worker_info().id
            # Split the font files between workers
            if worker_total_num > 1:
                font_files_per_worker = np.array_split(
                    self.font_files, worker_total_num
                )[worker_id]
                self.font_files = font_files_per_worker.tolist()
        else:
            worker_total_num = 1
            worker_id = 1

        self.choices = list(
            itertools.product(self.font_files, self.alphabet, self.augmentations)
        )
        self.true_length = 0
        random.shuffle(self.choices)

        for font_file_path, char, augment in self.choices:
            # print(font_file_path, char, augment)
            data = self._load_and_process_glyph(font_file_path, ord(char), augment)
            if data is not None:
                yield data
                self.true_length += 1

    def _load_and_process_glyph(self, font_file_path, char_ord, augment):
        try:
            roll = augment.get("ROLL", 0)
            if "ROLL" in augment:
                del augment["ROLL"]

            glyph = Glyph(Path(font_file_path), int(char_ord), location=augment)
            svg_glyph = glyph.vectorize()
            node_glyph = svg_glyph.to_node_glyph()

            if not node_glyph.commands or len(node_glyph.commands) <= 1:
                return None

            true_contour_count = len(node_glyph.contours)

            # Augmentation: Randomly roll the starting point of closed contours
            if roll:
                for contour in node_glyph.contours:
                    if contour.nodes:
                        contour.roll(roll)

            if len(node_glyph.commands) > MAX_COMMANDS:
                return None

            encoded_svg = node_glyph.encode()
            if encoded_svg is None or len(encoded_svg) <= 1:
                return None

            raster = glyph.rasterize(GEN_IMAGE_SIZE[0])
            raster = np.transpose(raster, (2, 0, 1))

            target_sequence = encoded_svg[:-1]
            ground_truth = encoded_svg[1:]
            true_command = ground_truth[:, :NODE_COMMAND_WIDTH]
            true_coord = ground_truth[:, NODE_COMMAND_WIDTH:].astype(np.float32)

            return (
                {
                    "raster_image": torch.from_numpy(raster).float(),
                    "target_sequence": torch.from_numpy(target_sequence).long(),
                    "contour_count": torch.tensor(true_contour_count).float(),
                },
                {
                    "command": torch.from_numpy(true_command).float(),
                    "coord": torch.from_numpy(true_coord).float(),
                },
            )
        except Exception as e:
            # print(f"Couldn't process glyph {char_ord} from {font_file_path}: {e}")
            return None


def collate_fn(batch):
    with torch.profiler.record_function("collate"):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None, None
        return torch.utils.data.dataloader.default_collate(batch)


def get_vectorizer_data(augmentations=20, roll_augmentations=2):
    font_files_train, font_files_test = train_test_split(
        font_files, test_size=0.2, random_state=42, shuffle=False
    )

    train_dataset = VectorizerGlyphDataset(
        font_files_train,
        ALPHABET,
        is_train=True,
        augmentations=augmentations,
        roll_augmentations=roll_augmentations,
    )
    test_dataset = VectorizerGlyphDataset(
        font_files_test,
        ALPHABET,
        is_train=False,
        augmentations=augmentations,
        roll_augmentations=0,
    )
    return train_dataset, test_dataset
