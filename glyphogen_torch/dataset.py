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

from glyphogen_torch.command_defs import COORDINATE_WIDTH, NODE_COMMAND_WIDTH
from glyphogen_torch.glyph import Glyph
from glyphogen_torch.hyperparameters import (
    ALPHABET,
    BASE_DIR,
    GEN_IMAGE_SIZE,
    LIMIT,
    NUM_GLYPHS,
    MAX_COMMANDS,
)
from glyphogen_torch.rendering import get_style_image

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


class GlyphDataset(Dataset):
    def __init__(self, font_files, alphabet, is_train=True):
        self.alphabet = alphabet
        self.font_files_train, self.font_files_test = train_test_split(
            font_files, test_size=0.2, random_state=42
        )
        self.font_files = self.font_files_train if is_train else self.font_files_test
        self.data = []
        for font_file_path in tqdm.tqdm(self.font_files, desc="Loading dataset"):
            for i, char in enumerate(self.alphabet):
                self.data.append((font_file_path, i, char))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        font_file_path, glyph_index, char = self.data[idx]
        font_file = Path(font_file_path)

        try:
            style_image = get_style_image(font_file, variation={})
            glyph = Glyph(font_file, ord(char), location={})
            raster = glyph.rasterize(GEN_IMAGE_SIZE[0])
            svg_glyph = glyph.vectorize()
            node_glyph = svg_glyph.to_node_glyph()
            if not node_glyph.commands:
                print(f"No commands for {font_file_path} for char {char}")
                return None
            if len(node_glyph.commands) > MAX_COMMANDS:
                print(
                    f"Too many commands ({len(node_glyph.commands)}) for {font_file_path} for char {char}"
                )
                return None
            encoded_svg = node_glyph.encode()
            if encoded_svg is None:
                print(f"Failed to encode SVG for {font_file_path} for char {char}")
                return None

            glyph_id_one_hot = torch.nn.functional.one_hot(
                torch.tensor(glyph_index), NUM_GLYPHS
            ).float()

            # Ensure correct channel dimension for PyTorch
            style_image = np.transpose(style_image, (2, 0, 1))
            raster = np.transpose(raster, (2, 0, 1))

            target_sequence = encoded_svg[:-1]
            ground_truth = encoded_svg[1:]
            true_command = ground_truth[:, :NODE_COMMAND_WIDTH]
            true_coord = ground_truth[:, NODE_COMMAND_WIDTH:]

            return (
                {
                    "style_image": torch.from_numpy(style_image).float(),
                    "glyph_id": glyph_id_one_hot,
                    "target_sequence": torch.from_numpy(target_sequence).long(),
                },
                {
                    "raster": torch.from_numpy(raster).float(),
                    "command": torch.from_numpy(true_command).float(),
                    "coord": torch.from_numpy(true_coord).float(),
                },
            )
        except Exception as e:
            print(f"Error processing {font_file_path} for char {char}: {e}")
            return None


class PretrainGlyphDataset(IterableDataset):
    def __init__(
        self,
        font_files,
        alphabet: list[str],
        is_train=True,
        augmentations=0,
        roll_augmentations=0,
    ):
        self.alphabet = alphabet
        self.font_files_train, self.font_files_test = train_test_split(
            font_files, test_size=0.2, random_state=42
        )
        self.font_files = self.font_files_train if is_train else self.font_files_test
        self.is_train = is_train
        self.cache = {}
        self.augmentations = [{}]
        self.roll_augmentations = roll_augmentations if is_train else 0
        if is_train:
            self.augmentations += [
                {"XAUG": random.randint(0, 200), "YAUG": random.randint(-100, 100)}
                for _ in range(augmentations)
            ]
        new_augs = []
        for roll in range(0, self.roll_augmentations + 1):
            for aug in self.augmentations:
                new_aug = aug.copy()
                new_aug["ROLL"] = roll
                new_augs.append(new_aug)
        self.augmentations = new_augs
        self.true_length = None

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

        choices = list(
            itertools.product(self.font_files, self.alphabet, self.augmentations)
        )
        self.true_length = 0
        random.shuffle(choices)
        for font_file_path, char, augment in choices:
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
                (
                    torch.from_numpy(true_command).float(),
                    torch.from_numpy(true_coord).float(),
                ),
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


def get_full_model_data():
    train_dataset = GlyphDataset(font_files, ALPHABET, is_train=True)
    test_dataset = GlyphDataset(font_files, ALPHABET, is_train=False)
    return train_dataset, test_dataset


def get_pretrain_data(augmentations=20, roll_augmentations=2):
    train_dataset = PretrainGlyphDataset(
        font_files,
        ALPHABET,
        is_train=True,
        augmentations=augmentations,
        roll_augmentations=roll_augmentations,
    )
    test_dataset = PretrainGlyphDataset(
        font_files,
        ALPHABET,
        is_train=False,
        augmentations=augmentations,
        roll_augmentations=0,
    )
    return train_dataset, test_dataset
