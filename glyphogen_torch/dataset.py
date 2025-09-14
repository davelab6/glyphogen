import glob
from pathlib import Path
import random

import numpy as np
import torch
import tqdm
from fontTools.ttLib import TTFont
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, IterableDataset

from glyphogen_torch.command_defs import COORDINATE_WIDTH, NODE_COMMAND_WIDTH
from glyphogen_torch.glyph import Glyph
from glyphogen_torch.hyperparameters import (
    ALPHABET,
    BASE_DIR,
    GEN_IMAGE_SIZE,
    LIMIT,
    NUM_GLYPHS,
)
from glyphogen_torch.rendering import get_style_image

font_files = []
BANNED = ["noto", "bitcount", "nabla", "jersey", "rubik", "winky", "bungee"]
for file in list(glob.glob(BASE_DIR + "/*/*.ttf")):
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
                (
                    torch.from_numpy(style_image).float(),
                    glyph_id_one_hot,
                    torch.from_numpy(target_sequence).long(),
                ),
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
    def __init__(self, font_files, alphabet, is_train=True):
        self.alphabet = alphabet
        self.font_files_train, self.font_files_test = train_test_split(
            font_files, test_size=0.2, random_state=42
        )
        self.font_files = self.font_files_train if is_train else self.font_files_test
        self.cache = {}

    def __len__(self):
        return len(self.font_files) * len(self.alphabet)

    def __iter__(self):
        random.shuffle(self.font_files)
        for font_file_path in self.font_files:
            random.shuffle(self.alphabet)
            for char in self.alphabet:
                data = self._load_and_process_glyph(font_file_path, ord(char))
                if data is not None:
                    yield data

    def _load_and_process_glyph(self, font_file_path, char_ord):
        try:
            glyph = Glyph(Path(font_file_path), int(char_ord), location={})
            raster = glyph.rasterize(GEN_IMAGE_SIZE[0])
            if (font_file_path, char_ord) in self.cache:
                encoded_svg = self.cache[(font_file_path, char_ord)]
            else:
                svg_glyph = glyph.vectorize()
                node_glyph = svg_glyph.to_node_glyph()

                if not node_glyph.commands or len(node_glyph.commands) <= 1:
                    encoded_svg = None
                else:
                    encoded_svg = node_glyph.encode()
                self.cache[(font_file_path, char_ord)] = encoded_svg
            if encoded_svg is None or len(encoded_svg) <= 1:
                return None

            target_sequence = encoded_svg[:-1]
            ground_truth = encoded_svg[1:]
            true_command = ground_truth[:, :NODE_COMMAND_WIDTH]
            true_coord = ground_truth[:, NODE_COMMAND_WIDTH:].astype(np.float32)

            raster = np.transpose(raster, (2, 0, 1))

            return (
                (
                    torch.from_numpy(raster).float(),
                    torch.from_numpy(target_sequence).long(),
                ),
                (
                    torch.from_numpy(true_command).float(),
                    torch.from_numpy(true_coord).float(),
                ),
            )
        except Exception as e:
            # print(f"Couldn't process glyph {char_ord} from {font_file_path}: {e}")
            return None


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None
    return torch.utils.data.dataloader.default_collate(batch)


def get_full_model_data():
    train_dataset = GlyphDataset(font_files, ALPHABET, is_train=True)
    test_dataset = GlyphDataset(font_files, ALPHABET, is_train=False)
    return train_dataset, test_dataset


def get_pretrain_data():
    train_dataset = PretrainGlyphDataset(font_files, ALPHABET, is_train=True)
    test_dataset = PretrainGlyphDataset(font_files, ALPHABET, is_train=False)
    return train_dataset, test_dataset
