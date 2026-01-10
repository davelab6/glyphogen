import torch
import numpy as np
import pytest
from torch.utils.data import DataLoader

from glyphogen.nodeglyph import NodeGlyph
from glyphogen.representations.nodecommand import NodeCommand
from glyphogen.dataset import get_hierarchical_data, collate_fn, font_files
from glyphogen.hyperparameters import ALPHABET
from glyphogen.svgglyph import SVGGlyph

# Get the index for the commands from the grammar
SOS_INDEX = NodeCommand.encode_command("SOS")
EOS_INDEX = NodeCommand.encode_command("EOS")
N_INDEX = NodeCommand.encode_command("N")

# Set our own batch size
BATCH_SIZE = 16


@pytest.fixture(scope="module")
def dataset():
    # It's important to load the dataset once and reuse it for all tests
    test_dataset, train_dataset = get_hierarchical_data()
    return DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)


def test_svg_generation(dataset):
    """
    Tests that a batch from the dataset can be successfully decoded and
    converted into an SVG.
    """
    # Get one batch from the dataset
    batch = next(iter(dataset))
    if batch is None:
        pytest.skip("Could not get a batch from the dataset.")

    # Get the ground truth data for the first glyph in the batch
    first_glyph_targets = batch["gt_targets"][0]
    gt_contours = first_glyph_targets["gt_contours"]

    # We need the raw command sequences from each contour
    contour_sequences = [contour["sequence"] for contour in gt_contours]

    # Decode the sequences into a NodeGlyph object
    decoded_glyph = NodeGlyph.decode(contour_sequences, NodeCommand)

    # Generate an SVG from the NodeGlyph
    svg_string = SVGGlyph.from_node_glyph(decoded_glyph).to_svg_string()

    # The test just asserts that an SVG was produced without errors
    assert isinstance(svg_string, str)
    assert len(svg_string) > 0
