import torch
import numpy as np
import pytest
from torch.utils.data import DataLoader

from glyphogen_torch.glyph import NodeGlyph
from glyphogen_torch.command_defs import NODE_GLYPH_COMMANDS, NODE_COMMAND_WIDTH
from glyphogen_torch.dataset import GlyphDataset, collate_fn, font_files
from glyphogen_torch.hyperparameters import BATCH_SIZE, ALPHABET

# Get the index for the commands from the grammar
command_keys = list(NODE_GLYPH_COMMANDS.keys())
SOS_INDEX = command_keys.index("SOS")
Z_INDEX = command_keys.index("Z")
EOS_INDEX = command_keys.index("EOS")
NODE_COMMANDS = ["N", "NCI", "NCO", "L", "NH", "NV"]
NODE_COMMAND_INDICES = [command_keys.index(cmd) for cmd in NODE_COMMANDS]


@pytest.fixture(scope="module")
def dataset():
    # It's important to load the dataset once and reuse it for all tests
    test_dataset = GlyphDataset(font_files, ALPHABET, is_train=False)
    return DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)


def test_data_integrity(dataset):
    for ix, batch in enumerate(dataset):
        inputs, outputs = batch
        print("Testing batch %i" % ix)
        if ix == 50:
            break

        target_sequences = inputs[2]
        true_commands = outputs["command"]

        for i in range(target_sequences.shape[0]):
            # (a) The target_sequences always start with a SOS operator, followed by a node command
            first_command_one_hot = target_sequences[i, 0, :NODE_COMMAND_WIDTH]
            assert (
                torch.argmax(first_command_one_hot) == SOS_INDEX
            ), f"Sequence {i} does not start with SOS"
            second_command_one_hot = target_sequences[i, 1, :NODE_COMMAND_WIDTH]
            assert (
                torch.argmax(second_command_one_hot) in NODE_COMMAND_INDICES
            ), f"Sequence {i} does not have a node command after SOS"

            # Find the end of the sequence by looking for the EOS token
            true_command_indices = torch.argmax(true_commands[i], axis=-1)
            eos_indices = (true_command_indices == EOS_INDEX).nonzero()

            assert len(eos_indices) > 0, f"Sequence {i} does not have an EOS token"
            end_of_sequence_index = eos_indices[0]

            # (b) the true_commands may end with a Z and then an EOS token.
            if end_of_sequence_index > 0:
                command_before_eos_index = end_of_sequence_index - 1
                command_before_eos_one_hot = true_commands[i, command_before_eos_index]
                # It could be a Z or a node command
                assert (
                    torch.argmax(command_before_eos_one_hot) == Z_INDEX
                    or torch.argmax(command_before_eos_one_hot) in NODE_COMMAND_INDICES
                ), f"Sequence {i} does not have Z or node before EOS"


def test_svg_generation(dataset):
    for batch in dataset:
        inputs, outputs = batch
        if inputs is None:
            continue
        true_commands = outputs["command"]
        true_coords = outputs["coord"]
        # Just test the first one that comes along
        break

    svgs = []
    for i in range(true_commands.shape[0]):
        command_tensor = true_commands[i]
        coord_tensor = true_coords[i]

        decoded_glyph = NodeGlyph.from_numpy(
            command_tensor.numpy(), coord_tensor.numpy()
        )
        svg_string = decoded_glyph.to_svg_glyph().to_svg_string()
        svgs.append(svg_string)
    # Can't guarantee which one we will get, so let's not test for a specific letter
    assert len(svgs) > 0
