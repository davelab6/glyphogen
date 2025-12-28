import torch
import numpy as np
import pytest
from torch.utils.data import DataLoader

from glyphogen.glyph import NodeGlyph
from glyphogen.command_defs import NODE_GLYPH_COMMANDS, NODE_COMMAND_WIDTH
from glyphogen.dataset import get_hierarchical_data, collate_fn, font_files
from glyphogen.hyperparameters import ALPHABET

# Get the index for the commands from the grammar
command_keys = list(NODE_GLYPH_COMMANDS.keys())
SOS_INDEX = command_keys.index("SOS")
EOS_INDEX = command_keys.index("EOS")
N_INDEX = command_keys.index("N")

# Set our own batch size
BATCH_SIZE = 16


@pytest.fixture(scope="module")
def dataset():
    # It's important to load the dataset once and reuse it for all tests
    test_dataset, train_dataset = get_hierarchical_data()
    return DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)


@pytest.mark.skip(
    reason="This test is for the old non-hierarchical dataset format, needs updating for hierarchical"
)
def test_data_integrity(dataset):
    for ix, batch in enumerate(dataset):
        inputs, outputs = batch
        if not inputs or not outputs:
            continue

        print("Testing batch %i" % ix)
        if ix == 50:
            break

        target_sequences = inputs["target_sequence"]
        true_commands = outputs["command"]

        for i in range(target_sequences.shape[0]):
            # (a) The target_sequences always start with a SOS operator, followed by a node command
            first_command_one_hot = target_sequences[i, 0, :NODE_COMMAND_WIDTH]
            assert (
                torch.argmax(first_command_one_hot) == SOS_INDEX
            ), f"Sequence {i} does not start with SOS"
            second_command_one_hot = target_sequences[i, 1, :NODE_COMMAND_WIDTH]
            assert (
                torch.argmax(second_command_one_hot) == N_INDEX
            ), f"Sequence {i} does not have a node command after SOS"

            # Find the end of the sequence by looking for the EOS token
            true_command_indices = torch.argmax(true_commands[i], axis=-1)
            eos_indices = (true_command_indices == EOS_INDEX).nonzero()

            assert len(eos_indices) > 0, f"Sequence {i} does not have an EOS token"
            end_of_sequence_index = eos_indices[0]

            # (b) the true_commands must end with a node and then an EOS token.
            if end_of_sequence_index > 0:
                command_before_eos_index = end_of_sequence_index - 1
                command_before_eos_one_hot = true_commands[i, command_before_eos_index]
                assert (
                    torch.argmax(command_before_eos_one_hot) in NODE_COMMAND_INDICES
                ), f"Sequence {i} does not have a node before EOS"


def test_svg_generation(dataset):
    for batch in dataset:
        inputs, outputs = batch
        if inputs is None or outputs is None:
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
