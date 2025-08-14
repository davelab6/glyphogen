import tensorflow as tf
import numpy as np
import pytest

from deepvecfont3.glyph import NodeGlyph, NODE_GLYPH_COMMANDS, NODE_COMMAND_WIDTH
from deepvecfont3.hyperparameters import BATCH_SIZE

# Get the index for the commands from the grammar
command_keys = list(NODE_GLYPH_COMMANDS.keys())
SOS_INDEX = command_keys.index("SOS")
Z_INDEX = command_keys.index("Z")
EOS_INDEX = command_keys.index("EOS")
NODE_COMMANDS = ["N", "NCI", "NCO", "L"]
NODE_COMMAND_INDICES = [command_keys.index(cmd) for cmd in NODE_COMMANDS]

@pytest.fixture(scope="module")
def dataset():
    # It's important to load the dataset once and reuse it for all tests
    # to avoid performance issues.
    return tf.data.Dataset.load("test.tfds")


def test_data_integrity(dataset):
    for batch in dataset:
        inputs, outputs = batch

        target_sequences = inputs[2]
        true_commands = outputs["command"]
        true_coords = outputs["coord"]

        for i in range(target_sequences.shape[0]):
            # (a) The target_sequences always start with a SOS operator, followed by a node command
            first_command_one_hot = target_sequences[i, 0, :NODE_COMMAND_WIDTH]
            assert np.argmax(first_command_one_hot) == SOS_INDEX, f"Sequence {i} does not start with SOS"
            second_command_one_hot = target_sequences[i, 1, :NODE_COMMAND_WIDTH]
            assert np.argmax(second_command_one_hot) in NODE_COMMAND_INDICES, f"Sequence {i} does not have a node command after SOS"

            # Find the end of the sequence by looking for the EOS token
            true_command_indices = np.argmax(true_commands[i], axis=-1)
            eos_indices = np.where(true_command_indices == EOS_INDEX)[0]

            assert len(eos_indices) > 0, f"Sequence {i} does not have an EOS token"
            end_of_sequence_index = eos_indices[0]

            # (b) the true_commands may end with a Z and then an EOS token.
            if end_of_sequence_index > 0:
                command_before_eos_index = end_of_sequence_index - 1
                command_before_eos_one_hot = true_commands[i, command_before_eos_index]
                # It could be a Z or a node command
                assert np.argmax(command_before_eos_one_hot) == Z_INDEX or np.argmax(command_before_eos_one_hot) in NODE_COMMAND_INDICES, f"Sequence {i} does not have Z or node before EOS"

def test_svg_generation(dataset):
    batch = next(iter(dataset))
    inputs, outputs = batch
    true_commands = outputs["command"]
    true_coords = outputs["coord"]

    svgs = []
    for i in range(3):
        command_tensor = true_commands[i]
        coord_tensor = true_coords[i]

        decoded_glyph = NodeGlyph.from_numpy(
            command_tensor.numpy(), coord_tensor.numpy()
        )
        svg_string = decoded_glyph.to_svg_glyph().to_svg_string()
        svgs.append(svg_string)
    assert svgs[1] == "M 165 360 C 165 95 175 50 175 45 C 175 35 220 25 240 25 L 235 -5 C 205 -5 160 0 130 0 C 95 0 50 -5 15 -5 L 20 25 C 40 25 80 35 80 45 C 80 45 90 110 90 365 C 90 625 80 665 80 665 L 15 685 L 15 710 C 50 710 110 720 145 730 L 165 720 C 165 720 165 600 165 400 C 240 445 465 520 465 365 C 465 80 475 45 475 45 C 475 35 520 25 540 25 L 535 -5 C 505 -5 465 0 430 0 C 395 0 350 -5 315 -5 L 320 25 C 340 25 380 35 385 45 C 380 45 390 80 390 225 C 390 380 360 395 315 395 Z" # Letter h
