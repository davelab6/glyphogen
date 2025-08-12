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