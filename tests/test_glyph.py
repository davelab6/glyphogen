from pathlib import Path
import pytest
from glyphogen.glyph import (
    Glyph,
    SVGGlyph,
    NodeGlyph,
    NodeCommand,
    SVGCommand,
    NodeContour,
    Node,
)
from glyphogen.command_defs import (
    NODE_COMMAND_WIDTH,
)
from glyphogen.hyperparameters import GEN_IMAGE_SIZE
import numpy as np
import torch
from glyphogen.model import unroll_relative_coords
from glyphogen.command_defs import MAX_COORDINATE


def test_glyph_extraction():
    font_path = Path("tests/data/NotoSans[wdth,wght].ttf")
    location = {"wght": 400.0}
    codepoint = ord("a")

    glyph = Glyph(font_path, codepoint, location)

    # Rasterize the glyph
    rasterized_glyph = glyph.rasterize(GEN_IMAGE_SIZE[0])

    # Check the size of the rasterized numpy array
    assert rasterized_glyph.shape == (*GEN_IMAGE_SIZE, 1)

    # Vectorize the glyph
    vectorized_glyph = glyph.vectorize()

    assert isinstance(vectorized_glyph, SVGGlyph)
    assert len(vectorized_glyph.commands) > 0
    assert vectorized_glyph.commands[0].command == "M"
    node_glyph = vectorized_glyph.to_node_glyph()
    # Just checking if it runs for now
    assert isinstance(node_glyph, NodeGlyph)

    encoded_glyph = node_glyph.encode()
    if encoded_glyph is None:
        pytest.skip("Glyph 'a' in NotoSans is too complex, skipping remainder of test.")

    encoded_glyph_batch = np.expand_dims(encoded_glyph, axis=0)

    command_tensor = encoded_glyph_batch[:, :, :NODE_COMMAND_WIDTH]
    coord_tensor = encoded_glyph_batch[:, :, NODE_COMMAND_WIDTH:]
    command_tensor = torch.from_numpy(command_tensor).float()
    coord_tensor = torch.from_numpy(coord_tensor).float()
    # Unroll the relative coordinates to absolute coordinates
    coord_tensor_absolute = unroll_relative_coords(command_tensor, coord_tensor)
    import IPython

    IPython.embed()


def test_node_glyph_encoding():
    contour = NodeContour([])
    contour.push_command(NodeCommand("NCO", [50, 50, 28, 0]))
    contour.push_command(NodeCommand("NV", [100, 100, -28, +28]))
    contour.push_command(NodeCommand("NCI", [50, 150, 28, 0]))
    node_glyph = NodeGlyph([contour])

    encoded_glyph = node_glyph.encode()

    # Add a batch dimension and convert to tensor
    encoded_glyph_batch = np.expand_dims(encoded_glyph, axis=0)

    command_tensor = encoded_glyph_batch[:, :, :NODE_COMMAND_WIDTH]
    coord_tensor = encoded_glyph_batch[:, :, NODE_COMMAND_WIDTH:]

    # First command should be SOC
    assert np.argmax(command_tensor[0, 0]) == NodeCommand.encode_command("SOC")

    # Find the EOS token
    eos_index_in_encoded = np.where(
        np.argmax(command_tensor[0], axis=-1) == NodeCommand.encode_command("EOS")
    )[0]
    assert len(eos_index_in_encoded) > 0, "EOS token not found in encoded glyph"
    # The EOS token should be at the end of the actual sequence, before any padding
    # node_glyph.commands includes SOC, then the nodes. So the EOS should be at len(node_glyph.commands)
    assert eos_index_in_encoded[0] == len(node_glyph.commands)

    # There should be 3 nodes in between
    # We need to check the actual commands, not the padded ones.
    # The actual commands are from index 1 (after SOC) up to the EOS token.
    # len(node_glyph.commands) gives the number of commands *before* EOS, including SOC.
    # So the node commands are from index 1 to len(node_glyph.commands) - 1.
    # The EOS token is at index len(node_glyph.commands).

    # The slice for node commands is from index 1 up to the EOS token's index (exclusive).
    # The EOS token is at `eos_index_in_encoded[0]`.

    assert (
        np.sum(
            np.argmax(command_tensor[0, 1 : eos_index_in_encoded[0]], axis=-1)
            != NodeCommand.encode_command("L")
        )
        == 3
    )

    # Remove the batch dimension for from_numpy
    decoded_glyph = NodeGlyph.from_numpy(command_tensor[0], coord_tensor[0])

    assert len(decoded_glyph.contours) == 1
    # The decoded contour contains the nodes, not the SOC/Z commands
    assert len(decoded_glyph.contours[0].nodes) == 3
    # The .commands property generates commands from the nodes
    assert len(decoded_glyph.contours[0].commands) == 3
    assert decoded_glyph.contours[0].commands[0].command == "NCO"
    assert decoded_glyph.contours[0].commands[1].command == "NV"
    assert decoded_glyph.contours[0].commands[2].command == "NCI"


def test_svg_to_node_glyph_conversion():
    # M 10 20 L 30 40 C 50 60 70 80 90 100 Z
    svg_commands = [
        SVGCommand("M", [10, 20]),
        SVGCommand("L", [30, 40]),
        SVGCommand("C", [50, 60, 70, 80, 90, 100]),
        SVGCommand("Z", []),
    ]
    svg_glyph = SVGGlyph(svg_commands)
    node_glyph = svg_glyph.to_node_glyph()

    # Should be SOC + 3 nodes
    assert len(node_glyph.commands) == 4
    assert node_glyph.commands[0].command == "SOC"

    # Node 1 (from M and L)
    assert node_glyph.commands[1].command == "L"
    np.testing.assert_array_equal(node_glyph.commands[1].coordinates, [10, 20])

    # Node 2 (from L and C)
    assert node_glyph.commands[2].command == "NCO"  # Line in, Curve out
    # Coords: relative pos, relative handle delta
    # Pos: (30,40) -> rel to (10,20) is (20,20)
    # Handle: (50,60) -> rel to (30,40) is (20,20)
    np.testing.assert_array_equal(node_glyph.commands[2].coordinates, [20, 20, 20, 20])

    # Node 3 (from C and Z)
    assert node_glyph.commands[3].command == "NCI"  # Curve in, Line out
    # Coords: relative pos, relative handle delta
    # Pos: (90,100) -> rel to (30,40) is (60,60)
    # Handle: (70,80) -> rel to (90,100) is (-20,-20)
    np.testing.assert_array_equal(
        node_glyph.commands[3].coordinates, [60, 60, -20, -20]
    )


def test_roundtrip_conversion():
    # M 0 0 L 100 0 L 100 100 L 0 100 Z
    svg_commands = [
        SVGCommand("M", [0, 0]),
        SVGCommand("L", [100, 0]),
        SVGCommand("L", [100, 100]),
        SVGCommand("L", [0, 100]),
        SVGCommand("Z", []),
    ]
    svg_glyph = SVGGlyph(svg_commands)
    node_glyph = svg_glyph.to_node_glyph()

    # Should be SOC + 4 L nodes
    assert len(node_glyph.commands) == 5
    assert node_glyph.commands[0].command == "SOC"
    assert node_glyph.commands[1].command == "L"  # First node is absolute, so it's L
    assert node_glyph.commands[2].command == "LH"
    assert node_glyph.commands[3].command == "LV"
    assert node_glyph.commands[4].command == "LH"

    # Check coordinates (absolute for first, relative for rest)
    np.testing.assert_array_equal(
        node_glyph.commands[1].coordinates, [0, 0]
    )  # L command, absolute
    np.testing.assert_array_equal(
        node_glyph.commands[2].coordinates, [100, 0]
    )  # LH command, x-delta
    np.testing.assert_array_equal(
        node_glyph.commands[3].coordinates, [100, 0]
    )  # LV command, y-delta
    np.testing.assert_array_equal(
        node_glyph.commands[4].coordinates, [-100, 0]
    )  # LH command, x-delta

    reconverted_svg_glyph = node_glyph.to_svg_glyph()
    # The M is implicit in NodeGlyph, so the conversion adds it back.
    # The Z is also explicit.
    assert len(reconverted_svg_glyph.commands) == 5
    assert reconverted_svg_glyph.commands[0].command == "M"
    assert reconverted_svg_glyph.commands[0].coordinates == [0, 0]
    assert all(cmd.command == "L" for cmd in reconverted_svg_glyph.commands[1:4])
    assert reconverted_svg_glyph.commands[4].command == "Z"

    # Check coordinates
    assert reconverted_svg_glyph.commands[1].coordinates == [100, 0]
    assert reconverted_svg_glyph.commands[2].coordinates == [100, 100]
    assert reconverted_svg_glyph.commands[3].coordinates == [0, 100]

    assert reconverted_svg_glyph.to_svg_string() == "M 0 0 L 100 0 L 100 100 L 0 100 Z"


def test_coordinate_unrolling_roundtrip():
    # 1. Create a NodeGlyph with known absolute coordinates
    contour = NodeContour([])
    # Node 1
    contour.push_command(NodeCommand("NCO", [50, 50, 28, 0]))
    # Node 2
    contour.push_command(NodeCommand("NV", [100, 100, -28, 28]))
    # Node 3
    contour.push_command(NodeCommand("NCI", [50, 150, 28, 0]))
    node_glyph = NodeGlyph([contour])

    # 2. Get the encoded representation (relative, normalized coordinates)
    encoded_glyph = node_glyph.encode()
    assert encoded_glyph is not None

    # Add a batch dimension
    encoded_glyph_batch = np.expand_dims(encoded_glyph, axis=0)
    encoded_tensor = torch.from_numpy(encoded_glyph_batch).float()

    command_tensor = encoded_tensor[:, :, :NODE_COMMAND_WIDTH]
    coord_tensor_relative_normalized = encoded_tensor[:, :, NODE_COMMAND_WIDTH:]

    # 3. Denormalize the relative coordinates
    coord_tensor_relative_denormalized = (
        coord_tensor_relative_normalized * MAX_COORDINATE
    )

    # 4. Unroll to get absolute coordinates
    coord_tensor_absolute = unroll_relative_coords(
        command_tensor, coord_tensor_relative_denormalized
    )

    # The absolute positions of the nodes are:
    # Node 1: [50, 50]
    # Node 2: [100, 100]
    # Node 3: [50, 150]

    # The unrolled coordinates should match these.
    # The unrolled tensor has shape (batch, seq_len, num_coords).
    # We have one contour, so the first command is SOC.
    # The node commands start from index 1.

    # The first node's absolute position should be at index 1.
    unrolled_pos1 = coord_tensor_absolute[0, 1, 0:2].numpy()
    np.testing.assert_allclose(unrolled_pos1, [50, 50], atol=1e-4)

    # The second node's absolute position should be at index 2.
    unrolled_pos2 = coord_tensor_absolute[0, 2, 0:2].numpy()
    np.testing.assert_allclose(unrolled_pos2, [100, 100], atol=1e-4)

    # The third node's absolute position should be at index 3.
    unrolled_pos3 = coord_tensor_absolute[0, 3, 0:2].numpy()
    np.testing.assert_allclose(unrolled_pos3, [50, 150], atol=1e-4)


def test_node_contour_roll():
    # 1. Create a contour with a known sequence of nodes.
    contour = NodeContour([])
    nodes_data = [
        [0, 0],
        [100, 0],
        [100, 100],
        [0, 100],
    ]
    nodes = [Node(coords, contour) for coords in nodes_data]
    contour.nodes = nodes

    # Check initial state
    assert len(contour.nodes) == 4
    np.testing.assert_array_equal(contour.nodes[0].coordinates, [0, 0])

    # 2. Test a positive shift
    contour.roll(1)
    assert len(contour.nodes) == 4
    # Node at index 1 ([100, 0]) should now be at index 0
    np.testing.assert_array_equal(contour.nodes[0].coordinates, [100, 0])
    # Original first node ([0, 0]) should now be at the end
    np.testing.assert_array_equal(contour.nodes[3].coordinates, [0, 0])

    # 3. Test a negative shift (from the new state) back to original
    contour.roll(-1)
    assert len(contour.nodes) == 4
    np.testing.assert_array_equal(contour.nodes[0].coordinates, [0, 0])
    np.testing.assert_array_equal(contour.nodes[1].coordinates, [100, 0])

    # 4. Test a large shift that wraps around
    # 5 % 4 = 1. Same as rolling by 1.
    contour.roll(5)
    np.testing.assert_array_equal(contour.nodes[0].coordinates, [100, 0])
    np.testing.assert_array_equal(contour.nodes[3].coordinates, [0, 0])


def test_glyph_simplify():
    # Test one with cross-contour overlaps
    font_path = Path("tests/data/Roboto[wdth,wght].ttf")
    location = {"wght": 400.0}
    codepoint = ord("A")

    glyph = Glyph(font_path, codepoint, location)
    vectorized_glyph = glyph.vectorize(remove_overlaps=False)
    assert len(vectorized_glyph.to_node_glyph().contours) == 3

    vectorized_glyph = glyph.vectorize(remove_overlaps=True)
    # Should now have two contours
    assert len(vectorized_glyph.to_node_glyph().contours) == 2
