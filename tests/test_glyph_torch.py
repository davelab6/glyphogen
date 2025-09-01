from pathlib import Path
from glyphogen.glyph import (
    Glyph,
    SVGGlyph,
    NodeGlyph,
    NodeCommand,
    SVGCommand,
    NodeContour,
    NODE_COMMAND_WIDTH,
)
from glyphogen.hyperparameters import GEN_IMAGE_SIZE
import numpy as np


def test_glyph_extraction():
    font_path = Path("tests/data/NotoSans[wdth,wght].ttf")
    location = {"wght": 400.0}
    codepoint = ord("l")

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

    # Remove the batch dimension for from_numpy
    decoded_glyph = NodeGlyph.from_numpy(command_tensor[0], coord_tensor[0])

    assert len(decoded_glyph.contours) == 1
    assert len(decoded_glyph.contours[0].commands) == 4
    assert len(decoded_glyph.contours) == 1
    assert decoded_glyph.contours[0].commands[0].command == "NCO"
    assert decoded_glyph.contours[0].commands[1].command == "NV"
    assert decoded_glyph.contours[0].commands[2].command == "NCI"
    assert decoded_glyph.contours[0].commands[3].command == "Z"


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

    assert len(node_glyph.commands) == 4 # 3 nodes + Z
    assert node_glyph.commands[0].command == "L" # Start of line
    assert node_glyph.commands[0].coordinates == [10, 20]
    assert node_glyph.commands[1].command == "NCO" # Line to curve
    assert node_glyph.commands[1].coordinates == [30, 40, 20, 20] # 30,40 pos, 50,60 handle -> 20,20 relative
    assert node_glyph.commands[2].command == "NCI" # Curve to line
    assert node_glyph.commands[2].coordinates == [90, 100, -20, -20] # 90,100 pos, 70,80 handle -> -20,-20 relative
    assert node_glyph.commands[3].command == "Z"


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

    # Should be 4 L nodes and a Z
    assert len(node_glyph.commands) == 5
    assert node_glyph.commands[0].command == "L"
    assert node_glyph.commands[1].command == "L"
    assert node_glyph.commands[2].command == "L"
    assert node_glyph.commands[3].command == "L"
    assert node_glyph.commands[4].command == "Z"

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

def test_complex_svg_to_node_glyph_conversion():
    # M 247 493 C 383 493 493 383 493 246 C 493 169 458 101 403 56 L 247 0 L 0 246 C 0 383 110 493 247 493 Z
    svg_commands = [
        SVGCommand("M", [247, 493]),
        SVGCommand("C", [383, 493, 493, 383, 493, 246]),
        SVGCommand("C", [493, 169, 458, 101, 403, 56]),
        SVGCommand("L", [247, 0]),
        SVGCommand("L", [0, 246]),
        SVGCommand("C", [0, 383, 110, 493, 247, 493]),
        SVGCommand("Z", []),
    ]
    svg_glyph = SVGGlyph(svg_commands)
    node_glyph = svg_glyph.to_node_glyph()

    # There are 5 points in this path, so 5 nodes + Z
    assert len(node_glyph.commands) == 6
    # The nodes are at: 247,493; 493,246; 403,56; 247,0; 0,246
    # But we will have normalized them so that they are
    # 0,246; 247,493; 493,246; 403,56; 247,0
    # Let's check them in order.

    # Node 1: 0, 246. In: L, Out: C. Should be NCO.
    assert node_glyph.commands[0].command == "NCO"
    # Node 1: 247, 493. In: C, Out: C. All handles are aligned at y=493. Should be NH.
    assert node_glyph.commands[1].command == "NH"
    # Node 2: 493, 246. In: C, Out: C. Handle aligned vertically at x=493. Should be NV.
    assert node_glyph.commands[2].command == "NV"
    # Node 3: 403, 56. In: C, Out: L. Should be NCI.
    assert node_glyph.commands[3].command == "NCI"
    # Node 4: 247, 0. In: L, Out: L. Should be L.
    assert node_glyph.commands[4].command == "L"
    # Z command
    assert node_glyph.commands[5].command == "Z"

    # Test round-trip
    reconverted_svg_glyph = node_glyph.to_svg_glyph()
    # assert len(reconverted_svg_glyph.commands) == len(svg_commands)

    # Visually the same, but order is now normalized
    assert reconverted_svg_glyph.to_svg_string() == "M 0 246 C 0 383 110 493 247 493 C 383 493 493 383 493 246 C 493 169 458 101 403 56 L 247 0 Z"
