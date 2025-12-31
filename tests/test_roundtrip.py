from glyphogen.nodeglyph import NodeGlyph
import torch
import numpy as np
from glyphogen.glyph import SVGGlyph
from glyphogen.command_defs import (
    NodeCommand,
)


def test_coordinate_transforms():
    """
    Tests that the coordinate space transformations are perfect inverses.
    """
    box = [100, 100, 300, 400]  # x1, y1, x2, y2

    # Create a sample sequence tensor
    # [M, L, N]
    commands = torch.zeros(3, NodeCommand.command_width)
    commands[0, NodeCommand.encode_command("M")] = 1.0
    commands[1, NodeCommand.encode_command("L")] = 1.0
    commands[2, NodeCommand.encode_command("N")] = 1.0

    coords = torch.tensor(
        [
            [150, 250, 0, 0, 0, 0],  # M (absolute)
            [10, 20, 0, 0, 0, 0],  # L (relative)
            [-5, 15, 5, 5, -5, -5],  # N (relative)
        ],
        dtype=torch.float32,
    )

    # Pad to full sequence width
    padding = torch.zeros(3, NodeCommand.coordinate_width - coords.shape[1])
    coords = torch.cat([coords, padding], dim=1)

    sequence = torch.cat([commands, coords], dim=1)

    # Round trip
    sequence_mask_space = NodeCommand.image_space_to_mask_space(sequence, box)
    assert (
        sequence_mask_space[0, NodeCommand.command_width] == -0.5
    )  # One-quarter of the way across box, which is now -1 to +1
    sequence_roundtrip = NodeCommand.mask_space_to_image_space(sequence_mask_space, box)

    assert torch.allclose(
        sequence[:, NodeCommand.command_width :],
        sequence_roundtrip[:, NodeCommand.command_width :],
        atol=1e-6,
    ), f"Coordinate space round-trip failed:\n{sequence[:, NodeCommand.command_width :]}\n{sequence_roundtrip[:, NodeCommand.command_width :]}"


import pytest
from pathlib import Path
from glyphogen.hyperparameters import ALPHABET
from glyphogen.glyph import Glyph

# ... (existing test_coordinate_transforms and test_nodeglyph_encoding_decoding) ...


@pytest.mark.parametrize("char_to_test", list(ALPHABET))
def test_real_glyph_roundtrip(char_to_test):
    """
    Tests the full round-trip process on a real glyph from a font file.
    """
    font_path = Path("NotoSans[wdth,wght].ttf")
    if not font_path.exists():
        pytest.skip("NotoSans[wdth,wght].ttf not found, skipping real glyph test.")
    if char_to_test == "l":
        pytest.skip("Skipping 'l' as it has a bad contour construction.")

    # 1. Load original glyph and convert to NodeGlyph
    glyph = Glyph(font_path, ord(char_to_test), {})
    svg_glyph_orig = None
    try:
        svg_glyph_orig = glyph.vectorize()
    except NotImplementedError:
        pytest.skip(
            f"Skipping glyph '{char_to_test}' due to fontTools NotImplementedError."
        )
    if not svg_glyph_orig:
        return  # Skip empty glyphs

    # Get the original SVG string for comparison
    svg_orig_str = svg_glyph_orig.to_svg_string()

    # Now convert to NodeGlyph
    nodeglyph_orig = svg_glyph_orig.to_node_glyph()
    print("Debug node glyph: ", nodeglyph_orig.to_debug_string())
    # and back again
    svg_glyph_roundtrip = SVGGlyph.from_node_glyph(nodeglyph_orig)
    svg_roundtrip_str = svg_glyph_roundtrip.to_svg_string()

    # Compare original and round-tripped SVG strings
    if svg_orig_str != svg_roundtrip_str:
        print(f"Original SVG:\n{svg_orig_str}")
        print(f"Roundtrip SVG:\n{svg_roundtrip_str}")
    assert svg_orig_str == svg_roundtrip_str

    # 2a. NodeGlyph -> List[List[NodeCommand]]
    contours_commands = nodeglyph_orig.command_lists(NodeCommand)

    # 2b. List[List[NodeCommand]] -> NodeGlyph
    nodeglyph_reconstructed = NodeGlyph(
        [
            NodeCommand.contour_from_commands(contour, tolerant=False)
            for contour in contours_commands
        ],
        nodeglyph_orig.origin,
    )

    assert (
        nodeglyph_orig == nodeglyph_reconstructed
    ), f"NodeGlyph reconstruction from command lists failed:\nOriginal:\n{nodeglyph_orig.to_debug_string()}\nReconstructed:\n{nodeglyph_reconstructed.to_debug_string()}"

    # }."

    # 3a. NodeGlyph -> encoded sequence
    encoded_sequences = nodeglyph_orig.encode(NodeCommand)

    # This can happen for blank glyphs like 'space'
    if encoded_sequences is None:
        assert svg_orig_str == ""
        return

    # 3b. Encoded sequence -> NodeGlyph
    nodeglyph_roundtrip = NodeGlyph.decode(encoded_sequences, NodeCommand)

    # 4. Compare original and round-tripped NodeGlyph objects
    assert (
        nodeglyph_orig == nodeglyph_roundtrip
    ), "NodeGlyph encoding round-trip failed."
