import torch
import numpy as np
from glyphogen.glyph import SVGGlyph, NodeGlyph
from glyphogen.coordinate import image_space_to_mask_space, mask_space_to_image_space
from glyphogen.command_defs import NODE_GLYPH_COMMANDS, COORDINATE_WIDTH, NODE_COMMAND_WIDTH, SVGCommand

def test_coordinate_transforms():
    """
    Tests that the coordinate space transformations are perfect inverses.
    """
    box = [100, 100, 300, 400]  # x1, y1, x2, y2
    
    # Create a sample sequence tensor
    # [M, L, N]
    commands = torch.zeros(3, NODE_COMMAND_WIDTH)
    commands[0, list(NODE_GLYPH_COMMANDS.keys()).index("M")] = 1.0
    commands[1, list(NODE_GLYPH_COMMANDS.keys()).index("L")] = 1.0
    commands[2, list(NODE_GLYPH_COMMANDS.keys()).index("N")] = 1.0

    coords = torch.tensor([
        [150, 250, 0, 0, 0, 0],    # M (absolute)
        [10, 20, 0, 0, 0, 0],      # L (relative)
        [-5, 15, 5, 5, -5, -5],    # N (relative)
    ], dtype=torch.float32)

    # Pad to full sequence width
    padding = torch.zeros(3, COORDINATE_WIDTH - coords.shape[1])
    coords = torch.cat([coords, padding], dim=1)
    
    sequence = torch.cat([commands, coords], dim=1)

    # Round trip
    sequence_mask_space = image_space_to_mask_space(sequence, box)
    sequence_roundtrip = mask_space_to_image_space(sequence_mask_space, box)

    assert torch.allclose(sequence, sequence_roundtrip, atol=1e-6)

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

    # 1. Load original glyph and convert to NodeGlyph
    glyph = Glyph(font_path, ord(char_to_test), {})
    try:
        nodeglyph_orig = glyph.vectorize().to_node_glyph()
    except NotImplementedError:
        pytest.skip(f"Skipping glyph '{char_to_test}' due to fontTools NotImplementedError.")
    
    # Get the original SVG string for comparison
    svg_orig_str = nodeglyph_orig.to_svg_glyph().to_svg_string()

    # 2. NodeGlyph -> encoded sequence
    encoded_sequences = nodeglyph_orig.encode()
    
    # This can happen for blank glyphs like 'space'
    if encoded_sequences is None:
        assert svg_orig_str == ""
        return

    # 3. Encoded sequence -> NodeGlyph
    reconstructed_input = []
    for seq in encoded_sequences:
        seq_tensor = torch.from_numpy(seq)
        cmds = seq_tensor[:, :NODE_COMMAND_WIDTH]
        coords = seq_tensor[:, NODE_COMMAND_WIDTH:]
        reconstructed_input.append((cmds, coords))

    nodeglyph_roundtrip = NodeGlyph.from_numpy(reconstructed_input)
    
    # 4. NodeGlyph -> SVG
    # svg_roundtrip_str = nodeglyph_roundtrip.to_svg_glyph().to_svg_string()

    # 5. Compare original and round-tripped NodeGlyph objects
    assert nodeglyph_orig == nodeglyph_roundtrip

