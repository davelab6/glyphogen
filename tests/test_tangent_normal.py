from pathlib import Path
import numpy as np
import pytest
from glyphogen.command_defs import TangentNormalCommand
from glyphogen.glyph import Glyph
from glyphogen.hyperparameters import ALPHABET
from glyphogen.nodeglyph import Node, NodeContour, NodeGlyph


def test_tangent_normal_line_roundtrip():
    """
    Tests that a simple square contour can be encoded into TangentNormalCommands
    and decoded back to the original contour.
    """
    # 1. Create a simple square NodeContour
    nodes = [
        Node(np.array([0, 0]), contour=None),
        Node(np.array([100, 0]), contour=None),
        Node(np.array([100, 100]), contour=None),
        Node(np.array([0, 100]), contour=None),
    ]
    original_contour = NodeContour(nodes)
    for node in nodes:
        node._contour = original_contour

    # 2. Encode it
    commands = TangentNormalCommand.emit(original_contour.nodes)

    # 3. Decode it
    decoded_contour = TangentNormalCommand.contour_from_commands(commands)

    # 4. Compare
    # The decoded contour should have the same number of nodes
    assert len(original_contour.nodes) == len(
        decoded_contour.nodes
    ), f"Expected {len(original_contour.nodes)} nodes, got {len(decoded_contour.nodes)}"

    for i, original_node in enumerate(original_contour.nodes):
        decoded_node = decoded_contour.nodes[i]
        assert np.allclose(
            original_node.coordinates, decoded_node.coordinates, atol=1e-4
        ), f"Node {i} mismatch: Original {original_node.coordinates}, Decoded {decoded_node.coordinates}"

    print("\n--- TangentNormalCommand line roundtrip test passed! ---")


def test_tangent_normal_curve_roundtrip():
    """
    Tests that a 'D'-shaped contour with a curve can be encoded and decoded.
    """
    # 1. Create a 'D' shape. It has two nodes: top-left and bottom-left.
    # The segment from node 0 to 1 is a curve.
    # The segment from node 1 to 0 is a straight line.
    h_len = 100 * 2 / 3  # Approximation for a nice curve

    nodes = [
        Node(
            np.array([0, 100]),
            out_handle=np.array([h_len, 100]),
            in_handle=None,
            contour=None,
        ),
        Node(
            np.array([0, 0]),
            out_handle=None,
            in_handle=np.array([h_len, 0]),
            contour=None,
        ),
    ]
    original_contour = NodeContour(nodes)
    for node in nodes:
        node._contour = original_contour

    # 2. Encode it
    commands = TangentNormalCommand.emit(original_contour.nodes)

    # 3. Decode it
    decoded_contour = TangentNormalCommand.contour_from_commands(commands)

    # 4. Compare
    # The decoded contour should have the same number of nodes
    assert len(original_contour.nodes) == len(
        decoded_contour.nodes
    ), f"Expected {len(original_contour.nodes)} nodes, got {len(decoded_contour.nodes)}"

    for i, original_node in enumerate(original_contour.nodes):
        decoded_node = decoded_contour.nodes[i]
        assert np.allclose(
            original_node.coordinates, decoded_node.coordinates, atol=1e-4
        ), f"Node {i} coordinate mismatch: Original {original_node.coordinates}, Decoded {decoded_node.coordinates}"

        if original_node.out_handle is not None:
            assert (
                decoded_node.out_handle is not None
            ), f"Node {i} decoded out-handle is None"
            assert np.allclose(
                original_node.out_handle, decoded_node.out_handle, atol=1e-4
            ), f"Node {i} out-handle mismatch: Original {original_node.out_handle}, Decoded {decoded_node.out_handle}"

        if original_node.in_handle is not None:
            assert (
                decoded_node.in_handle is not None
            ), f"Node {i} decoded in-handle is None"
            assert np.allclose(
                original_node.in_handle, decoded_node.in_handle, atol=1e-4
            ), f"Node {i} in-handle mismatch: Original {original_node.in_handle}, Decoded {decoded_node.in_handle}"

    print("\n--- TangentNormalCommand curve roundtrip test passed! ---")


# @pytest.mark.parametrize("char_to_test", list(ALPHABET))
# def test_real_glyph_roundtrip(char_to_test):
#     """
#     Tests the full round-trip process on a real glyph from a font file.
#     """
#     font_path = Path("NotoSans[wdth,wght].ttf")
#     if not font_path.exists():
#         pytest.skip("NotoSans[wdth,wght].ttf not found, skipping real glyph test.")
#     if char_to_test == "l":
#         pytest.skip("Skipping 'l' as it has a bad contour construction.")

#     # 1. Load original glyph and convert to NodeGlyph
#     glyph = Glyph(font_path, ord(char_to_test), {})
#     svg_glyph_orig = None
#     try:
#         svg_glyph_orig = glyph.vectorize()
#     except NotImplementedError:
#         pytest.skip(
#             f"Skipping glyph '{char_to_test}' due to fontTools NotImplementedError."
#         )
#     if not svg_glyph_orig:
#         return  # Skip empty glyphs

#     # Get the original SVG string for comparison
#     svg_orig_str = svg_glyph_orig.to_svg_string()

#     # Now convert to NodeGlyph
#     nodeglyph_orig = svg_glyph_orig.to_node_glyph()

#     # 2a. NodeGlyph -> List[List[NodeCommand]]
#     contours_commands = nodeglyph_orig.command_lists(TangentNormalCommand)

#     # 2b. List[List[NodeCommand]] -> NodeGlyph
#     nodeglyph_reconstructed = NodeGlyph(
#         [
#             TangentNormalCommand.contour_from_commands(contour, tolerant=False)
#             for contour in contours_commands
#         ],
#         nodeglyph_orig.origin,
#     )

#     # Assert that the on-curve points are preserved.
#     # We don't assert full handle equality, because the NS encoding "perfects"
#     # the handles of almost-smooth nodes, which is a desirable cleanup.
#     assert len(nodeglyph_orig.contours) == len(nodeglyph_reconstructed.contours)
#     for i, orig_contour in enumerate(nodeglyph_orig.contours):
#         reconstructed_contour = nodeglyph_reconstructed.contours[i]
#         assert len(orig_contour.nodes) == len(reconstructed_contour.nodes)
#         for j, orig_node in enumerate(orig_contour.nodes):
#             reconstructed_node = reconstructed_contour.nodes[j]
#             assert np.allclose(
#                 orig_node.coordinates, reconstructed_node.coordinates, atol=1e-5
#             ), f"Node {j} in contour {i} of glyph '{char_to_test}' has mismatched coordinates"
#             # Optional: check that smoothness is preserved
#             if orig_node.is_smooth:
#                 assert (
#                     reconstructed_node.is_smooth
#                 ), f"Node {j} in contour {i} of glyph '{char_to_test}' lost its smoothness"

#     # 3a. NodeGlyph -> encoded sequence
#     encoded_sequences = nodeglyph_orig.encode(TangentNormalCommand)

#     # This can happen for blank glyphs like 'space'
#     if encoded_sequences is None:
#         assert svg_orig_str == ""
#         return

#     # 3b. Encoded sequence -> NodeGlyph
#     nodeglyph_roundtrip = NodeGlyph.decode(encoded_sequences, TangentNormalCommand)

#     # 4. Compare original and round-tripped NodeGlyph objects
#     # We assert on-curve coordinate preservation, not perfect handle equality.
#     assert len(nodeglyph_orig.contours) == len(nodeglyph_roundtrip.contours)
#     for i, orig_contour in enumerate(nodeglyph_orig.contours):
#         reconstructed_contour = nodeglyph_roundtrip.contours[i]
#         assert len(orig_contour.nodes) == len(reconstructed_contour.nodes)
#         for j, orig_node in enumerate(orig_contour.nodes):
#             reconstructed_node = reconstructed_contour.nodes[j]
#             assert np.allclose(
#                 orig_node.coordinates, reconstructed_node.coordinates, atol=1e-5
#             ), f"Node {j} in contour {i} of glyph '{char_to_test}' has mismatched coordinates after encode/decode"
