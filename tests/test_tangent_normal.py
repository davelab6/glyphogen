from pathlib import Path
import numpy as np
import pytest
import torch
from glyphogen.command_defs import RelativePolarCommand
from glyphogen.glyph import SVGGlyph
from glyphogen.nodeglyph import Node, NodeContour, NodeGlyph


def test_relative_polar_line_roundtrip():
    """
    Tests that a simple square contour can be encoded into TangentNormalCommands
    and decoded back to the original contour, using L_LEFT.
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
    commands = RelativePolarCommand.emit(original_contour.nodes)

    # 3. Check for correct commands
    assert commands[0].command == "SOS"
    assert commands[1].command == "M"
    assert commands[2].command == "L_POLAR"  # Initial straight line
    r0, phi0 = commands[2].coordinates
    assert np.isclose(r0, 100)
    assert np.isclose(phi0, 0.0)
    # Next three are left turns of ~+pi/2 with length 100
    assert commands[3].command == "L_LEFT_90"
    r1 = commands[3].coordinates[0]
    assert np.isclose(r1, 100)
    assert commands[4].command == "L_LEFT_90"
    r2 = commands[4].coordinates[0]
    assert np.isclose(r2, 100)
    assert commands[5].command == "L_LEFT_90"
    r3 = commands[5].coordinates[0]
    assert np.isclose(r3, 100)

    # 4. Decode it
    decoded_contour = RelativePolarCommand.contour_from_commands(commands)

    # 5. Compare
    assert len(original_contour.nodes) == len(
        decoded_contour.nodes
    ), f"Expected {len(original_contour.nodes)} nodes, got {len(decoded_contour.nodes)}"

    for i, original_node in enumerate(original_contour.nodes):
        decoded_node = decoded_contour.nodes[i]
        assert np.allclose(
            original_node.coordinates, decoded_node.coordinates, atol=1e-4
        ), f"Node {i} mismatch: Original {original_node.coordinates}, Decoded {decoded_node.coordinates}"

    print("\n--- RelativePolarCommand line roundtrip test passed! ---")


def test_relative_polar_z_shape_roundtrip():
    """
    Tests that a Z-shaped contour can be encoded and decoded correctly,
    including diagonal lines with non-zero phi.
    """
    nodes = [
        Node(np.array([0, 100]), contour=None),
        Node(np.array([100, 100]), contour=None),
        Node(np.array([0, 0]), contour=None),
        Node(np.array([100, 0]), contour=None),
    ]
    original_contour = NodeContour(nodes)
    for node in nodes:
        node._contour = original_contour

    commands = RelativePolarCommand.emit(original_contour.nodes)

    # Expected commands: SOS, M, L_POLAR (0,100 to 100,100), L_POLAR (100,100 to 0,0), L_POLAR (0,0 to 100,0), L_POLAR (100,0 to 0,100), EOS
    assert commands[0].command == "SOS"
    assert commands[1].command == "M"
    assert np.allclose(commands[1].coordinates, [0, 100])

    # Segment 1: (0,100) -> (100,100) (straight right)
    assert commands[2].command == "L_POLAR"
    r, phi = commands[2].coordinates
    assert np.isclose(r, 100)
    assert np.isclose(phi, 0.0)

    # Segment 2: (100,100) -> (0,0) (diagonal down-left)
    assert commands[3].command == "L_POLAR"
    r, phi = commands[3].coordinates
    assert np.isclose(r, np.sqrt(100**2 + 100**2))
    assert np.isclose(phi, -np.pi * 3 / 4) # Relative to f_hat=[1,0]

    # Segment 3: (0,0) -> (100,0) (straight right)
    assert commands[4].command == "L_POLAR"
    r, phi = commands[4].coordinates
    assert np.isclose(r, 100)
    assert np.isclose(phi, np.pi * 3 / 4) # Relative to f_hat=[cos(-3pi/4), sin(-3pi/4)]

    # Segment 4: (100,0) -> (0,100) (diagonal up-left, closing)
    assert commands[5].command == "L_POLAR"
    r, phi = commands[5].coordinates
    assert np.isclose(r, np.sqrt(100**2 + 100**2))
    assert np.isclose(phi, np.pi * 3 / 4) # Relative to f_hat=[1,0]

    assert commands[6].command == "EOS"

    # Decode and compare
    decoded_contour = RelativePolarCommand.contour_from_commands(commands)
    assert len(original_contour.nodes) == len(decoded_contour.nodes)
    for i, original_node in enumerate(original_contour.nodes):
        decoded_node = decoded_contour.nodes[i]
        assert np.allclose(
            original_node.coordinates, decoded_node.coordinates, atol=1e-4
        ), f"Node {i} mismatch: Original {original_node.coordinates}, Decoded {decoded_node.coordinates}"

    # Unroll and verify absolute coordinates
    command_tensors = []
    coord_tensors = []
    max_coords = RelativePolarCommand.coordinate_width
    for cmd in commands:
        command_tensors.append(RelativePolarCommand.encode_command_one_hot(cmd.command))
        padded_coords = cmd.coordinates + [0] * (max_coords - len(cmd.coordinates))
        coord_tensors.append(torch.tensor(padded_coords, dtype=torch.float32))
    
    sequence_tensor = torch.cat([torch.stack(command_tensors), torch.stack(coord_tensors)], dim=1)
    unrolled_sequence = RelativePolarCommand.unroll_relative_coordinates(sequence_tensor)
    _, abs_coords = RelativePolarCommand.split_tensor(unrolled_sequence)

    assert np.allclose(abs_coords[1, 0:2], [0, 100]) # M
    assert np.allclose(abs_coords[2, 0:2], [100, 100]) # L_POLAR (0,100) -> (100,100)
    assert np.allclose(abs_coords[3, 0:2], [0, 0]) # L_POLAR (100,100) -> (0,0)
    assert np.allclose(abs_coords[4, 0:2], [100, 0]) # L_POLAR (0,0) -> (100,0)
    assert np.allclose(abs_coords[5, 0:2], [0, 100]) # L_POLAR (100,0) -> (0,100)

    print("\n--- RelativePolarCommand Z-shape roundtrip test passed! ---")


def test_relative_polar_real_z_glyph_roundtrip():
    """
    Tests the full round-trip process for a real Z-shaped glyph using
    RelativePolarCommand.
    """
    svg_string = "M 34 0 L 584 0 L 584 96 L 192 96 L 572 638 L 572 690 L 49 690 L 49 594 L 414 594 L 34 52 L 34 0 Z"
    svg_glyph = SVGGlyph.from_svg_string(svg_string)
    nodeglyph_orig = svg_glyph.to_node_glyph()

    # 2a. NodeGlyph -> List[List[RelativePolarCommand]]
    contours_commands = nodeglyph_orig.command_lists(RelativePolarCommand)
    assert len(contours_commands) == 1
    commands = contours_commands[0]

    # Manually trace expected commands and coordinates
    # Initial f_hat = [1,0]
    # M 34 0
    assert commands[0].command == "SOS"
    assert commands[1].command == "M"
    assert np.allclose(commands[1].coordinates, [34, 0])

    # Segment 1: (34,0) -> (584,0) (L 550 0)
    # delta_pos = [550, 0], r = 550, phi = 0 (relative to f_hat=[1,0])
    assert commands[2].command == "L_POLAR"
    assert np.isclose(commands[2].coordinates[0], 550)
    assert np.isclose(commands[2].coordinates[1], 0)

    # Segment 2: (584,0) -> (584,96) (L 0 96)
    # f_hat is still [1,0]
    # delta_pos = [0, 96], r = 96, phi = pi/2 (relative to f_hat=[1,0])
    assert commands[3].command == "L_LEFT_90"
    assert np.isclose(commands[3].coordinates[0], 96)

    # Segment 3: (584,96) -> (192,96) (L -392 0)
    # f_hat is now [0,1]
    # delta_pos = [-392, 0], r = 392, phi = pi/2 (relative to f_hat=[0,1])
    assert commands[4].command == "L_LEFT_90"
    assert np.isclose(commands[4].coordinates[0], 392)

    # Segment 4: (192,96) -> (572,638) (L 380 542)
    # f_hat is now [-1,0]
    # delta_pos = [380, 542], r = sqrt(380^2 + 542^2) = 661.09
    # r_hat = [0,-1]
    # phi = arctan2(dot([380,542], [0,-1]), dot([380,542], [-1,0])) = arctan2(-542, -380) = -2.19 rad
    assert commands[5].command == "L_POLAR"
    assert np.isclose(commands[5].coordinates[0], np.sqrt(380**2 + 542**2))
    assert np.isclose(commands[5].coordinates[1], np.arctan2(-542, -380))

    # Segment 5: (572,638) -> (572,690) (L 0 52)
    # f_hat is now direction of [380,542]
    # delta_pos = [0, 52], r = 52
    # r_hat = [-f_hat[1], f_hat[0]]
    # phi = arctan2(dot([0,52], r_hat), dot([0,52], f_hat))
    # This is getting too complex to manually calculate.
    # Let's just check the roundtrip.

    # 3a. NodeGlyph -> encoded sequence
    encoded_sequences = nodeglyph_orig.encode(RelativePolarCommand)

    # This can happen for blank glyphs like 'space'
    if encoded_sequences is None:
        assert svg_glyph.to_svg_string() == ""
        return

    # 3b. Encoded sequence -> NodeGlyph
    nodeglyph_roundtrip = NodeGlyph.decode(encoded_sequences, RelativePolarCommand)

    # 4. Compare original and round-tripped NodeGlyph objects
    assert len(nodeglyph_orig.contours) == len(nodeglyph_roundtrip.contours)
    for i, orig_contour in enumerate(nodeglyph_orig.contours):
        reconstructed_contour = nodeglyph_roundtrip.contours[i]
        assert len(orig_contour.nodes) == len(reconstructed_contour.nodes)
        for j, orig_node in enumerate(orig_contour.nodes):
            reconstructed_node = reconstructed_contour.nodes[j]
            assert np.allclose(
                orig_node.coordinates, reconstructed_node.coordinates, atol=1e-5
            ), f"Node {j} in contour {i} of glyph has mismatched coordinates"
            # For now, we don't have handles, so we don't check them.

    print("\n--- RelativePolarCommand real Z-glyph roundtrip test passed! ---")


# def test_relative_polar_curve_roundtrip():
#     """
#     Tests that a 'D'-shaped contour with a curve can be encoded and decoded.
#     """
#     # 1. Create a 'D' shape. It has two nodes: top-left and bottom-left.
#     # The segment from node 0 to 1 is a curve.
#     # The segment from node 1 to 0 is a straight line.
#     h_len = 100 * 2 / 3  # Approximation for a nice curve

#     nodes = [
#         Node(
#             np.array([0, 100]),
#             out_handle=np.array([h_len, 100]),
#             in_handle=None,
#             contour=None,
#         ),
#         Node(
#             np.array([0, 0]),
#             out_handle=None,
#             in_handle=np.array([h_len, 0]),
#             contour=None,
#         ),
#     ]
#     original_contour = NodeContour(nodes)
#     for node in nodes:
#         node._contour = original_contour

#     # 2. Encode it
#     commands = RelativePolarCommand.emit(original_contour.nodes)

#     # 3. Decode it
#     decoded_contour = RelativePolarCommand.contour_from_commands(commands)

#     # 4. Compare
#     # The decoded contour should have the same number of nodes
#     assert len(original_contour.nodes) == len(
#         decoded_contour.nodes
#     ), f"Expected {len(original_contour.nodes)} nodes, got {len(decoded_contour.nodes)}"

#     for i, original_node in enumerate(original_contour.nodes):
#         decoded_node = decoded_contour.nodes[i]
#         assert np.allclose(
#             original_node.coordinates, decoded_node.coordinates, atol=1e-4
#         ), f"Node {i} coordinate mismatch: Original {original_node.coordinates}, Decoded {decoded_node.coordinates}"

#         if original_node.out_handle is not None:
#             assert (
#                 decoded_node.out_handle is not None
#             ), f"Node {i} decoded out-handle is None"
#             assert np.allclose(
#                 original_node.out_handle, decoded_node.out_handle, atol=1e-4
#             ), f"Node {i} out-handle mismatch: Original {original_node.out_handle}, Decoded {decoded_node.out_handle}"

#         if original_node.in_handle is not None:
#             assert (
#                 decoded_node.in_handle is not None
#             ), f"Node {i} decoded in-handle is None"
#             assert np.allclose(
#                 original_node.in_handle, decoded_node.in_handle, atol=1e-4
#             ), f"Node {i} in-handle mismatch: Original {original_node.in_handle}, Decoded {decoded_node.in_handle}"

#     print("\n--- RelativePolarCommand curve roundtrip test passed! ---")


# def test_relative_polar_unroll_relative():
#     """
#     Tests that the unroll_relative_coordinates method correctly converts
#     relative tangent-normal commands to absolute coordinates.
#     """
#     # 1. Create a 'D' shape
#     h_len = 100 * 2 / 3
#     commands = [
#         RelativePolarCommand("SOS", []),
#         RelativePolarCommand("M", [0, 100]),
#         RelativePolarCommand("L_POLAR", [0.0, 0.0]),
#         # Move south by 100 in polar (r=100, phi=-pi/2), with symmetric handles (in at -u, out at +u)
#         RelativePolarCommand("N_TANGENT", [100, -np.pi / 2, h_len, h_len]),
#         # Move north by 100 (r=100, phi=+pi/2) back to (0,100)
#         RelativePolarCommand("L_POLAR", [100, np.pi / 2]),
#         RelativePolarCommand("EOS", []),
#     ]

#     # 2. Convert to a tensor
#     command_tensors = []
#     coord_tensors = []
#     max_coords = RelativePolarCommand.coordinate_width
#     for cmd in commands:
#         command_tensors.append(RelativePolarCommand.encode_command_one_hot(cmd.command))
#         padded_coords = cmd.coordinates + [0] * (max_coords - len(cmd.coordinates))
#         coord_tensors.append(torch.tensor(padded_coords, dtype=torch.float32))

#     sequence_tensor = torch.cat(
#         [torch.stack(command_tensors), torch.stack(coord_tensors)], dim=1
#     )

#     # 3. Unroll it
#     unrolled_sequence = RelativePolarCommand.unroll_relative_coordinates(
#         sequence_tensor
#     )
#     _, abs_coords = RelativePolarCommand.split_tensor(unrolled_sequence)

#     # 4. Verify the absolute coordinates
#     # M command
#     assert np.allclose(abs_coords[1, 0:2], [0, 100])
#     # Zero-step command leaves position unchanged
#     assert np.allclose(abs_coords[2, 0:2], [0, 100])
#     # N command
#     assert np.allclose(abs_coords[3, 0:2], [0, 0])  # pos
#     assert np.allclose(abs_coords[3, 2:4], [0 - h_len, 0])  # in handle along -x
#     assert np.allclose(
#         abs_coords[3, 4:6], [0 + h_len, 0]
#     )  # out handle expressed at destination in the fixed frame
#     # L command
#     assert np.allclose(abs_coords[4, 0:2], [0, 100])

#     print("\n--- RelativePolarCommand unroll relative test passed! ---")


# def test_relative_polar_space_conversion_roundtrip():
#     """
#     Tests that the coordinate space conversion methods can roundtrip from
#     image space to mask space and back.
#     """
#     # 1. Create a sequence of commands in image space
#     commands = [
#         RelativePolarCommand("SOS", []),
#         RelativePolarCommand("M", [100, 200]),
#         RelativePolarCommand("N_POLAR", [50, 0.2, 10, -0.3, 12, 0.4]),
#         RelativePolarCommand("L_POLAR", [30, 0.1]),
#         RelativePolarCommand("EOS", []),
#     ]

#     # 2. Convert to a tensor
#     command_tensors = []
#     coord_tensors = []
#     max_coords = RelativePolarCommand.coordinate_width
#     for cmd in commands:
#         command_tensors.append(RelativePolarCommand.encode_command_one_hot(cmd.command))

#         # Pad coordinates to max_coords
#         padded_coords = cmd.coordinates + [0] * (max_coords - len(cmd.coordinates))
#         coord_tensors.append(torch.tensor(padded_coords, dtype=torch.float32))

#     command_tensor = torch.stack(command_tensors)
#     coord_tensor = torch.stack(coord_tensors)
#     sequence_tensor = torch.cat([command_tensor, coord_tensor], dim=1)

#     # 3. Define a bounding box
#     box = torch.tensor([50, 50, 250, 350], dtype=torch.float32)  # x1, y1, x2, y2

#     # 4. Normalize to mask space
#     normalized_sequence = RelativePolarCommand.image_space_to_mask_space(
#         sequence_tensor, box
#     )

#     # 5. Denormalize back to image space
#     denormalized_sequence = RelativePolarCommand.mask_space_to_image_space(
#         normalized_sequence, box
#     )

#     # 6. Compare
#     assert torch.allclose(
#         sequence_tensor, denormalized_sequence, atol=1e-4
#     ), "Roundtrip failed: tensors do not match"

#     print("\n--- RelativePolarCommand space conversion roundtrip test passed! ---")


# def test_relative_polar_tangent_curve_roundtrip():
#     """
#     Tests that a curve with handles aligned to the tangent is correctly
#     encoded and decoded with N_TANGENT.
#     """
#     nodes = [
#         Node(
#             np.array([0, 0]), out_handle=np.array([50, 15]), contour=None
#         ),  # not aligned
#         Node(
#             np.array([100, 100]),
#             in_handle=np.array([50, 100]),
#             out_handle=np.array([150, 100]),
#             contour=None,
#         ),
#         Node(
#             np.array([200, 0]), in_handle=np.array([150, 15]), contour=None
#         ),  # not aligned
#     ]
#     original_contour = NodeContour(nodes)
#     for node in nodes:
#         node._contour = original_contour
#     commands = RelativePolarCommand.emit(original_contour.nodes)
#     # We expect SOS, M, then a curve; depending on chord vs tangent, general or tangent form
#     assert commands[3].command in ("N_TANGENT", "N_POLAR")

#     # 4. Decode it
#     decoded_contour = RelativePolarCommand.contour_from_commands(commands)
#     assert len(original_contour.nodes) == len(decoded_contour.nodes)
#     for i, original_node in enumerate(original_contour.nodes):
#         decoded_node = decoded_contour.nodes[i]
#         assert np.allclose(
#             original_node.coordinates, decoded_node.coordinates, atol=1e-4
#         )
#         if original_node.out_handle is not None:
#             assert np.allclose(
#                 original_node.out_handle, decoded_node.out_handle, atol=1e-4
#             )
#         if original_node.in_handle is not None:
#             assert np.allclose(
#                 original_node.in_handle, decoded_node.in_handle, atol=1e-4
#             )

#     # 5. Unroll it
#     # Convert to a tensor
#     command_tensors = []
#     coord_tensors = []
#     max_coords = RelativePolarCommand.coordinate_width
#     for cmd in commands:
#         command_tensors.append(RelativePolarCommand.encode_command_one_hot(cmd.command))
#         padded_coords = cmd.coordinates + [0] * (max_coords - len(cmd.coordinates))
#         coord_tensors.append(torch.tensor(padded_coords, dtype=torch.float32))

#     sequence_tensor = torch.cat(
#         [torch.stack(command_tensors), torch.stack(coord_tensors)], dim=1
#     )

#     unrolled_sequence = RelativePolarCommand.unroll_relative_coordinates(
#         sequence_tensor
#     )
#     _, abs_coords = RelativePolarCommand.split_tensor(unrolled_sequence)

#     assert np.allclose(abs_coords[3, 0:2], [100, 100])  # pos
#     assert np.allclose(abs_coords[3, 2:4], [50, 100])  # in handle
#     assert np.allclose(abs_coords[3, 4:6], [150, 100])  # out handle


# def test_relative_polar_nci_tangent_roundtrip():
#     """
#     Tests that a curve with a tangent-aligned in-handle is correctly
#     encoded and decoded with NCI_TANGENT.
#     """
#     nodes = [
#         Node(np.array([0, 0]), contour=None),
#         Node(np.array([100, 0]), in_handle=np.array([50, 0]), contour=None),
#     ]
#     original_contour = NodeContour(nodes)
#     for node in nodes:
#         node._contour = original_contour
#     commands = RelativePolarCommand.emit(original_contour.nodes)
#     assert commands[3].command in ("NCI_TANGENT", "N_POLAR")

#     decoded_contour = RelativePolarCommand.contour_from_commands(commands)
#     assert len(original_contour.nodes) == len(decoded_contour.nodes)
#     for i, original_node in enumerate(original_contour.nodes):
#         decoded_node = decoded_contour.nodes[i]
#         assert np.allclose(
#             original_node.coordinates, decoded_node.coordinates, atol=1e-4
#         )
#         if original_node.out_handle is not None:
#             assert np.allclose(
#                 original_node.out_handle, decoded_node.out_handle, atol=1e-4
#             )
#         if original_node.in_handle is not None:
#             assert np.allclose(
#                 original_node.in_handle, decoded_node.in_handle, atol=1e-4
#             )


# def test_relative_polar_nco_tangent_roundtrip():
#     """
#     Tests that a curve with a tangent-aligned out-handle is correctly
#     encoded and decoded with NCO_TANGENT.
#     """
#     nodes = [
#         Node(np.array([0, 0]), out_handle=np.array([50, 0]), contour=None),
#         Node(np.array([100, 0]), contour=None),
#     ]
#     original_contour = NodeContour(nodes)
#     for node in nodes:
#         node._contour = original_contour
#     commands = RelativePolarCommand.emit(original_contour.nodes)
#     assert commands[3].command in ("NCO_TANGENT", "N_POLAR")
#     decoded_contour = RelativePolarCommand.contour_from_commands(commands)
#     assert len(original_contour.nodes) == len(decoded_contour.nodes)
#     for i, original_node in enumerate(original_contour.nodes):
#         decoded_node = decoded_contour.nodes[i]
#         assert np.allclose(
#             original_node.coordinates, decoded_node.coordinates, atol=1e-4
#         )
#         if original_node.out_handle is not None:
#             assert np.allclose(
#                 original_node.out_handle, decoded_node.out_handle, atol=1e-4
#             )
#         if original_node.in_handle is not None:
#             assert np.allclose(
#                 original_node.in_handle, decoded_node.in_handle, atol=1e-4
#             )


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
#     contours_commands = nodeglyph_orig.command_lists(RelativePolarCommand)

#     # 2b. List[List[NodeCommand]] -> NodeGlyph
#     nodeglyph_reconstructed = NodeGlyph(
#         [
#             RelativePolarCommand.contour_from_commands(contour, tolerant=False)
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
#     encoded_sequences = nodeglyph_orig.encode(RelativePolarCommand)

#     # This can happen for blank glyphs like 'space'
#     if encoded_sequences is None:
#         assert svg_orig_str == ""
#         return

#     # 3b. Encoded sequence -> NodeGlyph
#     nodeglyph_roundtrip = NodeGlyph.decode(encoded_sequences, RelativePolarCommand)

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
