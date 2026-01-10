import torch
from glyphogen.representations.nodecommand import NodeCommand


def test_relative_coordinate_normalization():
    """
    Tests that relative coordinates are only scaled, not translated,
    during normalization.
    """
    # Bbox is offset from the origin to make translation effects obvious
    box = torch.tensor([100, 100, 300, 300], dtype=torch.float32)  # 200x200 box

    # M command is absolute, L command is relative
    m_cmd = torch.nn.functional.one_hot(
        torch.tensor(NodeCommand.encode_command("M")),
        num_classes=NodeCommand.command_width,
    )
    l_cmd = torch.nn.functional.one_hot(
        torch.tensor(NodeCommand.encode_command("L")),
        num_classes=NodeCommand.command_width,
    )

    # M to (150, 150) in image space. L is a (0,0) relative move.
    coords = torch.zeros(2, NodeCommand.coordinate_width)
    coords[0, 0:2] = torch.tensor([150.0, 150.0])
    coords[1, 0:2] = torch.tensor([0.0, 0.0])

    commands = torch.stack([m_cmd, l_cmd]).float()
    sequence_img_space = torch.cat([commands, coords], dim=1)

    # Normalize to mask space
    sequence_mask_space = NodeCommand.image_space_to_mask_space(sequence_img_space, box)
    _, coords_norm = NodeCommand.split_tensor(sequence_mask_space)

    # --- Asserts for image_space_to_mask_space ---

    # The M coord (150, 150) is at (50, 50) inside the 200x200 box starting at (100, 100).
    # Normalized to [0,1] this is (50/200, 50/200) = (0.25, 0.25).
    # Shifted to [-1,1] this is (0.25*2 - 1) = -0.5.
    expected_m_norm = torch.tensor([-0.5, -0.5])
    assert torch.allclose(coords_norm[0, 0:2], expected_m_norm, atol=1e-6)

    # The L coord is a (0,0) delta. It should only be scaled, resulting in (0,0).
    expected_l_norm = torch.tensor([0.0, 0.0])
    assert torch.allclose(coords_norm[1, 0:2], expected_l_norm, atol=1e-6)

    # --- Asserts for mask_space_to_image_space (round trip) ---

    # Denormalize back to image space
    sequence_img_space_roundtrip = NodeCommand.mask_space_to_image_space(
        sequence_mask_space, box
    )
    _, coords_roundtrip = NodeCommand.split_tensor(sequence_img_space_roundtrip)

    # Check if the round-tripped coordinates match the original
    assert torch.allclose(coords_roundtrip, coords, atol=1e-6)
