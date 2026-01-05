import torch
import numpy as np
from glyphogen.command_defs import NodeCommand


def test_unroll_relative_coordinates():
    # Create a sample sequence tensor
    # SOS
    # M 10, 20
    # L 5, 5  (abs: 15, 25)
    # LH 10 (abs: 25, 25)
    # LV -5 (abs: 25, 20)
    # N 1,1, 2,2, 3,3 (abs pos: 26, 21, abs in handle: 28, 23, abs out handle: 29, 24)
    # NS 2,2, pi/4, 10, 20 (abs pos: 28, 23)
    # EOS
    def cmd(s):
        return torch.nn.functional.one_hot(
            torch.tensor(NodeCommand.encode_command(s)),
            num_classes=NodeCommand.command_width,
        )

    def pad(ary):
        return ary + [0.0] * (NodeCommand.coordinate_width - len(ary))

    coord_list = [
        pad([]),
        pad([10.0, 20.0]),
        pad([5.0, 5.0]),
        pad([10.0]),
        pad([-5.0]),
        pad([1.0, 1.0, 2.0, 2.0, 3.0, 3.0]),
        pad([2.0, 2.0, np.pi / 4, 10.0, 20.0]),
        pad([]),
    ]
    coords = torch.tensor(coord_list, requires_grad=True)

    commands = torch.stack(
        [
            cmd("SOS"),
            cmd("M"),
            cmd("L"),
            cmd("LH"),
            cmd("LV"),
            cmd("N"),
            cmd("NS"),
            cmd("EOS"),
        ]
    ).float()
    sequence = torch.cat([commands, coords], dim=1)

    # This test uses an un-normalized sequence, so we call unroll directly.
    # The sequence from the model would first be denormalized by mask_space_to_image_space
    absolute_sequence = NodeCommand.unroll_relative_coordinates(sequence)
    _, absolute_coords = NodeCommand.split_tensor(absolute_sequence)

    expected_coords = torch.zeros(8, NodeCommand.coordinate_width)
    expected_coords[1, 0:2] = torch.tensor([10.0, 20.0])
    expected_coords[2, 0:2] = torch.tensor([15.0, 25.0])
    expected_coords[3, 0:2] = torch.tensor([25.0, 25.0])
    expected_coords[4, 0:2] = torch.tensor([25.0, 20.0])
    expected_coords[5, 0:6] = torch.tensor([26.0, 21.0, 28.0, 23.0, 29.0, 24.0])
    # NS node
    ns_pos = torch.tensor([28.0, 23.0])
    angle = np.pi / 4
    len_in = 10.0
    len_out = 20.0
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    in_handle = ns_pos - torch.tensor([len_in * cos_a, len_in * sin_a])
    out_handle = ns_pos + torch.tensor([len_out * cos_a, len_out * sin_a])
    expected_coords[6, 0:6] = torch.cat([ns_pos, in_handle, out_handle])

    assert torch.allclose(absolute_coords, expected_coords, atol=1e-6)

    # Test differentiability
    absolute_coords.sum().backward()
    assert coords.grad is not None

