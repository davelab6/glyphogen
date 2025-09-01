import torch
from .glyph import NodeCommand
from .hyperparameters import GEN_IMAGE_SIZE
import pydiffvg
import numpy as np

# There's no Metal operator for pydiffvg, and anyway running it on
# GPU is actually unstable and sometimes produces all zeroes
# (https://github.com/BachiLi/diffvg/issues/96) so we use CPU.
pydiffvg.set_use_gpu(False)


@torch.compile
def simplify_nodes(cmd, coord):
    """Converts optimized node commands into simpler L and N commands."""
    command_tensor = torch.argmax(cmd, axis=-1)
    command_keys = list(NodeCommand.grammar.keys())

    def get_cmd_code(c):
        return NodeCommand.encode_command(c)

    def update_cmd_coord(cmd, coord, indices, new_cmd_str, new_coord):
        new_cmd = get_cmd_code(new_cmd_str)
        new_cmd_one_hot = (
            torch.nn.functional.one_hot(
                torch.tensor(new_cmd), num_classes=cmd.shape[-1]
            )
            .to(cmd.dtype)
            .to(cmd.device)
        )
        new_cmds = new_cmd_one_hot.unsqueeze(0).repeat(indices.shape[0], 1)
        indices = indices.squeeze(dim=1).to(cmd.device)
        cmd[indices] = new_cmds
        coord[indices] = new_coord
        return cmd, coord

    # Find indices of all commands
    nh_indices = (command_tensor == get_cmd_code("NH")).nonzero()
    nv_indices = (command_tensor == get_cmd_code("NV")).nonzero()
    nci_indices = (command_tensor == get_cmd_code("NCI")).nonzero()
    nco_indices = (command_tensor == get_cmd_code("NCO")).nonzero()

    # Convert NH command to N
    if nh_indices.numel() > 0:
        nh_coords = coord[nh_indices.squeeze(dim=1)]
        new_nh_coords = torch.stack(
            [
                nh_coords[:, 0],
                nh_coords[:, 1],
                nh_coords[:, 2],
                torch.zeros_like(nh_coords[:, 1]),
                nh_coords[:, 3],
                torch.zeros_like(nh_coords[:, 1]),
            ],
            axis=1,
        )
        cmd, coord = update_cmd_coord(cmd, coord, nh_indices, "N", new_nh_coords)

    # Convert NV command to N
    if nv_indices.numel() > 0:
        nv_coords = coord[nv_indices.squeeze(dim=1)]
        new_nv_coords = torch.stack(
            [
                nv_coords[:, 0],
                nv_coords[:, 1],
                torch.zeros_like(nv_coords[:, 0]),
                nv_coords[:, 2],
                torch.zeros_like(nv_coords[:, 0]),
                nv_coords[:, 3],
            ],
            axis=1,
        )
        cmd, coord = update_cmd_coord(cmd, coord, nv_indices, "N", new_nv_coords)

    # Convert NCI command to N
    if nci_indices.numel() > 0:
        nci_coords = coord[nci_indices.squeeze(dim=1)]
        new_nci_coords = torch.stack(
            [
                nci_coords[:, 0],
                nci_coords[:, 1],
                nci_coords[:, 2],
                nci_coords[:, 3],
                torch.zeros_like(nci_coords[:, 0]),
                torch.zeros_like(nci_coords[:, 1]),
            ],
            axis=1,
        )
        cmd, coord = update_cmd_coord(cmd, coord, nci_indices, "N", new_nci_coords)

    # Convert NCO command to N
    if nco_indices.numel() > 0:
        nco_coords = coord[nco_indices.squeeze(dim=1)]
        new_nco_coords = torch.stack(
            [
                nco_coords[:, 0],
                nco_coords[:, 1],
                torch.zeros_like(nco_coords[:, 0]),
                torch.zeros_like(nco_coords[:, 1]),
                nco_coords[:, 2],
                nco_coords[:, 3],
            ],
            axis=1,
        )
        cmd, coord = update_cmd_coord(cmd, coord, nco_indices, "N", new_nco_coords)

    return cmd, coord


@torch.compile
def nodes_to_segments(cmd, coord):
    cmd, coord = simplify_nodes(cmd, coord)
    command_tensor = np.argmax(cmd.detach().cpu().numpy(), axis=-1)
    coord_np = coord.detach().cpu().numpy()

    command_keys = list(NodeCommand.grammar.keys())
    cmd_n_val = command_keys.index("N")
    cmd_z_val = command_keys.index("Z")
    cmd_eos_val = command_keys.index("EOS")
    cmd_sos_val = command_keys.index("SOS")

    all_points = []
    all_num_cp = []
    contour_splits = []
    point_splits = []

    contour_start_index = -1

    for i in range(len(command_tensor)):
        command = command_tensor[i]
        is_sos = command == cmd_sos_val
        is_z = command == cmd_z_val
        is_eos = command == cmd_eos_val
        is_contour_boundary = is_z or is_eos

        if contour_start_index > -1 and is_contour_boundary:
            p1_cmd, p1_coord = command_tensor[i - 1], coord_np[i - 1]
            p2_cmd, p2_coord = (
                command_tensor[contour_start_index],
                coord_np[contour_start_index],
            )
            is_curve = (p1_cmd == cmd_n_val) and (p2_cmd == cmd_n_val)
            p1_pos, p2_pos = p1_coord[0:2], p2_coord[0:2]
            p1_hout, p2_hin = p1_pos + p1_coord[4:6], p2_pos + p2_coord[2:4]

            if is_curve:
                all_points.extend([p1_hout, p2_hin, p2_pos])
                all_num_cp.append(2)
            else:
                all_points.append(p2_pos)
                all_num_cp.append(0)

            contour_splits.append(len(all_num_cp))
            point_splits.append(len(all_points))
            contour_start_index = -1

        if is_eos:
            break

        is_node = not (is_sos or is_z or is_eos)

        if is_node:
            if contour_start_index == -1:
                all_points.append(coord_np[i, 0:2])
                contour_start_index = i
            else:
                p1_cmd, p1_coord = command_tensor[i - 1], coord_np[i - 1]
                p2_cmd, p2_coord = command, coord_np[i]
                is_curve = (p1_cmd == cmd_n_val) and (p2_cmd == cmd_n_val)
                p1_pos, p2_pos = p1_coord[0:2], p2_coord[0:2]
                p1_hout, p2_hin = p1_pos + p1_coord[4:6], p2_pos + p2_coord[2:4]

                if is_curve:
                    all_points.extend([p1_hout, p2_hin, p2_pos])
                    all_num_cp.append(2)
                else:
                    all_points.append(p2_pos)
                    all_num_cp.append(0)

    if not all_points:
        return (
            torch.empty(0, 2, dtype=torch.float32),
            torch.empty(0, dtype=torch.int32),
            [],
            [],
        )

    return (
        torch.from_numpy(np.array(all_points, dtype=np.float32)),
        torch.from_numpy(np.array(all_num_cp, dtype=np.int32)),
        contour_splits,
        point_splits,
    )


def rasterize_batch(cmds, coords):
    """Render a batch of glyphs from their node representation."""
    images = []
    for i in range(cmds.shape[0]):
        points, num_control_points, num_cp_splits, point_splits = nodes_to_segments(
            cmds[i].clone(), coords[i].clone()
        )

        if points.shape[0] == 0:
            img = torch.ones(
                1, GEN_IMAGE_SIZE[0], GEN_IMAGE_SIZE[1], dtype=torch.float32
            )
            images.append(img)
            continue

        shapes = []
        num_cp_start = 0
        point_start = 0
        for num_cp_end, point_end in zip(num_cp_splits, point_splits):
            num_cp = num_control_points[num_cp_start:num_cp_end]
            path_points = points[point_start:point_end]
            path = pydiffvg.Path(
                num_control_points=num_cp.to(torch.int32).cpu(),
                points=path_points.cpu(),
                is_closed=True,
            )
            shapes.append(path)
            num_cp_start = num_cp_end
            point_start = point_end

        # If there are no shapes, return a blank image
        if len(shapes) == 0:
            img = torch.ones(
                1, GEN_IMAGE_SIZE[0], GEN_IMAGE_SIZE[1], dtype=torch.float32
            )
            images.append(img)
            continue

        scale_factor = GEN_IMAGE_SIZE[0] / 1457
        shape_to_canvas = torch.tensor(
            [[scale_factor, 0.0, 4.0], [0.0, -scale_factor, 351.0], [0.0, 0.0, 1.0]]
        )

        path_group = pydiffvg.ShapeGroup(
            shape_ids=torch.arange(len(shapes)),
            fill_color=torch.tensor([0.0, 0.0, 0.0, 1.0]),
            use_even_odd_rule=True,
            shape_to_canvas=shape_to_canvas,
        )
        shape_groups = [path_group]
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            GEN_IMAGE_SIZE[0], GEN_IMAGE_SIZE[1], shapes, shape_groups
        )

        render = pydiffvg.RenderFunction.apply
        try:
            img = render(
                GEN_IMAGE_SIZE[0], GEN_IMAGE_SIZE[1], 4, 4, 1, None, *scene_args
            )
        except Exception as e:
            print(f"Rendering failed ({e})")
            img = torch.zeros(
                GEN_IMAGE_SIZE[0], GEN_IMAGE_SIZE[1], 4, dtype=torch.float32
            )

        img = torch.max(img) - img
        # Get alpha channel and add channel dimension
        img = img[:, :, 3].unsqueeze(0)
        images.append(img)

    return torch.stack(images)


def rasterize_and_save(cmds, coords, filename="produced.png"):
    """Render a single glyph and save it to a file."""
    cmds = cmds.unsqueeze(0)
    coords = coords.unsqueeze(0)
    images = rasterize_batch(cmds, coords)
    # imwrite expects H, W, C
    pydiffvg.imwrite(images[0].permute(1, 2, 0).cpu(), filename, gamma=1.0)
