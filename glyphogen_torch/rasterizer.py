import torch
from .command_defs import NodeCommand
from .hyperparameters import GEN_IMAGE_SIZE
import pydiffvg
import numpy as np

# There's no Metal operator for pydiffvg, and anyway running it on
# GPU is actually unstable and sometimes produces all zeroes
# (https://github.com/BachiLi/diffvg/issues/96) so we use CPU.
pydiffvg.set_use_gpu(False)

command_keys = list(NodeCommand.grammar.keys())
cmd_n_val = command_keys.index("N")
cmd_z_val = command_keys.index("Z")
cmd_eos_val = command_keys.index("EOS")
cmd_sos_val = command_keys.index("SOS")


@torch.compiler.disable()
def simplify_nodes(cmd, coord):
    """Converts optimized node commands into simpler L and N commands."""
    command_tensor = torch.argmax(cmd, dim=-1)

    def get_cmd_code(c):
        return NodeCommand.encode_command(c)

    is_nh = (command_tensor == get_cmd_code("NH")).unsqueeze(-1)
    is_nv = (command_tensor == get_cmd_code("NV")).unsqueeze(-1)
    is_nci = (command_tensor == get_cmd_code("NCI")).unsqueeze(-1)
    is_nco = (command_tensor == get_cmd_code("NCO")).unsqueeze(-1)

    is_simplified_to_n = is_nh | is_nv | is_nci | is_nco

    n_cmd_code = get_cmd_code("N")
    n_cmd_one_hot = torch.nn.functional.one_hot(
        torch.tensor(n_cmd_code, device=cmd.device), num_classes=cmd.shape[-1]
    ).to(cmd.dtype)

    new_cmd = torch.where(is_simplified_to_n, n_cmd_one_hot, cmd)

    coord_for_nh = torch.stack(
        [
            coord[..., 0],
            coord[..., 1],
            coord[..., 2],
            torch.zeros_like(coord[..., 3]),
            coord[..., 3],
            torch.zeros_like(coord[..., 5]),
        ],
        dim=-1,
    )
    coord_for_nv = torch.stack(
        [
            coord[..., 0],
            coord[..., 1],
            torch.zeros_like(coord[..., 2]),
            coord[..., 2],
            torch.zeros_like(coord[..., 4]),
            coord[..., 3],
        ],
        dim=-1,
    )
    coord_for_nci = torch.stack(
        [
            coord[..., 0],
            coord[..., 1],
            coord[..., 2],
            coord[..., 3],
            torch.zeros_like(coord[..., 4]),
            torch.zeros_like(coord[..., 5]),
        ],
        dim=-1,
    )
    coord_for_nco = torch.stack(
        [
            coord[..., 0],
            coord[..., 1],
            torch.zeros_like(coord[..., 2]),
            torch.zeros_like(coord[..., 3]),
            coord[..., 2],
            coord[..., 3],
        ],
        dim=-1,
    )

    new_coord = torch.where(is_nh, coord_for_nh, coord)
    new_coord = torch.where(is_nv, coord_for_nv, new_coord)
    new_coord = torch.where(is_nci, coord_for_nci, new_coord)
    new_coord = torch.where(is_nco, coord_for_nco, new_coord)

    return new_cmd, new_coord


@torch.compiler.disable()
def nodes_to_segments(cmd, coord):
    cmd, coord = simplify_nodes(cmd, coord)
    command_tensor = torch.argmax(cmd, dim=-1)

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
            p1_cmd, p1_coord = command_tensor[i - 1], coord[i - 1]
            p2_cmd, p2_coord = (
                command_tensor[contour_start_index],
                coord[contour_start_index],
            )
            is_curve = (p1_cmd == cmd_n_val) and (p2_cmd == cmd_n_val)
            p1_pos, p2_pos = p1_coord[0:2], p2_coord[0:2]
            p1_hout, p2_hin = p1_pos + p1_coord[4:6], p2_pos + p2_coord[2:4]

            if is_curve:
                all_points.extend([p1_hout, p2_hin, p2_pos])
                all_num_cp.extend([2])
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
                all_points.append(coord[i, 0:2])
                contour_start_index = i
            else:
                p1_cmd, p1_coord = command_tensor[i - 1], coord[i - 1]
                p2_cmd, p2_coord = command, coord[i]
                is_curve = (p1_cmd == cmd_n_val) and (p2_cmd == cmd_n_val)
                p1_pos, p2_pos = p1_coord[0:2], p2_coord[0:2]
                p1_hout, p2_hin = p1_pos + p1_coord[4:6], p2_pos + p2_coord[2:4]

                if is_curve:
                    all_points.extend([p1_hout, p2_hin, p2_pos])
                    all_num_cp.extend([2])
                else:
                    all_points.append(p2_pos)
                    all_num_cp.append(0)

    if not all_points:
        return (
            torch.empty(0, 2, dtype=torch.float32, device=coord.device),
            torch.empty(0, dtype=torch.int32, device=coord.device),
            [],
            [],
        )

    return (
        torch.stack(all_points),
        torch.tensor(all_num_cp, dtype=torch.int32, device=coord.device),
        contour_splits,
        point_splits,
    )


@torch.compiler.disable(recursive=False)
def rasterize_batch(cmds, coords, seed=42, img_size=None, requires_grad=True):
    """Render a batch of glyphs from their node representation."""
    if img_size is None:
        img_size = GEN_IMAGE_SIZE[0]
    coords.requires_grad_(requires_grad)
    dead_image = torch.ones(1, img_size, img_size, dtype=torch.float32)
    images = []
    for i in range(cmds.shape[0]):
        # If there's no EOS token or no Z token, don't bother
        if cmd_eos_val not in torch.argmax(
            cmds[i], axis=-1
        ) or cmd_z_val not in torch.argmax(cmds[i], axis=-1):
            images.append(dead_image)
            continue
        points, num_control_points, num_cp_splits, point_splits = nodes_to_segments(
            cmds[i].clone(), coords[i].clone()
        )

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
            images.append(dead_image)
            continue

        scale_factor = img_size / 1000.0
        shape_to_canvas = torch.tensor(
            [
                [scale_factor, 0.0, 0.0],
                [0.0, scale_factor, 0.33 * img_size],
                [0.0, 0.0, 1.0],
            ]
        )

        path_group = pydiffvg.ShapeGroup(
            shape_ids=torch.arange(len(shapes)),
            fill_color=torch.tensor([0.0, 0.0, 0.0, 1.0]),
            use_even_odd_rule=True,
            shape_to_canvas=shape_to_canvas,
        )
        shape_groups = [path_group]
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            img_size, img_size, shapes, shape_groups
        )

        render = pydiffvg.RenderFunction.apply
        try:
            img = render(img_size, img_size, 2, 2, seed, None, *scene_args)
        except Exception as e:
            print(f"Failed to rasterize: {e}")
            images.append(dead_image)
            continue

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
