import tensorflow as tf
from .glyph import NodeCommand
from .hyperparameters import BATCH_SIZE, GEN_IMAGE_SIZE, MAX_COMMANDS
import pydiffvg_tensorflow as pydiffvg
import numpy as np

# There's no Metal operator for pydiffvg, and anyway running it on
# GPU is actually unstable and sometimes produces all zeroes
# (https://github.com/BachiLi/diffvg/issues/96) so we use CPU.
pydiffvg.set_use_gpu(False)


@tf.function
def simplify_nodes(cmd, coord):
    """Converts optimized node commands into simpler L and N commands."""
    command_tensor = tf.argmax(cmd, axis=-1)
    command_keys = list(NodeCommand.grammar.keys())

    def get_cmd_code(c):
        return NodeCommand.encode_command(c)

    def update_cmd_coord(cmd, coord, indices, new_cmd_str, new_coord):
        new_cmd = get_cmd_code(new_cmd_str)
        new_cmd_one_hot = tf.one_hot(new_cmd, depth=tf.shape(cmd)[-1], dtype=cmd.dtype)
        new_cmds = tf.tile(
            tf.expand_dims(new_cmd_one_hot, 0), [tf.shape(indices)[0], 1]
        )
        cmd = tf.tensor_scatter_nd_update(cmd, indices, new_cmds)
        coord = tf.tensor_scatter_nd_update(coord, indices, new_coord)
        return cmd, coord

    # Find indices of all commands
    nh_indices = tf.cast(
        tf.where(tf.equal(command_tensor, get_cmd_code("NH"))), tf.int32
    )
    nv_indices = tf.cast(
        tf.where(tf.equal(command_tensor, get_cmd_code("NV"))), tf.int32
    )
    nci_indices = tf.cast(
        tf.where(tf.equal(command_tensor, get_cmd_code("NCI"))), tf.int32
    )
    nco_indices = tf.cast(
        tf.where(tf.equal(command_tensor, get_cmd_code("NCO"))), tf.int32
    )

    # Convert NH command to N
    if tf.size(nh_indices) > 0:
        nh_coords = tf.gather_nd(coord, nh_indices)
        new_nh_coords = tf.stack(
            [
                nh_coords[:, 0],
                nh_coords[:, 1],
                nh_coords[:, 2],
                tf.zeros_like(nh_coords[:, 1]),
                nh_coords[:, 3],
                tf.zeros_like(nh_coords[:, 1]),
            ],
            axis=1,
        )
        cmd, coord = update_cmd_coord(cmd, coord, nh_indices, "N", new_nh_coords)

    # Convert NV command to N
    if tf.size(nv_indices) > 0:
        nv_coords = tf.gather_nd(coord, nv_indices)
        new_nv_coords = tf.stack(
            [
                nv_coords[:, 0],
                nv_coords[:, 1],
                tf.zeros_like(nv_coords[:, 0]),
                nv_coords[:, 2],
                tf.zeros_like(nv_coords[:, 0]),
                nv_coords[:, 3],
            ],
            axis=1,
        )
        cmd, coord = update_cmd_coord(cmd, coord, nv_indices, "N", new_nv_coords)

    # Convert NCI command to N
    if tf.size(nci_indices) > 0:
        nci_coords = tf.gather_nd(coord, nci_indices)
        new_nci_coords = tf.stack(
            [
                nci_coords[:, 0],
                nci_coords[:, 1],
                nci_coords[:, 2],
                nci_coords[:, 3],
                tf.zeros_like(nci_coords[:, 0]),
                tf.zeros_like(nci_coords[:, 1]),
            ],
            axis=1,
        )
        cmd, coord = update_cmd_coord(cmd, coord, nci_indices, "N", new_nci_coords)

    # Convert NCO command to N
    if tf.size(nco_indices) > 0:
        nco_coords = tf.gather_nd(coord, nco_indices)
        new_nco_coords = tf.stack(
            [
                nco_coords[:, 0],
                nco_coords[:, 1],
                tf.zeros_like(nco_coords[:, 0]),
                tf.zeros_like(nco_coords[:, 1]),
                nco_coords[:, 2],
                nco_coords[:, 3],
            ],
            axis=1,
        )
        cmd, coord = update_cmd_coord(cmd, coord, nco_indices, "N", new_nco_coords)

    return cmd, coord


def nodes_to_segments(cmd, coord):
    cmd, coord = simplify_nodes(cmd, coord)
    command_tensor = np.argmax(cmd, axis=-1)

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
            # finalize_contour_fn
            p1_cmd = command_tensor[i - 1]
            p1_coord = coord[i - 1]
            p2_cmd = command_tensor[contour_start_index]
            p2_coord = coord[contour_start_index]

            is_curve = (p1_cmd == cmd_n_val) and (p2_cmd == cmd_n_val)

            p1_pos, p2_pos = p1_coord[0:2], p2_coord[0:2]
            p1_hout, p2_hin = p1_pos + p1_coord[4:6], p2_pos + p2_coord[2:4]

            if is_curve:
                all_points.append(p1_hout)
                all_points.append(p2_hin)
                all_points.append(p2_pos)
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
                # new_contour_fn
                all_points.append(coord[i, 0:2])
                contour_start_index = i
            else:
                # existing_contour_fn
                p1_cmd = command_tensor[i - 1]
                p1_coord = coord[i - 1]
                p2_cmd = command
                p2_coord = coord[i]

                is_curve = (p1_cmd == cmd_n_val) and (p2_cmd == cmd_n_val)

                p1_pos, p2_pos = p1_coord[0:2], p2_coord[0:2]
                p1_hout, p2_hin = p1_pos + p1_coord[4:6], p2_pos + p2_coord[2:4]

                if is_curve:
                    all_points.append(p1_hout)
                    all_points.append(p2_hin)
                    all_points.append(p2_pos)
                    all_num_cp.append(2)
                else:
                    all_points.append(p2_pos)
                    all_num_cp.append(0)

    if not all_points:
        return (
            np.array([], dtype=np.float32).reshape(0, 2),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        )

    return (
        np.array(all_points, dtype=np.float32),
        np.array(all_num_cp, dtype=np.int32),
        np.array(contour_splits, dtype=np.int32),
        np.array(point_splits, dtype=np.int32),
    )


def rasterize_batch(cmds, coords):
    """Render a batch of glyphs from their node representation."""

    def _rasterize_single_glyph_py(cmd_np, coord_np):
        # This function is wrapped in tf.py_function, so it gets numpy arrays.
        print("Rasterizing glyph")
        points, num_control_points, num_cp_splits, point_splits = nodes_to_segments(
            cmd_np, coord_np
        )
        print(points, num_control_points)

        shapes = []
        num_cp_start = 0
        point_start = 0
        for num_cp_end, point_end in zip(num_cp_splits, point_splits):
            num_cp = num_control_points[num_cp_start:num_cp_end]
            path_points = points[point_start:point_end]
            # tf.print("Num control points:", num_cp)
            # tf.print("Path points:", path_points)
            # pydiffvg expects tensors, so we convert back
            path = pydiffvg.Path(
                num_control_points=tf.convert_to_tensor(num_cp),
                points=tf.convert_to_tensor(path_points),
                is_closed=True,
            )
            shapes.append(path)
            num_cp_start = num_cp_end
            point_start = point_end

        if len(shapes) == 0:
            return np.ones((GEN_IMAGE_SIZE[0], GEN_IMAGE_SIZE[1], 1), dtype=np.float32)

        # Please don't change these. They are correct.
        scale_factor = GEN_IMAGE_SIZE[0] / 1457
        shape_to_canvas = tf.constant(
            [[scale_factor, 0.0, 4.0], [0.0, -scale_factor, 351.0], [0.0, 0.0, 1.0]],
            dtype=tf.float32,
        )

        path_group = pydiffvg.ShapeGroup(
            shape_ids=tf.range(len(shapes)),
            fill_color=tf.constant([0.0, 0.0, 0.0, 1.0]),
            use_even_odd_rule=True,
            shape_to_canvas=shape_to_canvas,
        )
        shape_groups = [path_group]
        scene_args = pydiffvg.serialize_scene(
            GEN_IMAGE_SIZE[0], GEN_IMAGE_SIZE[1], shapes, shape_groups
        )

        render = pydiffvg.render
        print("Rendering", shapes)
        img = render(
            tf.constant(GEN_IMAGE_SIZE[0]),  # width
            tf.constant(GEN_IMAGE_SIZE[1]),  # height
            tf.constant(4),  # num_samples_x
            tf.constant(4),  # num_samples_y
            tf.constant(1),  # seed
            *scene_args,
        )
        print("Done")
        # Invert it, we want black on white
        img = tf.reduce_max(img) - img
        # Pydiffvg outputs RGBA, we just want the alpha channel
        img_numpy = tf.expand_dims(img[:, :, 3], -1).numpy()
        return img_numpy.astype(np.float32)

    images = tf.map_fn(
        lambda elems: tf.py_function(
            _rasterize_single_glyph_py, [elems[0], elems[1]], tf.float32
        ),
        (cmds, coords),
        fn_output_signature=tf.TensorSpec(
            shape=[GEN_IMAGE_SIZE[0], GEN_IMAGE_SIZE[1], 1], dtype=tf.float32
        ),
    )
    images.set_shape([None, GEN_IMAGE_SIZE[0], GEN_IMAGE_SIZE[1], 1])
    return images
