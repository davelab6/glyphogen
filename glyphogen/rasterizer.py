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


@tf.function
def nodes_to_segments(cmd, coord):
    cmd, coord = simplify_nodes(cmd, coord)
    command_tensor = tf.argmax(cmd, axis=-1, output_type=tf.int32)
    max_len = MAX_COMMANDS

    command_keys = tf.constant(list(NodeCommand.grammar.keys()))
    cmd_n_val = tf.where(tf.equal(command_keys, "N"))[0][0]
    cmd_z_val = tf.where(tf.equal(command_keys, "Z"))[0][0]
    cmd_eos_val = tf.where(tf.equal(command_keys, "EOS"))[0][0]
    cmd_sos_val = tf.where(tf.equal(command_keys, "SOS"))[0][0]

    # A single flat TensorArray for all points/num_cp in the glyph
    all_points = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    all_num_cp = tf.TensorArray(tf.int32, size=0, dynamic_size=True)

    # Store the end indices of each contour
    contour_splits = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
    point_splits = tf.TensorArray(tf.int32, size=0, dynamic_size=True)

    i = tf.constant(0)
    contour_start_index = tf.constant(-1)

    loop_cond = lambda i, csi, ap, anc, cs, ps: i < max_len

    def loop_body(
        i, contour_start_index, all_points, all_num_cp, contour_splits, point_splits
    ):
        command = command_tensor[i]
        is_sos = tf.equal(command, tf.cast(cmd_sos_val, dtype=tf.int32))
        is_z = tf.equal(command, tf.cast(cmd_z_val, dtype=tf.int32))
        is_eos = tf.equal(command, tf.cast(cmd_eos_val, dtype=tf.int32))

        is_contour_boundary = tf.logical_or(is_z, is_eos)

        def finalize_contour_fn():
            """Logic to run when a contour boundary is hit."""
            nonlocal all_points, all_num_cp, contour_splits, point_splits
            p1_cmd = command_tensor[i - 1]
            p1_coord = coord[i - 1]
            p2_cmd = command_tensor[contour_start_index]
            p2_coord = coord[contour_start_index]

            is_curve = tf.logical_and(
                tf.equal(p1_cmd, tf.cast(cmd_n_val, dtype=tf.int32)),
                tf.equal(p2_cmd, tf.cast(cmd_n_val, dtype=tf.int32)),
            )

            p1_pos, p2_pos = p1_coord[0:2], p2_coord[0:2]
            p1_hout, p2_hin = p1_pos + p1_coord[4:6], p2_pos + p2_coord[2:4]

            def curve_fn():
                ap_new = all_points.write(all_points.size(), p1_hout)
                ap_new = ap_new.write(ap_new.size(), p2_hin)
                ap_new = ap_new.write(ap_new.size(), p2_pos)
                anc_new = all_num_cp.write(all_num_cp.size(), 2)
                return ap_new, anc_new

            def line_fn():
                ap_new = all_points.write(all_points.size(), p2_pos)
                anc_new = all_num_cp.write(all_num_cp.size(), 0)
                return ap_new, anc_new

            new_ap, new_anc = tf.cond(is_curve, curve_fn, line_fn)

            new_cs = contour_splits.write(contour_splits.size(), new_anc.size())
            new_ps = point_splits.write(point_splits.size(), new_ap.size())
            new_csi = -1
            return new_ap, new_anc, new_cs, new_ps, new_csi

        def passthrough_fn():
            """Logic to run when no contour boundary is hit."""
            return (
                all_points,
                all_num_cp,
                contour_splits,
                point_splits,
                contour_start_index,
            )

        all_points, all_num_cp, contour_splits, point_splits, contour_start_index = (
            tf.cond(
                tf.logical_and(contour_start_index > -1, is_contour_boundary),
                finalize_contour_fn,
                passthrough_fn,
            )
        )

        is_node = tf.logical_not(tf.logical_or(is_sos, tf.logical_or(is_z, is_eos)))

        def node_logic_fn():
            nonlocal contour_start_index, all_points, all_num_cp

            def new_contour_fn():
                # First node of a new contour
                ap_new = all_points.write(all_points.size(), coord[i, 0:2])
                return i, ap_new, all_num_cp

            def existing_contour_fn():
                # Subsequent node in an existing contour
                p1_cmd = command_tensor[i - 1]
                p1_coord = coord[i - 1]
                p2_cmd = command
                p2_coord = coord[i]

                is_curve = tf.logical_and(
                    tf.equal(p1_cmd, tf.cast(cmd_n_val, dtype=tf.int32)),
                    tf.equal(p2_cmd, tf.cast(cmd_n_val, dtype=tf.int32)),
                )

                p1_pos, p2_pos = p1_coord[0:2], p2_coord[0:2]
                p1_hout, p2_hin = p1_pos + p1_coord[4:6], p2_pos + p2_coord[2:4]

                def curve_segment_fn():
                    ap_new = all_points.write(all_points.size(), p1_hout)
                    ap_new = ap_new.write(ap_new.size(), p2_hin)
                    ap_new = ap_new.write(ap_new.size(), p2_pos)
                    anc_new = all_num_cp.write(all_num_cp.size(), 2)
                    return ap_new, anc_new

                def line_segment_fn():
                    ap_new = all_points.write(all_points.size(), p2_pos)
                    anc_new = all_num_cp.write(all_num_cp.size(), 0)
                    return ap_new, anc_new

                ap_new, anc_new = tf.cond(is_curve, curve_segment_fn, line_segment_fn)
                return contour_start_index, ap_new, anc_new

            csi, ap, anc = tf.cond(
                tf.equal(contour_start_index, -1), new_contour_fn, existing_contour_fn
            )
            return csi, ap, anc

        def non_node_logic_fn():
            return contour_start_index, all_points, all_num_cp

        contour_start_index, all_points, all_num_cp = tf.cond(
            is_node, node_logic_fn, non_node_logic_fn
        )

        return (
            i + 1,
            contour_start_index,
            all_points,
            all_num_cp,
            contour_splits,
            point_splits,
        )

    _, _, all_points, all_num_cp, contour_splits, point_splits = tf.while_loop(
        loop_cond,
        loop_body,
        [i, contour_start_index, all_points, all_num_cp, contour_splits, point_splits],
    )

    return (
        all_points.stack(),
        all_num_cp.stack(),
        contour_splits.stack(),
        point_splits.stack(),
    )


def rasterize_batch(cmds, coords):
    """Render a batch of glyphs from their node representation."""
    images = []
    for i in range(tf.shape(cmds)[0]):
        cmd = cmds[i]
        coord = coords[i]
        tf.print("Processing glyph", i, "with cmd shape:", tf.shape(cmd))
        if tf.shape(cmd)[0] == 0:
            images.append(
                tf.zeros([GEN_IMAGE_SIZE, GEN_IMAGE_SIZE, 1], dtype=tf.float32)
            )
            continue

        points, num_control_points, num_cp_splits, point_splits = nodes_to_segments(
            cmd, coord
        )

        shapes = []
        num_cp_start = 0
        point_start = 0
        for num_cp_end, point_end in zip(num_cp_splits, point_splits):
            num_cp = num_control_points[num_cp_start:num_cp_end]
            path_points = points[point_start:point_end]
            path = pydiffvg.Path(
                num_control_points=num_cp, points=path_points, is_closed=True
            )
            shapes.append(path)
            num_cp_start = num_cp_end
            point_start = point_end

        scale_factor = GEN_IMAGE_SIZE[0] / 1457
        shape_to_canvas = tf.constant(
            [[scale_factor, 0.0, 4.0], [0.0, -scale_factor, 351.0], [0.0, 0.0, 1.0]],
            dtype=tf.float32,
        )

        path_group = pydiffvg.ShapeGroup(
            shape_ids=tf.range(len(shapes)),
            fill_color=tf.constant([0.0, 0.0, 0.0, 1.0]),
            use_even_odd_rule=False,
            shape_to_canvas=shape_to_canvas,
        )
        shape_groups = [path_group]
        scene_args = pydiffvg.serialize_scene(
            GEN_IMAGE_SIZE[0], GEN_IMAGE_SIZE[1], shapes, shape_groups
        )
        render = pydiffvg.render
        img = render(
            tf.constant(GEN_IMAGE_SIZE[0]),  # width
            tf.constant(GEN_IMAGE_SIZE[1]),  # height
            tf.constant(4),  # num_samples_x
            tf.constant(4),  # num_samples_y
            tf.constant(1),  # seed
            *scene_args,
        )
        # Invert it, we want black on white
        img = tf.reduce_max(img) - img
        # Pydiffvg outputs RGBA, we just want the alpha channel
        img = tf.expand_dims(img[:, :, 3], -1)
        images.append(img)
    return tf.stack(images)
