from typing import Dict, List, Sequence, TYPE_CHECKING
import numpy as np
from glyphogen.nodeglyph import Node
from glyphogen.representations import (
    AbsoluteCoordinateRepresentation,
    CommandRepresentation,
    RelativePolarCoordinateRepresentation,
)
from jaxtyping import Float
import torch

if TYPE_CHECKING:
    from glyphogen.nodeglyph import NodeContour


class RelativePolarCommand(CommandRepresentation):
    """
    Turtle-like representation using relative polar motion and local angles.
    """

    grammar = {
        "SOS": 0,
        "M": 2,
        "L_POLAR": 2,  # r, phi
        "L_LEFT_90": 1,  # distance
        "L_RIGHT_90": 1,  # distance
        "N_POLAR": 6,  # r, phi, in_len, in_phi, out_len, out_phi
        "N_POLAR_IN": 4,  # r, phi, in_len, in_phi (out handle absent)
        "N_POLAR_OUT": 4,  # r, phi, out_len, out_phi (in handle absent)
        "N_SMOOTH": 5,  # r, phi, out_phi, len_in, len_out
        "EOS": 0,
    }
    coordinate_representation = RelativePolarCoordinateRepresentation

    @staticmethod
    def _unit(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        if n > 1e-6:
            return v / n
        return fallback

    @classmethod
    def emit(cls, nodes: List["Node"]) -> Sequence["RelativePolarCommand"]:
        commands: List[RelativePolarCommand] = []
        if not nodes:
            return commands

        commands.append(cls("SOS", []))
        # Emit move to for the first node (absolute position)
        first_node = nodes[0]
        pos = AbsoluteCoordinateRepresentation.emit_node_position(first_node)
        commands.append(cls("M", pos.tolist()))

        f_hat = np.array(
            [1.0, 0.0], dtype=np.float32
        )  # Initial f_hat for the first segment

        for i, node in enumerate(nodes):
            p_curr = node
            p_prev = (
                node.previous
            )  # This will be the last node for the first node in a closed contour

            # Calculate delta_pos for the current node relative to the previous node
            if i == 0:  # For the first node, its position is relative to itself (0,0)
                delta_pos = np.array([0.0, 0.0], dtype=np.float32)
            else:
                delta_pos = p_curr.coordinates - p_prev.coordinates

            r = float(np.linalg.norm(delta_pos))
            r_hat = np.array([-f_hat[1], f_hat[0]], dtype=np.float32)

            # Calculate phi for the node position relative to current f_hat
            if r > 1e-6:
                phi = float(
                    np.arctan2(np.dot(delta_pos, r_hat), np.dot(delta_pos, f_hat))
                )
            else:
                phi = 0.0

            # Calculate handle polar coordinates relative to current f_hat
            out_handle_world_rel = (
                p_curr.out_handle - p_curr.coordinates
                if p_curr.out_handle is not None
                else np.array([0.0, 0.0])
            )
            in_handle_world_rel = (
                p_curr.in_handle - p_curr.coordinates
                if p_curr.in_handle is not None
                else np.array([0.0, 0.0])
            )

            out_len = float(np.linalg.norm(out_handle_world_rel))
            out_phi = (
                float(
                    np.arctan2(
                        np.dot(out_handle_world_rel, r_hat),
                        np.dot(out_handle_world_rel, f_hat),
                    )
                )
                if out_len > 1e-6
                else 0.0
            )

            in_len = float(np.linalg.norm(in_handle_world_rel))
            in_phi = (
                float(
                    np.arctan2(
                        np.dot(in_handle_world_rel, r_hat),
                        np.dot(in_handle_world_rel, f_hat),
                    )
                )
                if in_len > 1e-6
                else 0.0
            )

            # Determine command type
            if p_curr.is_line:  # No handles
                if np.isclose(phi, np.pi / 2):
                    commands.append(cls("L_LEFT_90", [r]))
                elif np.isclose(phi, -np.pi / 2):
                    commands.append(cls("L_RIGHT_90", [r]))
                else:
                    commands.append(cls("L_POLAR", [r, phi]))
            else:  # Has at least one handle
                has_in = p_curr.in_handle is not None and in_len > 1e-6
                has_out = p_curr.out_handle is not None and out_len > 1e-6

                if has_in and not has_out:
                    commands.append(cls("N_POLAR_IN", [r, phi, in_len, in_phi]))
                elif has_out and not has_in:
                    commands.append(cls("N_POLAR_OUT", [r, phi, out_len, out_phi]))
                elif p_curr.is_smooth:
                    commands.append(cls("N_SMOOTH", [r, phi, out_phi, in_len, out_len]))
                else:
                    commands.append(
                        cls("N_POLAR", [r, phi, in_len, in_phi, out_len, out_phi])
                    )

            # Update f_hat for the next segment based on the outgoing handle of the current node
            if p_curr.out_handle is not None:
                next_f_hat_vec = p_curr.out_handle - p_curr.coordinates
                norm = np.linalg.norm(next_f_hat_vec)
                if norm > 1e-6:
                    f_hat = next_f_hat_vec / norm
                else:
                    f_hat = cls._unit(
                        delta_pos, f_hat
                    )  # Fallback to chord if handle is zero
            else:
                f_hat = cls._unit(
                    delta_pos, f_hat
                )  # For line segments, f_hat is direction of chord

        commands.append(cls("EOS", []))
        return commands

    @classmethod
    def contour_from_commands(
        cls, commands: Sequence[CommandRepresentation], tolerant: bool = True
    ) -> "NodeContour":
        from glyphogen.nodeglyph import NodeContour  # avoid circular import

        contour = NodeContour([])
        all_commands = list(commands)
        # Pop SOS
        if all_commands[0].command == "SOS":
            all_commands.pop(0)
        if len(all_commands) < 1:
            return contour

        command_tensors = []
        coord_tensors = []
        max_coords = cls.coordinate_width
        for cmd in all_commands:
            command_tensors.append(cls.encode_command_one_hot(cmd.command))
            padded = cmd.coordinates + [0] * (max_coords - len(cmd.coordinates))
            coord_tensors.append(torch.tensor(padded, dtype=torch.float32))

        sequence_tensor = torch.cat(
            [torch.stack(command_tensors), torch.stack(coord_tensors)], dim=1
        )
        _, abs_coords = cls.split_tensor(
            cls.unroll_relative_coordinates(sequence_tensor)
        )
        abs_coords_np = abs_coords.cpu().numpy()

        nodes_list: List[Node] = []

        # Iterate from the first actual node command (after SOS and M)
        for idx, command in enumerate(all_commands[1:], start=1):
            if command.command == "EOS":
                break

            pos_abs = abs_coords_np[idx, 0:2]
            in_abs = (
                abs_coords_np[idx, 2:4].copy()
                if command.command
                in [
                    "N_POLAR",
                    "N_POLAR_IN",
                    "N_SMOOTH",
                ]
                else None
            )
            out_abs = (
                abs_coords_np[idx, 4:6].copy()
                if command.command
                in [
                    "N_POLAR",
                    "N_POLAR_OUT",
                    "N_SMOOTH",
                ]
                else None
            )

            new_node = Node(
                coordinates=pos_abs.copy(),
                in_handle=in_abs,
                out_handle=out_abs,
                contour=None,  # Will be set by NodeContour constructor
            )
            nodes_list.append(new_node)

        return NodeContour(nodes_list)

    @classmethod
    def unroll_relative_coordinates(cls, sequence: torch.Tensor) -> torch.Tensor:
        commands, rel = cls.split_tensor(sequence)
        # abs_coords needs to be 6 wide to store pos, in_handle, out_handle
        abs_coords = torch.zeros(
            rel.shape[0], 6, device=sequence.device, dtype=sequence.dtype
        )

        current_pos = torch.zeros(2, device=sequence.device, dtype=sequence.dtype)
        f_hat = torch.tensor([1.0, 0.0], device=sequence.device, dtype=sequence.dtype)

        for i in range(sequence.shape[0]):
            idx = int(torch.argmax(commands[i]).item())
            co = rel[i]
            r_hat = torch.stack([-f_hat[1], f_hat[0]])

            pos_abs = current_pos
            in_abs = current_pos
            out_abs = current_pos
            next_f_hat = f_hat  # Default to keeping f_hat the same

            if idx == cls.encode_command("M"):
                current_pos = co[0:2]
                pos_abs = current_pos
            elif idx == cls.encode_command("L_POLAR"):
                r, phi = co[0], co[1]
                cos_phi, sin_phi = torch.cos(phi), torch.sin(phi)
                rotation_matrix = torch.stack(
                    [
                        torch.stack([cos_phi, -sin_phi]),
                        torch.stack([sin_phi, cos_phi]),
                    ]
                )
                move_dir = torch.matmul(rotation_matrix, f_hat)

                delta = r * move_dir
                current_pos = current_pos + delta
                pos_abs = current_pos
                next_f_hat = (
                    move_dir  # f_hat for next segment is direction of this segment
                )
            elif idx == cls.encode_command("L_LEFT_90"):
                r = co[0]
                # 90 degree left turn: cos=0, sin=1
                rotation_matrix = torch.tensor(
                    [[0.0, -1.0], [1.0, 0.0]],
                    device=sequence.device,
                    dtype=sequence.dtype,
                )
                move_dir = torch.matmul(rotation_matrix, f_hat)

                delta = r * move_dir
                current_pos = current_pos + delta
                pos_abs = current_pos
                next_f_hat = (
                    move_dir  # f_hat for next segment is direction of this segment
                )
            elif idx == cls.encode_command("L_RIGHT_90"):
                r = co[0]
                # 90 degree right turn: cos=0, sin=-1
                rotation_matrix = torch.tensor(
                    [[0.0, 1.0], [-1.0, 0.0]],
                    device=sequence.device,
                    dtype=sequence.dtype,
                )
                move_dir = torch.matmul(rotation_matrix, f_hat)

                delta = r * move_dir
                current_pos = current_pos + delta
                pos_abs = current_pos
                next_f_hat = (
                    move_dir  # f_hat for next segment is direction of this segment
                )

            elif idx == cls.encode_command("N_POLAR"):
                r, phi_node, in_len, in_phi, out_len, out_phi = co
                cos_phi, sin_phi = torch.cos(phi_node), torch.sin(phi_node)
                rotation_matrix_node = torch.stack(
                    [
                        torch.stack([cos_phi, -sin_phi]),
                        torch.stack([sin_phi, cos_phi]),
                    ]
                )

                # Position update
                node_dir = torch.matmul(rotation_matrix_node, f_hat)
                delta_pos = r * node_dir
                current_pos = current_pos + delta_pos
                pos_abs = current_pos

                # Handle directions expressed in the local (f_hat, r_hat) frame
                def _dir_from_polar(
                    length: torch.Tensor, angle: torch.Tensor
                ) -> torch.Tensor:
                    # cos component along f_hat, sin component along r_hat
                    return length * (
                        torch.cos(angle) * f_hat + torch.sin(angle) * r_hat
                    )

                out_handle_vec = _dir_from_polar(out_len, out_phi)
                in_handle_vec = _dir_from_polar(in_len, in_phi)

                out_handle_abs = pos_abs + out_handle_vec
                in_handle_abs = pos_abs + in_handle_vec

                in_abs = in_handle_abs
                out_abs = out_handle_abs

                # Update f_hat for the next segment based on the outgoing handle
                norm_out = torch.linalg.norm(out_handle_vec)
                if norm_out > 1e-6:
                    next_f_hat = out_handle_vec / norm_out
                else:
                    norm_delta = torch.linalg.norm(delta_pos)
                    if norm_delta > 1e-6:
                        next_f_hat = delta_pos / norm_delta
                    else:
                        next_f_hat = f_hat

            elif idx == cls.encode_command("N_POLAR_IN"):
                r, phi_node, in_len, in_phi = co[0:4]
                cos_phi, sin_phi = torch.cos(phi_node), torch.sin(phi_node)
                rotation_matrix_node = torch.stack(
                    [
                        torch.stack([cos_phi, -sin_phi]),
                        torch.stack([sin_phi, cos_phi]),
                    ]
                )

                node_dir = torch.matmul(rotation_matrix_node, f_hat)
                delta_pos = r * node_dir
                current_pos = current_pos + delta_pos
                pos_abs = current_pos

                def _dir_from_polar_in(
                    length: torch.Tensor, angle: torch.Tensor
                ) -> torch.Tensor:
                    return length * (
                        torch.cos(angle) * f_hat + torch.sin(angle) * r_hat
                    )

                in_handle_vec = _dir_from_polar_in(in_len, in_phi)
                in_handle_abs = pos_abs + in_handle_vec
                in_abs = in_handle_abs
                out_abs = pos_abs

                norm_delta = torch.linalg.norm(delta_pos)
                if norm_delta > 1e-6:
                    next_f_hat = delta_pos / norm_delta
                else:
                    next_f_hat = f_hat

            elif idx == cls.encode_command("N_POLAR_OUT"):
                r, phi_node, out_len, out_phi = co[0:4]
                cos_phi, sin_phi = torch.cos(phi_node), torch.sin(phi_node)
                rotation_matrix_node = torch.stack(
                    [
                        torch.stack([cos_phi, -sin_phi]),
                        torch.stack([sin_phi, cos_phi]),
                    ]
                )

                node_dir = torch.matmul(rotation_matrix_node, f_hat)
                delta_pos = r * node_dir
                current_pos = current_pos + delta_pos
                pos_abs = current_pos

                def _dir_from_polar_out(
                    length: torch.Tensor, angle: torch.Tensor
                ) -> torch.Tensor:
                    return length * (
                        torch.cos(angle) * f_hat + torch.sin(angle) * r_hat
                    )

                out_handle_vec = _dir_from_polar_out(out_len, out_phi)
                out_handle_abs = pos_abs + out_handle_vec
                in_abs = pos_abs
                out_abs = out_handle_abs

                norm_out = torch.linalg.norm(out_handle_vec)
                if norm_out > 1e-6:
                    next_f_hat = out_handle_vec / norm_out
                else:
                    norm_delta = torch.linalg.norm(delta_pos)
                    if norm_delta > 1e-6:
                        next_f_hat = delta_pos / norm_delta
                    else:
                        next_f_hat = f_hat

            elif idx == cls.encode_command("N_SMOOTH"):
                # co = [r, phi_node, out_phi, in_len, out_len]
                r, phi_node, out_phi, in_len, out_len = co[0:5]

                cos_phi, sin_phi = torch.cos(phi_node), torch.sin(phi_node)
                rotation_matrix_node = torch.stack(
                    [
                        torch.stack([cos_phi, -sin_phi]),
                        torch.stack([sin_phi, cos_phi]),
                    ]
                )
                node_dir = torch.matmul(rotation_matrix_node, f_hat)
                delta_pos = r * node_dir
                current_pos = current_pos + delta_pos
                pos_abs = current_pos

                # Smooth: out handle angle is given; in handle is pi opposite
                out_dir = torch.cos(out_phi) * f_hat + torch.sin(out_phi) * r_hat
                in_dir = -out_dir

                out_handle_abs = pos_abs + out_len * out_dir
                in_handle_abs = pos_abs + in_len * in_dir

                in_abs = in_handle_abs
                out_abs = out_handle_abs

                norm_out = torch.linalg.norm(out_dir)
                if norm_out > 1e-6:
                    next_f_hat = out_dir / norm_out
                else:
                    norm_delta = torch.linalg.norm(delta_pos)
                    if norm_delta > 1e-6:
                        next_f_hat = delta_pos / norm_delta
                    else:
                        next_f_hat = f_hat
            # SOS and EOS do not change position or f_hat

            def _zero_small(t: torch.Tensor) -> torch.Tensor:
                return torch.where(torch.abs(t) < 1e-5, torch.zeros_like(t), t)

            abs_coords[i, 0:2] = _zero_small(pos_abs)
            abs_coords[i, 2:4] = _zero_small(in_abs)
            abs_coords[i, 4:6] = _zero_small(out_abs)
            f_hat = next_f_hat  # Update f_hat for the next iteration

        return torch.cat([commands, abs_coords], dim=1)

    @classmethod
    def image_space_to_mask_space(cls, sequence, box: Float[torch.Tensor, "4"]):
        """
        Normalizes a sequence's image-space coordinates to the model's internal
        [-1, 1] range relative to a given bounding box.

        For polar coordinates:
        - M: absolute (x, y) -> translate and scale to [-1, 1]
        - Distances (r, lengths): scale by 2/avg_dim
        - Angles (phi): divide by π to get [-1, 1]
        """
        commands, coords_img = cls.split_tensor(sequence)
        x1, y1, x2, y2 = box
        width = torch.clamp(x2 - x1, min=1.0)
        height = torch.clamp(y2 - y1, min=1.0)
        avg_dim = (width + height) / 2.0

        command_indices = torch.argmax(commands, dim=-1)
        coord_width = coords_img.shape[1]

        # Build masks for each command type
        m_mask = command_indices == cls.encode_command("M")
        l_polar_mask = command_indices == cls.encode_command("L_POLAR")
        l_left_mask = command_indices == cls.encode_command("L_LEFT_90")
        l_right_mask = command_indices == cls.encode_command("L_RIGHT_90")
        n_polar_mask = command_indices == cls.encode_command("N_POLAR")
        n_polar_in_mask = command_indices == cls.encode_command("N_POLAR_IN")
        n_polar_out_mask = command_indices == cls.encode_command("N_POLAR_OUT")
        n_smooth_mask = command_indices == cls.encode_command("N_SMOOTH")

        # Start with a copy
        coords_norm = coords_img.clone()

        # --- M: absolute position ---
        # Translate then scale to [-1, 1]
        m_norm = coords_img.clone()
        m_norm[:, 0] = ((coords_img[:, 0] - x1) / width) * 2 - 1
        m_norm[:, 1] = ((coords_img[:, 1] - y1) / height) * 2 - 1
        coords_norm = torch.where(m_mask.unsqueeze(1), m_norm, coords_norm)

        # --- L_POLAR: [r, phi] ---
        # r -> scale, phi -> /π
        l_polar_norm = coords_img.clone()
        l_polar_norm[:, 0] = coords_img[:, 0] * (2.0 / avg_dim)  # r
        l_polar_norm[:, 1] = coords_img[:, 1] / np.pi  # phi
        coords_norm = torch.where(l_polar_mask.unsqueeze(1), l_polar_norm, coords_norm)

        # --- L_LEFT_90, L_RIGHT_90: [r] ---
        l_turn_norm = coords_img.clone()
        l_turn_norm[:, 0] = coords_img[:, 0] * (2.0 / avg_dim)  # r
        coords_norm = torch.where(l_left_mask.unsqueeze(1), l_turn_norm, coords_norm)
        coords_norm = torch.where(l_right_mask.unsqueeze(1), l_turn_norm, coords_norm)

        # --- N_POLAR: [r, phi, in_len, in_phi, out_len, out_phi] ---
        n_polar_norm = coords_img.clone()
        n_polar_norm[:, 0] = coords_img[:, 0] * (2.0 / avg_dim)  # r
        n_polar_norm[:, 1] = coords_img[:, 1] / np.pi  # phi
        n_polar_norm[:, 2] = coords_img[:, 2] * (2.0 / avg_dim)  # in_len
        n_polar_norm[:, 3] = coords_img[:, 3] / np.pi  # in_phi
        n_polar_norm[:, 4] = coords_img[:, 4] * (2.0 / avg_dim)  # out_len
        n_polar_norm[:, 5] = coords_img[:, 5] / np.pi  # out_phi
        coords_norm = torch.where(n_polar_mask.unsqueeze(1), n_polar_norm, coords_norm)

        # --- N_POLAR_IN: [r, phi, in_len, in_phi] ---
        n_polar_in_norm = coords_img.clone()
        n_polar_in_norm[:, 0] = coords_img[:, 0] * (2.0 / avg_dim)  # r
        n_polar_in_norm[:, 1] = coords_img[:, 1] / np.pi  # phi
        n_polar_in_norm[:, 2] = coords_img[:, 2] * (2.0 / avg_dim)  # in_len
        n_polar_in_norm[:, 3] = coords_img[:, 3] / np.pi  # in_phi
        coords_norm = torch.where(
            n_polar_in_mask.unsqueeze(1), n_polar_in_norm, coords_norm
        )

        # --- N_POLAR_OUT: [r, phi, out_len, out_phi] ---
        n_polar_out_norm = coords_img.clone()
        n_polar_out_norm[:, 0] = coords_img[:, 0] * (2.0 / avg_dim)  # r
        n_polar_out_norm[:, 1] = coords_img[:, 1] / np.pi  # phi
        n_polar_out_norm[:, 2] = coords_img[:, 2] * (2.0 / avg_dim)  # out_len
        n_polar_out_norm[:, 3] = coords_img[:, 3] / np.pi  # out_phi
        coords_norm = torch.where(
            n_polar_out_mask.unsqueeze(1), n_polar_out_norm, coords_norm
        )

        # --- N_SMOOTH: [r, phi, out_phi, len_in, len_out] ---
        n_smooth_norm = coords_img.clone()
        n_smooth_norm[:, 0] = coords_img[:, 0] * (2.0 / avg_dim)  # r
        n_smooth_norm[:, 1] = coords_img[:, 1] / np.pi  # phi
        n_smooth_norm[:, 2] = coords_img[:, 2] / np.pi  # out_phi
        n_smooth_norm[:, 3] = coords_img[:, 3] * (2.0 / avg_dim)  # len_in
        n_smooth_norm[:, 4] = coords_img[:, 4] * (2.0 / avg_dim)  # len_out
        coords_norm = torch.where(
            n_smooth_mask.unsqueeze(1), n_smooth_norm, coords_norm
        )

        return torch.cat([commands, coords_norm], dim=-1)

    @classmethod
    def mask_space_to_image_space(cls, sequence, box):
        """
        Denormalizes a sequence's [-1, 1] coordinates back to image space.

        For polar coordinates:
        - M: scale and translate from [-1, 1] to absolute (x, y)
        - Distances (r, lengths): multiply by avg_dim/2
        - Angles (phi): multiply by π
        """
        commands, coords_norm = cls.split_tensor(sequence)
        x1, y1, x2, y2 = box
        width = torch.clamp(x2 - x1, min=1.0)
        height = torch.clamp(y2 - y1, min=1.0)
        avg_dim = (width + height) / 2.0

        command_indices = torch.argmax(commands, dim=-1)

        # Build masks for each command type
        m_mask = command_indices == cls.encode_command("M")
        l_polar_mask = command_indices == cls.encode_command("L_POLAR")
        l_left_mask = command_indices == cls.encode_command("L_LEFT_90")
        l_right_mask = command_indices == cls.encode_command("L_RIGHT_90")
        n_polar_mask = command_indices == cls.encode_command("N_POLAR")
        n_polar_in_mask = command_indices == cls.encode_command("N_POLAR_IN")
        n_polar_out_mask = command_indices == cls.encode_command("N_POLAR_OUT")
        n_smooth_mask = command_indices == cls.encode_command("N_SMOOTH")

        # Start with a copy
        coords_img = coords_norm.clone()

        # --- M: absolute position ---
        m_denorm = coords_norm.clone()
        m_denorm[:, 0] = (coords_norm[:, 0] + 1) / 2 * width + x1
        m_denorm[:, 1] = (coords_norm[:, 1] + 1) / 2 * height + y1
        coords_img = torch.where(m_mask.unsqueeze(1), m_denorm, coords_img)

        # --- L_POLAR: [r, phi] ---
        l_polar_denorm = coords_norm.clone()
        l_polar_denorm[:, 0] = coords_norm[:, 0] * (avg_dim / 2.0)  # r
        l_polar_denorm[:, 1] = coords_norm[:, 1] * np.pi  # phi
        coords_img = torch.where(l_polar_mask.unsqueeze(1), l_polar_denorm, coords_img)

        # --- L_LEFT_90, L_RIGHT_90: [r] ---
        l_turn_denorm = coords_norm.clone()
        l_turn_denorm[:, 0] = coords_norm[:, 0] * (avg_dim / 2.0)  # r
        coords_img = torch.where(l_left_mask.unsqueeze(1), l_turn_denorm, coords_img)
        coords_img = torch.where(l_right_mask.unsqueeze(1), l_turn_denorm, coords_img)

        # --- N_POLAR: [r, phi, in_len, in_phi, out_len, out_phi] ---
        n_polar_denorm = coords_norm.clone()
        n_polar_denorm[:, 0] = coords_norm[:, 0] * (avg_dim / 2.0)  # r
        n_polar_denorm[:, 1] = coords_norm[:, 1] * np.pi  # phi
        n_polar_denorm[:, 2] = coords_norm[:, 2] * (avg_dim / 2.0)  # in_len
        n_polar_denorm[:, 3] = coords_norm[:, 3] * np.pi  # in_phi
        n_polar_denorm[:, 4] = coords_norm[:, 4] * (avg_dim / 2.0)  # out_len
        n_polar_denorm[:, 5] = coords_norm[:, 5] * np.pi  # out_phi
        coords_img = torch.where(n_polar_mask.unsqueeze(1), n_polar_denorm, coords_img)

        # --- N_POLAR_IN: [r, phi, in_len, in_phi] ---
        n_polar_in_denorm = coords_norm.clone()
        n_polar_in_denorm[:, 0] = coords_norm[:, 0] * (avg_dim / 2.0)  # r
        n_polar_in_denorm[:, 1] = coords_norm[:, 1] * np.pi  # phi
        n_polar_in_denorm[:, 2] = coords_norm[:, 2] * (avg_dim / 2.0)  # in_len
        n_polar_in_denorm[:, 3] = coords_norm[:, 3] * np.pi  # in_phi
        coords_img = torch.where(
            n_polar_in_mask.unsqueeze(1), n_polar_in_denorm, coords_img
        )

        # --- N_POLAR_OUT: [r, phi, out_len, out_phi] ---
        n_polar_out_denorm = coords_norm.clone()
        n_polar_out_denorm[:, 0] = coords_norm[:, 0] * (avg_dim / 2.0)  # r
        n_polar_out_denorm[:, 1] = coords_norm[:, 1] * np.pi  # phi
        n_polar_out_denorm[:, 2] = coords_norm[:, 2] * (avg_dim / 2.0)  # out_len
        n_polar_out_denorm[:, 3] = coords_norm[:, 3] * np.pi  # out_phi
        coords_img = torch.where(
            n_polar_out_mask.unsqueeze(1), n_polar_out_denorm, coords_img
        )

        # --- N_SMOOTH: [r, phi, out_phi, len_in, len_out] ---
        n_smooth_denorm = coords_norm.clone()
        n_smooth_denorm[:, 0] = coords_norm[:, 0] * (avg_dim / 2.0)  # r
        n_smooth_denorm[:, 1] = coords_norm[:, 1] * np.pi  # phi
        n_smooth_denorm[:, 2] = coords_norm[:, 2] * np.pi  # out_phi
        n_smooth_denorm[:, 3] = coords_norm[:, 3] * (avg_dim / 2.0)  # len_in
        n_smooth_denorm[:, 4] = coords_norm[:, 4] * (avg_dim / 2.0)  # len_out
        coords_img = torch.where(
            n_smooth_mask.unsqueeze(1), n_smooth_denorm, coords_img
        )

        return torch.cat([commands, coords_img], dim=-1)

    @classmethod
    def tensors_to_segments(cls, cmd, coord):
        raise NotImplementedError(
            "We don't need this; use SVGCommand to render for visualization."
        )

    # Class-level storage for normalization statistics
    _stats_initialized = False
    _mean_tensor = None
    _std_tensor = None

    @classmethod
    def initialize_stats(cls, stats_path: str = "data/coord_stats.pt"):
        """Load stats and create broadcastable tensors for standardization."""
        from collections import defaultdict

        try:
            stats = torch.load(stats_path)
        except FileNotFoundError:
            print(
                f"Warning: {stats_path} not found. Using default (0,1) stats. "
                "Run analyze_dataset_stats.py to generate it."
            )
            stats = defaultdict(lambda: {"mean": 0.0, "std": 1.0})

        mean_tensor = torch.zeros(cls.command_width, cls.coordinate_width)
        std_tensor = torch.ones(cls.command_width, cls.coordinate_width)
        cmd_indices = {cmd: cls.encode_command(cmd) for cmd in cls.grammar.keys()}

        for cmd_name, cmd_idx in cmd_indices.items():
            if cmd_name == "M":
                mean_tensor[cmd_idx, 0] = stats["M_abs_x"]["mean"]
                std_tensor[cmd_idx, 0] = stats["M_abs_x"]["std"]
                mean_tensor[cmd_idx, 1] = stats["M_abs_y"]["mean"]
                std_tensor[cmd_idx, 1] = stats["M_abs_y"]["std"]
            elif cmd_name in [
                "L_POLAR",
                "L_LEFT_90",
                "L_RIGHT_90",
                "N_POLAR",
                "N_POLAR_IN",
                "N_POLAR_OUT",
                "N_SMOOTH",
            ]:
                # All these commands have an on-curve relative move (r, phi)
                mean_tensor[cmd_idx, 0] = stats["ON_CURVE_R"]["mean"]
                std_tensor[cmd_idx, 0] = stats["ON_CURVE_R"]["std"]
                if cmd_name == "L_POLAR" or cmd_name.startswith("N_"):
                    mean_tensor[cmd_idx, 1] = stats["ON_CURVE_PHI"]["mean"]
                    std_tensor[cmd_idx, 1] = stats["ON_CURVE_PHI"]["std"]

            if cmd_name == "N_POLAR":
                mean_tensor[cmd_idx, 2] = stats["IN_HANDLE_LEN"]["mean"]
                std_tensor[cmd_idx, 2] = stats["IN_HANDLE_LEN"]["std"]
                mean_tensor[cmd_idx, 3] = stats["IN_HANDLE_PHI"]["mean"]
                std_tensor[cmd_idx, 3] = stats["IN_HANDLE_PHI"]["std"]
                mean_tensor[cmd_idx, 4] = stats["OUT_HANDLE_LEN"]["mean"]
                std_tensor[cmd_idx, 4] = stats["OUT_HANDLE_LEN"]["std"]
                mean_tensor[cmd_idx, 5] = stats["OUT_HANDLE_PHI"]["mean"]
                std_tensor[cmd_idx, 5] = stats["OUT_HANDLE_PHI"]["std"]
            elif cmd_name == "N_POLAR_IN":
                mean_tensor[cmd_idx, 2] = stats["IN_HANDLE_LEN"]["mean"]
                std_tensor[cmd_idx, 2] = stats["IN_HANDLE_LEN"]["std"]
                mean_tensor[cmd_idx, 3] = stats["IN_HANDLE_PHI"]["mean"]
                std_tensor[cmd_idx, 3] = stats["IN_HANDLE_PHI"]["std"]
            elif cmd_name == "N_POLAR_OUT":
                mean_tensor[cmd_idx, 2] = stats["OUT_HANDLE_LEN"]["mean"]
                std_tensor[cmd_idx, 2] = stats["OUT_HANDLE_LEN"]["std"]
                mean_tensor[cmd_idx, 3] = stats["OUT_HANDLE_PHI"]["mean"]
                std_tensor[cmd_idx, 3] = stats["OUT_HANDLE_PHI"]["std"]
            elif cmd_name == "N_SMOOTH":
                mean_tensor[cmd_idx, 2] = stats["OUT_HANDLE_PHI"]["mean"]  # out_phi
                std_tensor[cmd_idx, 2] = stats["OUT_HANDLE_PHI"]["std"]
                mean_tensor[cmd_idx, 3] = stats["IN_HANDLE_LEN"]["mean"]  # len_in
                std_tensor[cmd_idx, 3] = stats["IN_HANDLE_LEN"]["std"]
                mean_tensor[cmd_idx, 4] = stats["OUT_HANDLE_LEN"]["mean"]  # len_out
                std_tensor[cmd_idx, 4] = stats["OUT_HANDLE_LEN"]["std"]

        cls._mean_tensor = mean_tensor
        cls._std_tensor = std_tensor
        cls._stats_initialized = True

    @classmethod
    def get_initial_stats_dict(cls) -> Dict[str, List[float]]:
        """Get an initial stats dictionary with empty lists for each relevant field."""
        return {
            "M_abs_x": [],
            "M_abs_y": [],
            "ON_CURVE_R": [],
            "ON_CURVE_PHI": [],
            "IN_HANDLE_LEN": [],
            "IN_HANDLE_PHI": [],
            "OUT_HANDLE_LEN": [],
            "OUT_HANDLE_PHI": [],
        }

    @classmethod
    def update_stats_dict_with_command(cls, STAT_GROUPS, command, coord_vec):
        if command == "M":
            STAT_GROUPS["M_abs_x"].append(coord_vec[0].item())
            STAT_GROUPS["M_abs_y"].append(coord_vec[1].item())
        elif command in [
            "L_POLAR",
            "L_LEFT_90",
            "L_RIGHT_90",
            "N_POLAR",
            "N_POLAR_IN",
            "N_POLAR_OUT",
            "N_SMOOTH",
        ]:
            # All these commands have an on-curve relative move (r, phi)
            STAT_GROUPS["ON_CURVE_R"].append(coord_vec[0].item())
            if command == "L_POLAR" or command.startswith("N_"):
                STAT_GROUPS["ON_CURVE_PHI"].append(coord_vec[1].item())

            if command == "N_POLAR":
                STAT_GROUPS["IN_HANDLE_LEN"].append(coord_vec[2].item())
                STAT_GROUPS["IN_HANDLE_PHI"].append(coord_vec[3].item())
                STAT_GROUPS["OUT_HANDLE_LEN"].append(coord_vec[4].item())
                STAT_GROUPS["OUT_HANDLE_PHI"].append(coord_vec[5].item())
            elif command == "N_POLAR_IN":
                STAT_GROUPS["IN_HANDLE_LEN"].append(coord_vec[2].item())
                STAT_GROUPS["IN_HANDLE_PHI"].append(coord_vec[3].item())
            elif command == "N_POLAR_OUT":
                STAT_GROUPS["OUT_HANDLE_LEN"].append(coord_vec[2].item())
                STAT_GROUPS["OUT_HANDLE_PHI"].append(coord_vec[3].item())
            elif command == "N_SMOOTH":
                STAT_GROUPS["OUT_HANDLE_PHI"].append(coord_vec[2].item())  # out_phi
                STAT_GROUPS["IN_HANDLE_LEN"].append(coord_vec[3].item())  # len_in
                STAT_GROUPS["OUT_HANDLE_LEN"].append(coord_vec[4].item())  # len_out

    @classmethod
    def get_stats_for_sequence(cls, command_indices: torch.Tensor):
        """Get mean and std tensors for a sequence of command indices."""
        if not cls._stats_initialized:
            cls.initialize_stats()
        assert cls._mean_tensor is not None and cls._std_tensor is not None
        # Move stats to same device as input
        mean_tensor = cls._mean_tensor.to(command_indices.device)
        std_tensor = cls._std_tensor.to(command_indices.device)
        means = mean_tensor[command_indices]
        stds = std_tensor[command_indices]
        return means, stds

    @classmethod
    def standardize(cls, coords: torch.Tensor, means: torch.Tensor, stds: torch.Tensor):
        """Standardize coordinates using provided means and stds."""
        return (coords - means) / stds

    @classmethod
    def de_standardize(
        cls, coords_std: torch.Tensor, means: torch.Tensor, stds: torch.Tensor
    ):
        """De-standardize coordinates using provided means and stds."""
        return coords_std * stds + means
