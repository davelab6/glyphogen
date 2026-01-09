from abc import ABC
from typing import List, Optional, Self, Sequence, Union, TYPE_CHECKING
from jaxtyping import Float

import numpy as np
import numpy.typing as npt
import torch

from glyphogen.nodeglyph import Node, NodeContour

MAX_COORDINATE = 1000


# Dammit Python
class classproperty:
    def __init__(self, func):
        self.fget = func

    def __get__(self, instance, owner):
        return self.fget(owner)


class CoordinateRepresentation(ABC):
    @classmethod
    def emit_node_position(cls, n: "Node") -> npt.NDArray[np.float32]: ...

    @classmethod
    def emit_in_handle(cls, n: "Node") -> Optional[npt.NDArray[np.float32]]: ...

    @classmethod
    def emit_out_handle(cls, n: "Node") -> Optional[npt.NDArray[np.float32]]: ...


class AbsoluteCoordinateRepresentation(CoordinateRepresentation):
    @classmethod
    def emit_node_position(cls, n: "Node") -> npt.NDArray[np.float32]:
        return n.coordinates

    @classmethod
    def emit_in_handle(cls, n: "Node") -> Optional[npt.NDArray[np.float32]]:
        return n.in_handle

    @classmethod
    def emit_out_handle(cls, n: "Node") -> Optional[npt.NDArray[np.float32]]:
        return n.out_handle


class AbsolutePositionRelativeHandleRepresentation(CoordinateRepresentation):
    @classmethod
    def emit_node_position(cls, n: "Node") -> npt.NDArray[np.float32]:
        return n.coordinates

    @classmethod
    def emit_in_handle(cls, n: "Node") -> Optional[npt.NDArray[np.float32]]:
        if n.in_handle is None:
            return None
        return n.in_handle - n.coordinates

    @classmethod
    def emit_out_handle(cls, n: "Node") -> Optional[npt.NDArray[np.float32]]:
        if n.out_handle is None:
            return None
        return n.out_handle - n.coordinates


class RelativeCoordinateRepresentation(AbsolutePositionRelativeHandleRepresentation):
    """Handles are also relative to the node position."""

    @classmethod
    def emit_node_position(cls, n: "Node") -> npt.NDArray[np.float32]:
        if n.index == 0:
            return n.coordinates
        previous_node = n.previous
        return n.coordinates - previous_node.coordinates


class CommandRepresentation(ABC):
    grammar: dict[str, int]
    command: str
    coordinates: List[Union[int, float]]
    coordinate_representation: type[
        "CoordinateRepresentation"
    ]  # How would you like your coordinates?

    @classmethod
    def emit(cls, nodes: List["Node"]) -> Sequence[Self]: ...

    @classmethod
    def contour_from_commands(cls, commands: Sequence[Self]) -> "NodeContour":
        raise NotImplementedError()

    @classproperty
    def command_width(cls) -> int:
        return len(cls.grammar.keys())

    @classproperty
    def coordinate_width(cls) -> int:
        return max(cls.grammar.values())

    @classmethod
    def encode_command(cls, s: str) -> int:
        return list(cls.grammar.keys()).index(s)

    @classmethod
    def encode_command_one_hot(cls, s: str) -> torch.Tensor:
        index = cls.encode_command(s)
        one_hot = torch.zeros(cls.command_width, dtype=torch.float32)
        one_hot[index] = 1.0
        return one_hot

    @classmethod
    def decode_command(cls, index: int) -> str:
        return list(cls.grammar.keys())[index]

    @classmethod
    def decode_command_one_hot(cls, one_hot: torch.Tensor) -> str:
        index = int(torch.argmax(one_hot).item())
        return cls.decode_command(index)

    @classmethod
    def split_tensor(cls, tensor: torch.Tensor) -> Sequence[torch.Tensor]:
        """Splits a tensor of shape (N, command_width + coordinate_width)
        or (B, N, command_width + coordinate_width) into a tuple of
        (commands, coordinates) tensors."""
        command_width = cls.command_width
        if tensor.dim() == 2:
            commands = tensor[:, :command_width]
            coordinates = tensor[:, command_width:]
        elif tensor.dim() == 3:
            commands = tensor[:, :, :command_width]
            coordinates = tensor[:, :, command_width:]
        else:
            raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}")
        return commands, coordinates

    def __init__(self, command: str, coordinates: List[Union[int, float]]):
        if command not in self.grammar:
            raise ValueError(f"Invalid command: {command}")
        if len(coordinates) != self.grammar[command]:
            raise ValueError(
                f"Invalid number of coordinates for command {command}: expected {self.grammar[command]}, got {len(coordinates)}"
            )
        self.command = command
        self.coordinates = coordinates


class SVGCommand(CommandRepresentation):
    grammar = {
        "M": 2,  # Move to (2 coordinates)
        "L": 2,  # Line to (2 coordinates)
        "H": 1,  # Horizontal line to (1 coordinate)
        "V": 1,  # Vertical line to (1 coordinate)
        "C": 6,  # Cubic Bezier curve (6 coordinates)
        "Z": 0,  # Close path (no coordinates)
    }
    coordinate_representation = AbsoluteCoordinateRepresentation

    @classmethod
    def emit(cls, nodes: List["Node"]) -> Sequence[Self]:
        commands = []
        if not nodes:
            return commands
        # Emit move to for the first node
        first_node = nodes[0]
        pos = cls.coordinate_representation.emit_node_position(first_node)
        commands.append(SVGCommand("M", pos.tolist()))
        for node in nodes[1:] + [first_node]:
            # How does the previous node join to this?
            prev_node = node.previous
            if (node.in_handle is None) and (prev_node.out_handle is None):
                # Straight line
                pos = cls.coordinate_representation.emit_node_position(node).tolist()
                commands.append(SVGCommand("L", pos))
            else:
                # Cubic Bezier curve
                out_coords = cls.coordinate_representation.emit_out_handle(prev_node)
                if out_coords is None:
                    out_coords = cls.coordinate_representation.emit_node_position(
                        prev_node
                    )
                in_handle = cls.coordinate_representation.emit_in_handle(node)
                if in_handle is None:
                    in_handle = cls.coordinate_representation.emit_node_position(node)
                pos = cls.coordinate_representation.emit_node_position(node).tolist()
                coords = out_coords.tolist() + in_handle.tolist() + pos
                commands.append(SVGCommand("C", coords))
        # If the last command is a line back to the start, drop it, it's redundant with the Z
        if len(commands) > 1:
            last_cmd = commands[-1]
            if last_cmd.command == "L":
                start_pos = cls.coordinate_representation.emit_node_position(first_node)
                last_pos = np.array(last_cmd.coordinates, dtype=np.float32)
                if np.allclose(start_pos, last_pos):
                    commands.pop()
        commands.append(SVGCommand("Z", []))  # Close path
        return commands

    @classmethod
    def contour_from_commands(
        cls, commands: Sequence[CommandRepresentation]
    ) -> "NodeContour":
        contour = NodeContour([])
        # Expect a M
        if not commands or commands[0].command != "M":
            raise ValueError("SVGCommand sequence must start with an 'M' command.")
        cur_node = contour.push(
            coordinates=np.array(commands[0].coordinates, dtype=np.float32),
            in_handle=None,
            out_handle=None,
        )
        for command in commands[1:]:
            if command.command == "L":
                pos = np.array(command.coordinates, dtype=np.float32)
                cur_node = contour.push(
                    np.array(command.coordinates, dtype=np.float32),
                    in_handle=None,
                    out_handle=None,
                )
            elif command.command == "C":
                coords = command.coordinates
                out_handle = np.array(coords[0:2], dtype=np.float32)
                cur_node.out_handle = out_handle
                in_handle = np.array(coords[2:4], dtype=np.float32)
                pos = np.array(coords[4:6], dtype=np.float32)
                new_node = contour.push(
                    coordinates=pos,
                    in_handle=in_handle,
                    out_handle=None,
                )
                cur_node = new_node
            elif command.command == "Z":
                # Close path, do nothing
                pass
            else:
                raise ValueError(f"Unsupported SVG command: {command.command}")
        # If we ended up back at the start and the last node was a curve, merge the handles and remove the last node
        if np.array_equal(contour.nodes[0].coordinates, contour.nodes[-1].coordinates):
            start_node = contour.nodes[0]
            end_node = contour.nodes[-1]
            start_node.in_handle = end_node.in_handle
            contour.nodes.pop()

        return contour

    @classmethod
    def tensors_to_segments(cls, cmd, coord):
        """Convert an encoded command and coordinate tensor to segment points
        and control point counts for the diffvg renderer.

        This should be fairly simple for SVG commands as it maps to segments
        quite directly.
        """

        command_tensor = torch.argmax(cmd, dim=-1)
        all_points = []
        all_num_cp = []
        contour_splits = []
        point_splits = []
        current_contour_points = []
        current_contour_num_cp = []
        for i in range(len(command_tensor)):
            command = command_tensor[i]
            if command == cls.encode_command("M"):
                # Start a new contour
                if current_contour_points:
                    all_points.extend(current_contour_points)
                    all_num_cp.extend(current_contour_num_cp)
                    contour_splits.append(len(all_num_cp))
                    point_splits.append(len(all_points))
                    current_contour_points = []
                    current_contour_num_cp = []
                # Add the move-to point
                current_contour_points.append(coord[i, 0:2])
            elif command == cls.encode_command("L"):
                # Line to
                current_contour_points.append(coord[i, 0:2])
                current_contour_num_cp.append(0)
            elif command == cls.encode_command("C"):
                # Cubic Bezier curve
                current_contour_points.append(coord[i, 0:2])  # Control point 1
                current_contour_points.append(coord[i, 2:4])  # Control point 2
                current_contour_points.append(coord[i, 4:6])  # End point
                current_contour_num_cp.append(2)
            elif command == cls.encode_command("Z"):
                # Close path, do nothing
                pass
            else:
                raise ValueError(f"Unsupported SVG command in tensor: {command}")
        # Loop back to the start to close the contour
        if current_contour_points:
            current_contour_points.append(current_contour_points[0])
            current_contour_num_cp.append(0)
        if current_contour_points:
            all_points.extend(current_contour_points)
            all_num_cp.extend(current_contour_num_cp)
            contour_splits.append(len(all_num_cp))
            point_splits.append(len(all_points))
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


class NodeCommand(CommandRepresentation):
    grammar = {
        "SOS": 0,
        "M": 2,  # Absolute move to (x, y)
        "L": 2,  # Relative line to (dx, dy)
        "LH": 1,  # Relative horizontal line to (dx)
        "LV": 1,  # Relative vertical line to (dy)
        "N": 6,  # Relative node with two handles (dx, dy, dhix, dhiy, dhox, dhoy)
        "NS": 5,  # Relative smooth node (dx, dy, angle, length_in, length_out)
        "NH": 4,  # Relative horizontal-handle node (dx, dy, dh_in_x, dh_out_x)
        "NV": 4,  # Relative vertical-handle node (dx, dy, dh_in_y, dh_out_y)
        "NCI": 4,  # Relative node with in-handle only (dx, dy, dhix, dhiy)
        "NCO": 4,  # Relative node with out-handle only (dx, dy, dhox, dhoy)
        "EOS": 0,
    }
    coordinate_representation = RelativeCoordinateRepresentation

    @classmethod
    def emit(cls, nodes: List["Node"]) -> Sequence["NodeCommand"]:
        commands = []
        if not nodes:
            return commands
        # Emit SOS
        commands.append(NodeCommand("SOS", []))
        # Emit move to for the first node
        first_node = nodes[0]
        pos = cls.coordinate_representation.emit_node_position(first_node)
        commands.append(NodeCommand("M", pos.tolist()))
        for node in nodes:
            if node == first_node:
                rel_pos = [0.0, 0.0]
            else:
                rel_pos = cls.coordinate_representation.emit_node_position(
                    node
                ).tolist()
            in_handle = cls.coordinate_representation.emit_in_handle(node)
            out_handle = cls.coordinate_representation.emit_out_handle(node)
            if node.is_line:
                if node.is_horizontal_line:
                    # Horizontal line
                    commands.append(NodeCommand("LH", [rel_pos[0]]))
                elif node.is_vertical_line:
                    # Vertical line
                    commands.append(NodeCommand("LV", [rel_pos[1]]))
                else:
                    # Straight line
                    commands.append(NodeCommand("L", rel_pos))
            elif node.handles_horizontal:
                assert in_handle is not None and out_handle is not None
                commands.append(
                    NodeCommand("NH", rel_pos + [in_handle[0], out_handle[0]])
                )
            elif node.handles_vertical:
                assert in_handle is not None and out_handle is not None
                commands.append(
                    NodeCommand("NV", rel_pos + [in_handle[1], out_handle[1]])
                )
            elif node.is_smooth:
                assert in_handle is not None and out_handle is not None
                # For a smooth node, the handles are collinear and opposite.
                # We define the geometry by the angle of the outgoing handle,
                # and the lengths of both handles.
                # To be robust to slight imprecision in the source font,
                # we average the direction of the two handles.
                vec_in = in_handle
                vec_out = out_handle
                vec_in_opposite = -vec_in
                norm_in = np.linalg.norm(vec_in_opposite)
                norm_out = np.linalg.norm(vec_out)

                # Average the unit vectors
                avg_vec = (vec_in_opposite / norm_in) + (vec_out / norm_out)
                angle = np.arctan2(avg_vec[1], avg_vec[0])

                length_in = np.linalg.norm(vec_in)
                length_out = np.linalg.norm(vec_out)
                commands.append(
                    NodeCommand(
                        "NS",
                        rel_pos + [float(angle), float(length_in), float(length_out)],
                    )
                )
            elif in_handle is not None and out_handle is not None:
                commands.append(
                    NodeCommand("N", rel_pos + in_handle.tolist() + out_handle.tolist())
                )
            elif in_handle is not None:
                commands.append(NodeCommand("NCI", rel_pos + in_handle.tolist()))
            elif out_handle is not None:
                commands.append(NodeCommand("NCO", rel_pos + out_handle.tolist()))
            else:
                # This case should be handled by is_line, but as a fallback
                commands.append(NodeCommand("L", rel_pos))

        # Emit EOS
        commands.append(NodeCommand("EOS", []))

        return commands

    @classmethod
    def contour_from_commands(
        cls, commands: Sequence[CommandRepresentation], tolerant=True
    ) -> "NodeContour":
        contour = NodeContour([])
        commands = list(commands)
        # Pop SOS
        if commands[0].command == "SOS":
            commands.pop(0)
        # Expect a M
        if len(commands) < 1:
            return contour
        if not commands or commands[0].command != "M" and not tolerant:
            raise ValueError(
                f"NodeCommand second command must be 'M' command, found {commands[0].command}."
            )
        cur_coord = np.array(commands[0].coordinates[0:2], dtype=np.float32)
        if cur_coord.shape != (2,) and tolerant:
            cur_coord = np.pad(cur_coord, (0, 2 - cur_coord.shape[0]), "constant")
        command = "M"
        for step in commands[1:]:
            assert cur_coord.shape == (
                2,
            ), f"Messed up after command {command}, shape is {cur_coord.shape}"
            command = step.command
            coords = step.coordinates
            if command == "L":
                cur_coord += np.array(coords[0:2], dtype=np.float32)
                contour.push(
                    coordinates=cur_coord.copy(), in_handle=None, out_handle=None
                )
            elif command == "LH":
                cur_coord[0] += coords[0]
                contour.push(
                    coordinates=cur_coord.copy(), in_handle=None, out_handle=None
                )
            elif command == "LV":
                cur_coord[1] += coords[0]
                contour.push(
                    coordinates=cur_coord.copy(), in_handle=None, out_handle=None
                )
            elif command == "N":
                cur_coord += np.array(coords[0:2], dtype=np.float32)
                in_handle = cur_coord + np.array(coords[2:4], dtype=np.float32)
                out_handle = cur_coord + np.array(coords[4:6], dtype=np.float32)
                contour.push(
                    coordinates=cur_coord.copy(),
                    in_handle=in_handle,
                    out_handle=out_handle,
                )
            elif command == "NS":
                cur_coord += np.array(coords[0:2], dtype=np.float32)
                angle = coords[2]
                length_in = coords[3]
                length_out = coords[4]
                # Reconstruct handles based on the convention:
                # angle is for the outgoing handle, incoming is opposite.
                out_handle = np.array(
                    [
                        cur_coord[0] + length_out * np.cos(angle),
                        cur_coord[1] + length_out * np.sin(angle),
                    ],
                    dtype=np.float32,
                )
                in_handle = np.array(
                    [
                        cur_coord[0] - length_in * np.cos(angle),
                        cur_coord[1] - length_in * np.sin(angle),
                    ],
                    dtype=np.float32,
                )
                contour.push(
                    coordinates=cur_coord.copy(),
                    in_handle=in_handle,
                    out_handle=out_handle,
                )
            elif command == "NH":
                cur_coord += np.array(coords[0:2], dtype=np.float32)
                in_handle = cur_coord + np.array([coords[2], 0.0], dtype=np.float32)
                out_handle = cur_coord + np.array([coords[3], 0.0], dtype=np.float32)
                contour.push(
                    coordinates=cur_coord.copy(),
                    in_handle=in_handle,
                    out_handle=out_handle,
                )
            elif command == "NV":
                cur_coord += np.array(coords[0:2], dtype=np.float32)
                in_handle = cur_coord + np.array([0.0, coords[2]], dtype=np.float32)
                out_handle = cur_coord + np.array([0.0, coords[3]], dtype=np.float32)
                contour.push(
                    coordinates=cur_coord.copy(),
                    in_handle=in_handle,
                    out_handle=out_handle,
                )
            elif command == "NCI":
                cur_coord += np.array(coords[0:2], dtype=np.float32)
                in_handle = cur_coord + np.array(coords[2:4], dtype=np.float32)
                contour.push(
                    coordinates=cur_coord.copy(),
                    in_handle=in_handle,
                    out_handle=None,
                )
            elif command == "NCO":
                cur_coord += np.array(coords[0:2], dtype=np.float32)
                out_handle = cur_coord + np.array(coords[2:4], dtype=np.float32)
                contour.push(
                    coordinates=cur_coord.copy(),
                    in_handle=None,
                    out_handle=out_handle,
                )
            elif command == "EOS":
                # End of sequence, do nothing
                pass
            elif not tolerant:
                raise ValueError(f"Unsupported Node command: {command}")
        return contour

    @classmethod
    def image_space_to_mask_space(cls, sequence, box: Float[torch.Tensor, "4"]):
        """
        Normalizes a sequence's image-space coordinates to the model's internal
        [-1, 1] range relative to a given bounding box. This version is vectorized,
        differentiates between absolute (M) and relative (other) coordinates,
        and avoids in-place operations to be torch.compile-friendly.
        """
        commands, coords_img_space = cls.split_tensor(sequence)
        x1, y1, x2, y2 = box
        width = torch.clamp(x2 - x1, min=1.0)
        height = torch.clamp(y2 - y1, min=1.0)
        avg_dim = (width + height) / 2.0

        command_indices = torch.argmax(commands, dim=-1)
        m_mask = (command_indices == NodeCommand.encode_command("M")).unsqueeze(1)
        lv_mask = (command_indices == NodeCommand.encode_command("LV")).unsqueeze(1)
        ns_mask = (command_indices == NodeCommand.encode_command("NS")).unsqueeze(1)

        # --- Calculate treatment for ALL coordinates as if they were RELATIVE deltas ---
        coord_width = coords_img_space.shape[1]
        scale_vec_rel = torch.tensor(
            [2.0 / width, 2.0 / height] * (coord_width // 2)
            + [2.0 / width] * (coord_width % 2),
            device=sequence.device,
            dtype=sequence.dtype,
        )
        # LV command's first (and only) coord is a dY, so it should be scaled by height.
        scale_vec_lv = scale_vec_rel.clone()
        if coord_width > 0:
            scale_vec_lv[0] = 2.0 / height

        # NS has special scaling for angle and lengths
        scale_vec_ns = scale_vec_rel.clone()
        if coord_width > 4:
            scale_vec_ns[2] = 1.0 / np.pi  # Angle -> [-1, 1]
            scale_vec_ns[3] = 2.0 / avg_dim  # Lengths
            scale_vec_ns[4] = 2.0 / avg_dim

        relative_result = coords_img_space * scale_vec_rel
        relative_result = torch.where(
            lv_mask, coords_img_space * scale_vec_lv, relative_result
        )
        relative_result = torch.where(
            ns_mask, coords_img_space * scale_vec_ns, relative_result
        )

        # --- Calculate treatment for ALL coordinates as if they were ABSOLUTE positions ---
        absolute_result = coords_img_space.clone()
        # Translate
        absolute_result[:, 0] -= x1
        absolute_result[:, 1] -= y1
        # Scale to [0,1]
        absolute_result[:, 0] /= width
        absolute_result[:, 1] /= height
        # Shift to [-1,1]
        absolute_result = (absolute_result * 2) - 1

        # --- Combine ---
        # Where the command is 'M', use the absolute result, otherwise use the relative result.
        normalized_coords = torch.where(m_mask, absolute_result, relative_result)

        return torch.cat([commands, normalized_coords], dim=-1)

    @classmethod
    def mask_space_to_image_space(cls, sequence, box):
        """
        Denormalizes a sequence's [-1, 1] coordinates back to image space.
        This version is vectorized, differentiates between absolute (M) and
        relative (other) coordinates, and avoids in-place operations to be
        torch.compile-friendly.
        """
        commands, coords_norm = cls.split_tensor(sequence)
        x1, y1, x2, y2 = box
        width = torch.clamp(x2 - x1, min=1.0)
        height = torch.clamp(y2 - y1, min=1.0)
        avg_dim = (width + height) / 2.0

        command_indices = torch.argmax(commands, dim=-1)
        m_mask = (command_indices == NodeCommand.encode_command("M")).unsqueeze(1)
        lv_mask = (command_indices == NodeCommand.encode_command("LV")).unsqueeze(1)
        ns_mask = (command_indices == NodeCommand.encode_command("NS")).unsqueeze(1)

        # --- Handle Relative Coordinates (Scaling from [-1, 1] space) ---
        coord_width = coords_norm.shape[1]
        # Deltas in [-1,1] space should be scaled by width/2 to be correct in image space
        scale_vec_rel = torch.tensor(
            [width / 2.0, height / 2.0] * (coord_width // 2)
            + [width / 2.0] * (coord_width % 2),
            device=sequence.device,
            dtype=sequence.dtype,
        )
        # LV command's first (and only) coord is a dY, so it should be scaled by height.
        scale_vec_lv = scale_vec_rel.clone()
        if coord_width > 0:
            scale_vec_lv[0] = height / 2.0

        # NS has special scaling for angle and lengths
        scale_vec_ns = scale_vec_rel.clone()
        if coord_width > 4:
            scale_vec_ns[2] = np.pi  # Denormalize angle
            scale_vec_ns[3] = avg_dim / 2.0  # Denormalize lengths
            scale_vec_ns[4] = avg_dim / 2.0

        relative_result = coords_norm * scale_vec_rel
        relative_result = torch.where(
            lv_mask, coords_norm * scale_vec_lv, relative_result
        )
        relative_result = torch.where(
            ns_mask, coords_norm * scale_vec_ns, relative_result
        )

        # --- Handle Absolute 'M' Coordinates (Translation and Scaling from [-1, 1]) ---
        absolute_result = coords_norm.clone()
        # Shift from [-1,1] to [0,1]
        absolute_result = (absolute_result + 1) / 2
        # Scale to image dims
        absolute_result[:, 0] *= width
        absolute_result[:, 1] *= height
        # Translate
        absolute_result[:, 0] += x1
        absolute_result[:, 1] += y1

        # --- Combine ---
        # Where the command is 'M', use the absolute result, otherwise use the relative result.
        denormalized_coords = torch.where(m_mask, absolute_result, relative_result)

        return torch.cat([commands, denormalized_coords], dim=-1)

    @classmethod
    def unroll_relative_coordinates(cls, sequence: torch.Tensor) -> torch.Tensor:
        """
        Converts a sequence with relative coordinates to one with absolute coordinates.
        This is a differentiable and vectorized operation.
        """
        commands, rel_coords = cls.split_tensor(sequence)
        abs_coords = torch.zeros_like(rel_coords)

        command_indices = torch.argmax(commands, dim=-1)

        m_index = cls.encode_command("M")
        l_index = cls.encode_command("L")
        lh_index = cls.encode_command("LH")
        lv_index = cls.encode_command("LV")
        n_index = cls.encode_command("N")
        ns_index = cls.encode_command("NS")
        nh_index = cls.encode_command("NH")
        nv_index = cls.encode_command("NV")
        nci_index = cls.encode_command("NCI")
        nco_index = cls.encode_command("NCO")

        # Mask for commands that have relative XY motion
        relative_xy_mask = (
            (command_indices == l_index)
            | (command_indices == n_index)
            | (command_indices == ns_index)
            | (command_indices == nh_index)
            | (command_indices == nv_index)
            | (command_indices == nci_index)
            | (command_indices == nco_index)
        )

        # Create deltas without in-place assignment to keep dynamo happy
        zeros_like_xy = torch.zeros_like(rel_coords[:, 0:2])

        # Deltas for relative XY commands
        deltas_xy = torch.where(
            relative_xy_mask.unsqueeze(1), rel_coords[:, 0:2], zeros_like_xy
        )

        # Deltas for LH and LV (single-axis moves)
        lh_mask = command_indices == lh_index
        lv_mask = command_indices == lv_index
        lh_vec = torch.where(
            lh_mask, rel_coords[:, 0], torch.zeros_like(rel_coords[:, 0])
        )
        lv_vec = torch.where(
            lv_mask, rel_coords[:, 0], torch.zeros_like(rel_coords[:, 0])
        )

        # Combine into a single delta tensor
        deltas = torch.stack(
            (deltas_xy[:, 0] + lh_vec, deltas_xy[:, 1] + lv_vec), dim=1
        )

        # Seed absolute positions with the absolute move from the M command (vectorized)
        m_mask = command_indices == m_index
        base_pos = torch.where(
            m_mask.unsqueeze(1), rel_coords[:, 0:2], torch.zeros_like(deltas)
        )

        # Calculate absolute positions with cumsum
        abs_positions = torch.cumsum(deltas + base_pos, dim=0)

        # Mask for commands that have position coordinates
        has_pos_coords_mask = (
            m_mask
            | (command_indices == l_index)
            | (command_indices == lh_index)
            | (command_indices == lv_index)
            | (command_indices == n_index)
            | (command_indices == ns_index)
            | (command_indices == nh_index)
            | (command_indices == nv_index)
            | (command_indices == nci_index)
            | (command_indices == nco_index)
        )

        # Build absolute coordinates column-wise without in-place masked writes
        zeros_scalar = torch.zeros_like(rel_coords[:, 0])

        # Position columns (common to all node types)
        pos_x = torch.where(has_pos_coords_mask, abs_positions[:, 0], zeros_scalar)
        pos_y = torch.where(has_pos_coords_mask, abs_positions[:, 1], zeros_scalar)

        # --- Handle Calculation ---
        in_x = torch.zeros_like(pos_x)
        in_y = torch.zeros_like(pos_y)
        out_x = torch.zeros_like(pos_x)
        out_y = torch.zeros_like(pos_y)

        # Standard Node (N)
        n_mask = command_indices == n_index
        in_x = torch.where(n_mask, abs_positions[:, 0] + rel_coords[:, 2], in_x)
        in_y = torch.where(n_mask, abs_positions[:, 1] + rel_coords[:, 3], in_y)
        out_x = torch.where(n_mask, abs_positions[:, 0] + rel_coords[:, 4], out_x)
        out_y = torch.where(n_mask, abs_positions[:, 1] + rel_coords[:, 5], out_y)

        # Smooth Node (NS)
        ns_mask = command_indices == ns_index
        ns_angle = rel_coords[:, 2]
        ns_len_in = rel_coords[:, 3]
        ns_len_out = rel_coords[:, 4]
        cos_angle = torch.cos(ns_angle)
        sin_angle = torch.sin(ns_angle)

        ns_in_x = abs_positions[:, 0] - ns_len_in * cos_angle
        ns_in_y = abs_positions[:, 1] - ns_len_in * sin_angle
        ns_out_x = abs_positions[:, 0] + ns_len_out * cos_angle
        ns_out_y = abs_positions[:, 1] + ns_len_out * sin_angle

        in_x = torch.where(ns_mask, ns_in_x, in_x)
        in_y = torch.where(ns_mask, ns_in_y, in_y)
        out_x = torch.where(ns_mask, ns_out_x, out_x)
        out_y = torch.where(ns_mask, ns_out_y, out_y)

        # Other node types (NH, NV, NCI, NCO)
        nh_mask = command_indices == nh_index
        nv_mask = command_indices == nv_index
        nci_mask = command_indices == nci_index
        nco_mask = command_indices == nco_index

        # Handles for NH
        in_x = torch.where(nh_mask, abs_positions[:, 0] + rel_coords[:, 2], in_x)
        in_y = torch.where(nh_mask, abs_positions[:, 1], in_y)
        out_x = torch.where(nh_mask, abs_positions[:, 0] + rel_coords[:, 3], out_x)
        out_y = torch.where(nh_mask, abs_positions[:, 1], out_y)

        # Handles for NV
        in_x = torch.where(nv_mask, abs_positions[:, 0], in_x)
        in_y = torch.where(nv_mask, abs_positions[:, 1] + rel_coords[:, 2], in_y)
        out_x = torch.where(nv_mask, abs_positions[:, 0], out_x)
        out_y = torch.where(nv_mask, abs_positions[:, 1] + rel_coords[:, 3], out_y)

        # Handles for NCI (in-handle only)
        in_x = torch.where(nci_mask, abs_positions[:, 0] + rel_coords[:, 2], in_x)
        in_y = torch.where(nci_mask, abs_positions[:, 1] + rel_coords[:, 3], in_y)

        # Handles for NCO (out-handle only)
        out_x = torch.where(nco_mask, abs_positions[:, 0] + rel_coords[:, 2], out_x)
        out_y = torch.where(nco_mask, abs_positions[:, 1] + rel_coords[:, 3], out_y)

        abs_coords = torch.stack((pos_x, pos_y, in_x, in_y, out_x, out_y), dim=1)

        return torch.cat([commands, abs_coords], dim=1)

    @classmethod
    def tensors_to_segments(cls, cmd, coord):
        """Convert an encoded command and coordinate tensor to segment points
        and control point counts for the diffvg renderer.

        This almost certainly needs rewriting for the current representation.
        """

        command_tensor = torch.argmax(cmd, dim=-1)

        all_points = []
        all_num_cp = []
        contour_splits = []
        point_splits = []

        contour_nodes = []
        cmd_sos_val = cls.encode_command("SOS")
        cmd_eos_val = cls.encode_command("EOS")
        cmd_n_val = cls.encode_command("N")

        for i in range(len(command_tensor)):
            command = command_tensor[i]
            is_sos = command == cmd_sos_val
            is_eos = command == cmd_eos_val

            if is_sos:
                # Skip SOS token, it just marks the start
                continue
            elif is_eos:
                if len(contour_nodes) > 0:
                    # Process the collected contour
                    # First point is the start point
                    all_points.append(contour_nodes[0][1][0:2])

                    # Segments from node to node
                    for j in range(len(contour_nodes)):
                        p1_cmd, p1_coord = contour_nodes[j]
                        p2_cmd, p2_coord = contour_nodes[(j + 1) % len(contour_nodes)]

                        is_curve = (p1_cmd == cmd_n_val) and (p2_cmd == cmd_n_val)
                        p1_pos, p2_pos = p1_coord[0:2], p2_coord[0:2]
                        # Handle positions are absolute.
                        p1_hout = p1_coord[4:6]
                        p2_hin = p2_coord[2:4]
                        # If they change to relative, use this instead:
                        # p1_hout, p2_hin = p1_pos + p1_coord[4:6], p2_pos + p2_coord[2:4]

                        if is_curve:
                            all_points.extend([p1_hout, p2_hin, p2_pos])
                            all_num_cp.append(2)
                        else:
                            all_points.append(p2_pos)
                            all_num_cp.append(0)

                    contour_splits.append(len(all_num_cp))
                    point_splits.append(len(all_points))

                contour_nodes = []
                break
            else:
                # It's a node, add it to the current contour
                contour_nodes.append((command, coord[i]))

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


class TangentNormalCommand(CommandRepresentation):
    """
    A command representation where coordinates are relative to a local
    tangent-normal basis at each node.
    'M' is still in absolute world-space.
    'L' coordinates are (forward, sideways) relative to the previous tangent.
    """

    grammar = {
        "SOS": 0,
        "M": 2,
        "L": 2,
        "N": 6,
        "EOS": 0,
    }

    @classmethod
    def emit(cls, nodes: List["Node"]) -> Sequence["TangentNormalCommand"]:
        commands = []
        if not nodes:
            return commands

        commands.append(cls("SOS", []))
        commands.append(cls("M", nodes[0].coordinates.tolist()))

        # Initial basis: pointing right
        f_hat = np.array([1.0, 0.0], dtype=np.float32)

        for i in range(len(nodes)):
            p_prev = nodes[i]
            p_curr = nodes[(i + 1) % len(nodes)]  # Loop back to start

            # World-space delta for the on-curve point
            delta_pos = p_curr.coordinates - p_prev.coordinates

            # Project on-curve delta onto current basis
            r_hat = np.array([-f_hat[1], f_hat[0]], dtype=np.float32)
            turtle_dx = np.dot(delta_pos, f_hat)
            turtle_dy = np.dot(delta_pos, r_hat)

            # Determine the basis for the *next* step.
            # The new "forward" is the direction of the outgoing handle of the *current* point `p_prev`,
            # or the line direction if it was a line.
            if p_prev.out_handle is not None:
                next_f_hat_vec = p_prev.out_handle - p_prev.coordinates
            else:  # is_line
                next_f_hat_vec = p_curr.coordinates - p_prev.coordinates

            norm = np.linalg.norm(next_f_hat_vec)
            if norm > 1e-6:
                next_f_hat = next_f_hat_vec / norm
            else:
                next_f_hat = f_hat  # Keep current basis if segment is zero-length

            # A segment is a line if the prev node has no out-handle and curr node has no in-handle
            if p_prev.out_handle is None and p_curr.in_handle is None:
                commands.append(cls("L", [turtle_dx, turtle_dy]))
            else:
                # Project handles onto the same basis
                # We use the handles relative to their on-curve points
                in_handle_rel = (
                    p_curr.in_handle - p_curr.coordinates
                    if p_curr.in_handle is not None
                    else np.array([0, 0])
                )
                # The out-handle belongs to the *previous* node
                out_handle_rel = (
                    p_prev.out_handle - p_prev.coordinates
                    if p_prev.out_handle is not None
                    else np.array([0, 0])
                )

                h_in_f = np.dot(in_handle_rel, f_hat)
                h_in_s = np.dot(in_handle_rel, r_hat)
                h_out_f = np.dot(out_handle_rel, f_hat)
                h_out_s = np.dot(out_handle_rel, r_hat)

                commands.append(
                    cls("N", [turtle_dx, turtle_dy, h_in_f, h_in_s, h_out_f, h_out_s])
                )

            f_hat = next_f_hat

        commands.append(cls("EOS", []))
        return commands

    @classmethod
    def contour_from_commands(
        cls, commands: Sequence[CommandRepresentation], tolerant=True
    ) -> "NodeContour":
        contour = NodeContour([])
        commands = list(commands)
        if not commands or commands[0].command != "SOS":
            return contour

        commands.pop(0)
        if not commands or commands[0].command != "M":
            return contour

        current_pos = np.array(commands[0].coordinates, dtype=np.float32)
        # The first node has no handles from the M command
        prev_node = contour.push(
            coordinates=current_pos.copy(), in_handle=None, out_handle=None
        )
        commands.pop(0)

        # Initial basis: pointing right
        f_hat = np.array([1.0, 0.0], dtype=np.float32)

        for command in commands:
            if command.command == "EOS":
                break

            r_hat = np.array([-f_hat[1], f_hat[0]], dtype=np.float32)

            if command.command == "L":
                forward, sideways = command.coordinates
                delta_world = forward * f_hat + sideways * r_hat
                current_pos += delta_world
                prev_node = contour.push(
                    coordinates=current_pos.copy(), in_handle=None, out_handle=None
                )

                # Update basis from the line segment itself
                norm = np.linalg.norm(delta_world)
                if norm > 1e-6:
                    f_hat = delta_world / norm

            elif command.command == "N":
                turtle_dx, turtle_dy, h_in_f, h_in_s, h_out_f, h_out_s = (
                    command.coordinates
                )

                # Reconstruct world-space deltas from the turtle coordinates
                delta_pos = turtle_dx * f_hat + turtle_dy * r_hat
                delta_in_handle = h_in_f * f_hat + h_in_s * r_hat
                delta_out_handle = h_out_f * f_hat + h_out_s * r_hat

                # Set the out-handle of the PREVIOUS node
                prev_node.out_handle = prev_node.coordinates + delta_out_handle

                # Create the new node
                current_pos += delta_pos
                new_in_handle = current_pos + delta_in_handle
                prev_node = contour.push(
                    coordinates=current_pos.copy(),
                    in_handle=new_in_handle,
                    out_handle=None,
                )

                # Update basis from the outgoing handle we just computed for the previous node
                norm = np.linalg.norm(delta_out_handle)
                if norm > 1e-6:
                    f_hat = delta_out_handle / norm
        if (
            len(contour.nodes) > 1
            and np.allclose(
                contour.nodes[0].coordinates, contour.nodes[-1].coordinates, atol=1e-4
            )
        ):
            last_node = contour.nodes.pop()
            contour.nodes[0].in_handle = last_node.in_handle
        return contour
