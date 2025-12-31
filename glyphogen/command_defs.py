from abc import ABC
from typing import List, Optional, Self, Sequence, Union

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
    def emit_node_position(cls, n: Node) -> npt.NDArray[np.int_]: ...

    @classmethod
    def emit_in_handle(cls, n: Node) -> Optional[npt.NDArray[np.int_]]: ...

    @classmethod
    def emit_out_handle(cls, n: Node) -> Optional[npt.NDArray[np.int_]]: ...


class AbsoluteCoordinateRepresentation(CoordinateRepresentation):
    @classmethod
    def emit_node_position(cls, n: Node) -> npt.NDArray[np.int_]:
        return n.coordinates

    @classmethod
    def emit_in_handle(cls, n: Node) -> Optional[npt.NDArray[np.int_]]:
        return n.in_handle

    @classmethod
    def emit_out_handle(cls, n: Node) -> Optional[npt.NDArray[np.int_]]:
        return n.out_handle


class AbsolutePositionRelativeHandleRepresentation(CoordinateRepresentation):
    @classmethod
    def emit_node_position(cls, n: Node) -> npt.NDArray[np.int_]:
        return n.coordinates

    @classmethod
    def emit_in_handle(cls, n: Node) -> Optional[npt.NDArray[np.int_]]:
        if n.in_handle is None:
            return None
        return n.in_handle - n.coordinates

    @classmethod
    def emit_out_handle(cls, n: Node) -> Optional[npt.NDArray[np.int_]]:
        if n.out_handle is None:
            return None
        return n.out_handle - n.coordinates


class RelativeCoordinateRepresentation(AbsolutePositionRelativeHandleRepresentation):
    """Handles are also relative to the node position."""

    @classmethod
    def emit_node_position(cls, n: Node) -> npt.NDArray[np.int_]:
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
    def emit(cls, nodes: List[Node]) -> Sequence[Self]: ...

    @classmethod
    def contour_from_commands(cls, commands: Sequence[Self]) -> NodeContour:
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
    def split_tensor(cls, tensor: torch.Tensor) -> Sequence[torch.Tensor]:
        """Splits a tensor of shape (N, command_width + coordinate_width)
        into a tuple of (commands, coordinates) tensors."""
        command_width = cls.command_width
        commands = tensor[:, :command_width]
        coordinates = tensor[:, command_width:]
        return commands, coordinates

    def __init__(self, command: str, coordinates: List[Union[int, float]]):
        if command not in self.grammar:
            raise ValueError(f"Invalid SVG command: {command}")
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
    def emit(cls, nodes: List[Node]) -> Sequence[Self]:
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
    ) -> NodeContour:
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
        "NH": 4,  # Relative horizontal-handle node (dx, dy, dh_in_x, dh_out_x)
        "NV": 4,  # Relative vertical-handle node (dx, dy, dh_in_y, dh_out_y)
        "NCI": 4,  # Relative node with in-handle only (dx, dy, dhix, dhiy)
        "NCO": 4,  # Relative node with out-handle only (dx, dy, dhox, dhoy)
        "EOS": 0,
    }
    coordinate_representation = RelativeCoordinateRepresentation

    @classmethod
    def emit(cls, nodes: List[Node]) -> Sequence["NodeCommand"]:
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
    ) -> NodeContour:
        contour = NodeContour([])
        # Expect SOS
        if not commands or commands[0].command != "SOS" and not tolerant:
            raise ValueError(
                f"NodeCommand sequence must start with an 'SOS' command, found {commands[0].command}."
            )
        # Expect a M
        if not commands or commands[1].command != "M" and not tolerant:
            raise ValueError(
                f"NodeCommand second command must be 'M' command, found {commands[0].command}."
            )
        cur_coord = np.array(commands[1].coordinates[0:2], dtype=np.float32)
        if cur_coord.shape != (2,) and tolerant:
            cur_coord = np.pad(cur_coord, (0, 2 - cur_coord.shape[0]), "constant")
        command = "M"
        for step in commands[2:]:
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
    def image_space_to_mask_space(cls, sequence, box):
        """
        Normalizes a sequence's image-space coordinates to the model's internal
        [-1, 1] range relative to a given bounding box. Handles mixed
        absolute (M) and relative (other) commands. Lives here as it is
        dependent on the interpretation of the command sequences.
        """
        commands, coords_img_space = cls.split_tensor(sequence)

        x1, y1, x2, y2 = box
        width = x2 - x1 if x2 > x1 else 1
        height = y2 - y1 if y2 > y1 else 1

        command_indices = torch.argmax(commands, dim=-1)
        m_index = NodeCommand.encode_command("M")
        is_m_mask = (command_indices == m_index).unsqueeze(1)

        normalized = coords_img_space.clone()

        # Scale all x-like and y-like coordinates by width and height respectively
        # Careful here with LV, its first and only coordinate is y-like
        lv_index = NodeCommand.encode_command("LV")
        is_lv_mask = (command_indices == lv_index).unsqueeze(1)

        # Additionally translate the absolute 'M' coordinates
        if torch.sum(is_m_mask) > 0:
            m_coords = normalized[is_m_mask.squeeze(1)]
            m_coords[:, 0] -= x1
            m_coords[:, 1] -= y1
            normalized[is_m_mask.squeeze(1)] = m_coords

        if coords_img_space.shape[1] > 0:
            if torch.sum(is_lv_mask) > 0:
                lv_coords = normalized[is_lv_mask.squeeze(1)]
                lv_coords[:, 0] /= height
                normalized[is_lv_mask.squeeze(1)] = lv_coords

            non_lv_mask = ~is_lv_mask.squeeze(1)
            if torch.sum(non_lv_mask) > 0:
                non_lv_coords = normalized[non_lv_mask]
                non_lv_coords[:, 0::2] /= width
                non_lv_coords[:, 1::2] /= height
                normalized[non_lv_mask] = non_lv_coords

        # Everything else is relative so our work here is done

        # Final scaling to [-1, 1]
        normalized = (normalized * 2) - 1

        return torch.cat([commands, normalized], dim=-1)

    @classmethod
    def mask_space_to_image_space(cls, sequence, box):
        """
        Denormalizes a sequence's [-1, 1] coordinates back to image space.
        Handles mixed absolute (M) and relative (other) commands. Lives here as it is
        dependent on the interpretation of the command sequences.
        """
        commands, coords_norm = cls.split_tensor(sequence)

        x1, y1, x2, y2 = box
        width = x2 - x1 if x2 > x1 else 1
        height = y2 - y1 if y2 > y1 else 1

        command_indices = torch.argmax(commands, dim=-1)
        m_index = NodeCommand.encode_command("M")
        is_m_mask = (command_indices == m_index).unsqueeze(1)

        # Scale from [-1, 1] to [0, 1]
        coords_0_1 = (coords_norm + 1) / 2

        denormalized = coords_0_1.clone()

        # Translate the absolute 'M' coordinates
        # Scale all x-like and y-like coordinates
        # Careful here with LV, its first and only coordinate is y-like
        lv_index = NodeCommand.encode_command("LV")
        is_lv_mask = (command_indices == lv_index).unsqueeze(1)

        if coords_norm.shape[1] > 0:
            if torch.sum(is_lv_mask) > 0:
                lv_coords = denormalized[is_lv_mask.squeeze(1)]
                lv_coords[:, 0] *= height
                denormalized[is_lv_mask.squeeze(1)] = lv_coords

            non_lv_mask = ~is_lv_mask.squeeze(1)
            if torch.sum(non_lv_mask) > 0:
                non_lv_coords = denormalized[non_lv_mask]
                non_lv_coords[:, 0::2] *= width
                non_lv_coords[:, 1::2] *= height
                denormalized[non_lv_mask] = non_lv_coords

        if torch.sum(is_m_mask) > 0:
            m_coords = denormalized[is_m_mask.squeeze(1)]
            m_coords[:, 0] += x1
            m_coords[:, 1] += y1
            denormalized[is_m_mask.squeeze(1)] = m_coords

        return torch.cat([commands, denormalized], dim=-1)

    @classmethod
    def unroll_relative_coordinates(cls, sequence: torch.Tensor) -> torch.Tensor:
        """
        Converts a sequence with relative coordinates to one with absolute coordinates.
        This is a differentiable operation.
        """
        commands, rel_coords = cls.split_tensor(sequence)
        abs_coords = torch.zeros_like(rel_coords)
        
        current_pos = torch.zeros(2, device=sequence.device, dtype=sequence.dtype)

        command_indices = torch.argmax(commands, dim=-1)
        
        m_index = cls.encode_command("M")
        l_index = cls.encode_command("L")
        lh_index = cls.encode_command("LH")
        lv_index = cls.encode_command("LV")
        n_index = cls.encode_command("N")
        nh_index = cls.encode_command("NH")
        nv_index = cls.encode_command("NV")
        nci_index = cls.encode_command("NCI")
        nco_index = cls.encode_command("NCO")

        for i in range(sequence.shape[0]):
            cmd_idx = command_indices[i]
            
            if cmd_idx == m_index:
                current_pos = rel_coords[i, 0:2]
                abs_coords[i, 0:2] = current_pos
            elif cmd_idx in [l_index, n_index, nh_index, nv_index, nci_index, nco_index]:
                delta = rel_coords[i, 0:2]
                current_pos = current_pos + delta
                abs_coords[i, 0:2] = current_pos
                if cmd_idx == n_index: # Full node
                    abs_coords[i, 2:4] = current_pos + rel_coords[i, 2:4] # In-handle
                    abs_coords[i, 4:6] = current_pos + rel_coords[i, 4:6] # Out-handle
                elif cmd_idx == nci_index: # In-handle only
                    abs_coords[i, 2:4] = current_pos + rel_coords[i, 2:4]
                elif cmd_idx == nco_index: # Out-handle only
                    abs_coords[i, 2:4] = current_pos + rel_coords[i, 2:4] # Note: NCO stores out-handle in coords 2,3
                elif cmd_idx == nh_index: # Horizontal handles
                    abs_coords[i, 2] = current_pos[0] + rel_coords[i, 2] # In-handle x
                    abs_coords[i, 3] = current_pos[0] + rel_coords[i, 3] # Out-handle x
                elif cmd_idx == nv_index: # Vertical handles
                    abs_coords[i, 2] = current_pos[1] + rel_coords[i, 2] # In-handle y
                    abs_coords[i, 3] = current_pos[1] + rel_coords[i, 3] # Out-handle y

            elif cmd_idx == lh_index:
                delta = torch.zeros(2, device=sequence.device, dtype=sequence.dtype)
                delta[0] = rel_coords[i, 0]
                current_pos = current_pos + delta
                abs_coords[i, 0:2] = current_pos
            elif cmd_idx == lv_index:
                delta = torch.zeros(2, device=sequence.device, dtype=sequence.dtype)
                delta[1] = rel_coords[i, 0]
                current_pos = current_pos + delta
                abs_coords[i, 0:2] = current_pos

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
