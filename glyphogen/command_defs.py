from abc import ABC
from typing import List, Optional, Self, Sequence, Union, TYPE_CHECKING
from jaxtyping import Float

import numpy as np
import numpy.typing as npt
import torch

from glyphogen.nodeglyph import Node, NodeContour

MAX_COORDINATE = 1000

# Implementation node: There's an important notational convention we're using
# in the relative coordinate representations. We want all node commands to use
# relative coordinates. *All* of them, so that the model doesn't have to do
# gymnastics to understand that node 0's coordinates have a different interpretation
# to all the other nodes in the sequence. But then how do you establish initial
# position? The convention is that the first node in a contour gets emitted as
# two separate commands. First, an absolute "Move to" command to set the
# initial position, then the first node itself is emitted as a relative
# command with a delta of (0,0).


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


class RelativePolarCoordinateRepresentation(CoordinateRepresentation):
    """Coordinates are represented in polar form relative to the previous node.
    Handles are also relative to the current node position, in polar form."""

    @classmethod
    def emit_node_position(cls, n: "Node") -> npt.NDArray[np.float32]:
        if n.index == 0:
            # Definitionally this is 0,0 because we emit an explicit move to absolute position first
            return np.array([0.0, 0.0], dtype=np.float32)
        previous_node = n.previous
        delta = n.coordinates - previous_node.coordinates
        r = np.linalg.norm(delta)
        theta = np.arctan2(delta[1], delta[0])
        return np.array([r, theta], dtype=np.float32)

    @classmethod
    def emit_in_handle(cls, n: "Node") -> Optional[npt.NDArray[np.float32]]:
        if n.in_handle is None:
            return None
        delta = n.in_handle - n.coordinates
        r = np.linalg.norm(delta)
        theta = np.arctan2(delta[1], delta[0])
        return np.array([r, theta], dtype=np.float32)

    @classmethod
    def emit_out_handle(cls, n: "Node") -> Optional[npt.NDArray[np.float32]]:
        if n.out_handle is None:
            return None
        delta = n.out_handle - n.coordinates
        r = np.linalg.norm(delta)
        theta = np.arctan2(delta[1], delta[0])
        return np.array([r, theta], dtype=np.float32)


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


class SVGCommand(CommandRepresentation):
    grammar = {
        "M": 2,  # Move to absolute (x, y)
        "L": 2,  # Line to absolute (x, y)
        "C": 6,  # Cubic Bezier: (c1x, c1y, c2x, c2y, x, y)
        "Z": 0,  # Close path
    }
    coordinate_representation = AbsoluteCoordinateRepresentation

    @classmethod
    def emit(cls, nodes: List["Node"]) -> Sequence["SVGCommand"]:
        commands: List[SVGCommand] = []
        if not nodes:
            return commands
        commands.append(cls("M", nodes[0].coordinates.tolist()))
        for i in range(len(nodes)):
            prev = nodes[i]
            curr = nodes[(i + 1) % len(nodes)]
            if curr == nodes[0]:
                # Closing segment
                if prev.is_line:
                    commands.append(cls("L", curr.coordinates.tolist()))
                else:
                    assert prev.out_handle is not None and curr.in_handle is not None
                    c = [
                        float(prev.out_handle[0]),
                        float(prev.out_handle[1]),
                        float(curr.in_handle[0]),
                        float(curr.in_handle[1]),
                        float(curr.coordinates[0]),
                        float(curr.coordinates[1]),
                    ]
                    commands.append(cls("C", c))
                commands.append(cls("Z", []))
                break
            if curr.is_line:
                commands.append(cls("L", curr.coordinates.tolist()))
            else:
                assert prev.out_handle is not None and curr.in_handle is not None
                c = [
                    float(prev.out_handle[0]),
                    float(prev.out_handle[1]),
                    float(curr.in_handle[0]),
                    float(curr.in_handle[1]),
                    float(curr.coordinates[0]),
                    float(curr.coordinates[1]),
                ]
                commands.append(cls("C", c))
        return commands

    @classmethod
    def contour_from_commands(
        cls, commands: Sequence["SVGCommand"], tolerant: bool = True
    ) -> "NodeContour":
        contour = NodeContour([])
        commands = list(commands)
        if not commands or commands[0].command != "M":
            return contour
        current_pos = np.array(commands[0].coordinates, dtype=np.float32)
        prev_node = contour.push(
            coordinates=current_pos.copy(), in_handle=None, out_handle=None
        )

        for step in commands[1:]:
            cmd = step.command
            if cmd == "L":
                current_pos = np.array(step.coordinates, dtype=np.float32)
                prev_node = contour.push(
                    coordinates=current_pos.copy(), in_handle=None, out_handle=None
                )
            elif cmd == "C":
                c1x, c1y, c2x, c2y, x, y = step.coordinates
                prev_node.out_handle = np.array([c1x, c1y], dtype=np.float32)
                current_pos = np.array([x, y], dtype=np.float32)
                in_h = np.array([c2x, c2y], dtype=np.float32)
                prev_node = contour.push(
                    coordinates=current_pos.copy(), in_handle=in_h, out_handle=None
                )
            elif cmd == "Z":
                break
            else:
                if not tolerant:
                    raise ValueError(f"Unsupported SVG command: {cmd}")

        if len(contour.nodes) > 1 and np.allclose(
            contour.nodes[0].coordinates, contour.nodes[-1].coordinates, atol=1e-4
        ):
            last_node = contour.nodes.pop()
            contour.nodes[0].in_handle = last_node.in_handle
        return contour


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
        # Emit move to for the first node
        first_node = nodes[0]
        # Use AbsoluteCoordinateRepresentation for the M command
        pos = AbsoluteCoordinateRepresentation.emit_node_position(first_node)
        commands.append(cls("M", pos.tolist()))

        f_hat = np.array([1.0, 0.0], dtype=np.float32)

        for i in range(len(nodes)):
            p_prev = nodes[i]
            p_curr = nodes[(i + 1) % len(nodes)]

            delta_pos = p_curr.coordinates - p_prev.coordinates
            r = float(np.linalg.norm(delta_pos))
            r_hat = np.array([-f_hat[1], f_hat[0]], dtype=np.float32)

            if r > 1e-6:
                dir_chord = delta_pos / r
                phi = float(
                    np.arctan2(np.dot(delta_pos, r_hat), np.dot(delta_pos, f_hat))
                )
            else:
                dir_chord = f_hat
                phi = 0.0

            # Check for 90-degree turns
            if np.isclose(phi, np.pi / 2):
                commands.append(cls("L_LEFT_90", [r]))
            elif np.isclose(phi, -np.pi / 2):
                commands.append(cls("L_RIGHT_90", [r]))
            else:
                commands.append(cls("L_POLAR", [r, phi]))

            # Keep the frame fixed for tangent-encoded segments so that subsequent
            # commands interpret phi in the same basis; otherwise rotate to chord.
            # For now, we always rotate to chord for line segments.
            f_hat = cls._unit(delta_pos, f_hat)

        commands.append(cls("EOS", []))
        return commands

    @classmethod
    def contour_from_commands(
        cls, commands: Sequence[CommandRepresentation], tolerant: bool = True
    ) -> "NodeContour":
        from .nodeglyph import NodeContour  # avoid circular import

        contour = NodeContour([])
        all_commands = list(commands)
        if not all_commands or all_commands[0].command != "SOS":
            return contour
        if len(all_commands) < 2 or all_commands[1].command != "M":
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

        current_pos = np.array(all_commands[1].coordinates, dtype=np.float32)
        prev_node = contour.push(
            coordinates=current_pos.copy(), in_handle=None, out_handle=None
        )

        for idx, command in enumerate(all_commands[2:], start=2):
            if command.command == "EOS":
                break

            pos_abs = abs_coords_np[idx, 0:2]
            in_abs = abs_coords_np[idx, 2:4]
            out_abs = abs_coords_np[idx, 4:6]

            if command.command == "L_POLAR":
                r, _ = command.coordinates
                if abs(r) < 1e-6:
                    continue
                prev_node = contour.push(
                    coordinates=pos_abs.copy(), in_handle=None, out_handle=None
                )
            elif command.command == "L_LEFT_90":
                r = command.coordinates[0]
                if abs(r) < 1e-6:
                    continue
                prev_node = contour.push(
                    coordinates=pos_abs.copy(), in_handle=None, out_handle=None
                )
            elif command.command == "L_RIGHT_90":
                r = command.coordinates[0]
                if abs(r) < 1e-6:
                    continue
                prev_node = contour.push(
                    coordinates=pos_abs.copy(), in_handle=None, out_handle=None
                )
            else:
                if not tolerant:
                    raise ValueError(f"Unsupported command: {command.command}")

        if len(contour.nodes) > 1 and np.allclose(
            contour.nodes[0].coordinates, contour.nodes[-1].coordinates, atol=1e-4
        ):
            last_node = contour.nodes.pop()
            contour.nodes[0].in_handle = last_node.in_handle
        return contour

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
            r_hat = torch.tensor(
                [-f_hat[1], f_hat[0]], device=sequence.device, dtype=sequence.dtype
            )

            pos_abs = current_pos
            in_abs = current_pos
            out_abs = current_pos

            if idx == cls.encode_command("M"):
                current_pos = co[0:2]
                pos_abs = current_pos
            elif idx == cls.encode_command("L_POLAR"):
                r, phi = co[0], co[1]
                # The phi here is the *relative* angle change from the current f_hat
                # So we rotate f_hat by phi
                rotation_matrix = torch.tensor([
                    [torch.cos(phi), -torch.sin(phi)],
                    [torch.sin(phi), torch.cos(phi)]
                ], device=sequence.device, dtype=sequence.dtype)
                f_hat = torch.matmul(rotation_matrix, f_hat)
                
                delta = r * f_hat
                current_pos = current_pos + delta
                pos_abs = current_pos
            elif idx == cls.encode_command("L_LEFT_90"):
                r = co[0]
                phi = torch.tensor(np.pi / 2, device=sequence.device, dtype=sequence.dtype)
                rotation_matrix = torch.tensor([
                    [torch.cos(phi), -torch.sin(phi)],
                    [torch.sin(phi), torch.cos(phi)]
                ], device=sequence.device, dtype=sequence.dtype)
                f_hat = torch.matmul(rotation_matrix, f_hat)
                delta = r * f_hat
                current_pos = current_pos + delta
                pos_abs = current_pos
            elif idx == cls.encode_command("L_RIGHT_90"):
                r = co[0]
                phi = torch.tensor(-np.pi / 2, device=sequence.device, dtype=sequence.dtype)
                rotation_matrix = torch.tensor([
                    [torch.cos(phi), -torch.sin(phi)],
                    [torch.sin(phi), torch.cos(phi)]
                ], device=sequence.device, dtype=sequence.dtype)
                f_hat = torch.matmul(rotation_matrix, f_hat)
                delta = r * f_hat
                current_pos = current_pos + delta
                pos_abs = current_pos
            # SOS and EOS do not change position or f_hat

            def _zero_small(t: torch.Tensor) -> torch.Tensor:
                return torch.where(torch.abs(t) < 1e-5, torch.zeros_like(t), t)

            abs_coords[i, 0:2] = _zero_small(pos_abs)
            abs_coords[i, 2:4] = _zero_small(in_abs)
            abs_coords[i, 4:6] = _zero_small(out_abs)

        return torch.cat([commands, abs_coords], dim=1)

    @classmethod
    def image_space_to_mask_space(cls, sequence, box: Float[torch.Tensor, "4"]):
        raise NotImplementedError("We will handle this later")

    @classmethod
    def mask_space_to_image_space(cls, sequence, box):
        raise NotImplementedError("We will handle this later")

    @classmethod
    def tensors_to_segments(cls, cmd, coord):
        raise NotImplementedError(
            "We don't need this; use SVGCommand to render for visualization."
        )
