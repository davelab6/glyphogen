from pathlib import Path
from typing import Dict, List

import numpy as np
import numpy.typing as npt
import uharfbuzz as hb
from fontTools.pens.qu2cuPen import Qu2CuPen
from fontTools.pens.svgPathPen import SVGPathPen

from .rendering import render

BASIC_SVG_COMMANDS = {
    "M": 2,  # Move to (2 coordinates)
    "L": 2,  # Line to (2 coordinates)
    "H": 1,  # Horizontal line to (1 coordinate)
    "V": 1,  # Vertical line to (1 coordinate)
    "C": 6,  # Cubic Bezier curve (6 coordinates)
    "Z": 0,  # Close path (no coordinates)
}

NODE_GLYPH_COMMANDS = {
    "SOS": 0,
    "N": 6,  # Node with two handles (x, y, hin_x, hin_y, hout_x, hout_y)
    "NCI": 4,  # Node with curve in, line out (x, y, hin_x, hin_y)
    "NCO": 4,  # Node with line in, curve out (x, y, hout_x, hout_y)
    "L": 2,  # Line node (x, y)
    "Z": 0,  # Close path
    "EOS": 0,
}


NODE_COMMAND_WIDTH = len(NODE_GLYPH_COMMANDS.keys())
COORDINATE_WIDTH = max(NODE_GLYPH_COMMANDS.values())





class SVGCommand:
    grammar = BASIC_SVG_COMMANDS
    command: str
    coordinates: List[int]

    def __init__(self, command: str, coordinates: List[int]):
        if command not in self.grammar:
            raise ValueError(f"Invalid SVG command: {command}")
        if len(coordinates) != self.grammar[command]:
            raise ValueError(
                f"Invalid number of coordinates for command {command}: expected {self.grammar[command]}, got {len(coordinates)}"
            )
        self.command = command
        self.coordinates = coordinates


class NodeCommand(SVGCommand):
    grammar = NODE_GLYPH_COMMANDS


class NodeGlyph:
    commands: List[NodeCommand]

    def __init__(self, commands: List[NodeCommand]):
        self.commands = commands

    def encode(self) -> npt.NDArray[np.int_]:
        # We one-hot encode the commands
        command_width = len(NodeCommand.grammar.keys())
        # And we pad the coordinates to a fixed length
        max_coordinates = max(NodeCommand.grammar.values())
        output: List[List[int]] = []

        # Add SOS token
        sos_command_vector = [0] * command_width
        sos_command_vector[list(NodeCommand.grammar.keys()).index("SOS")] = 1
        sos_coordinates = [0] * max_coordinates
        output.append(sos_command_vector + sos_coordinates)

        for command in self.commands:
            # One-hot encode the command
            command_vector = [0] * command_width
            command_vector[
                list(NodeCommand.grammar.keys()).index(command.command)
            ] = 1
            # Pad the coordinates
            coordinates = command.coordinates + [0] * (
                max_coordinates - len(command.coordinates)
            )
            output.append(command_vector + coordinates)

        # Add EOS token
        eos_command_vector = [0] * command_width
        eos_command_vector[list(NodeCommand.grammar.keys()).index("EOS")] = 1
        eos_coordinates = [0] * max_coordinates
        output.append(eos_command_vector + eos_coordinates)

        return np.array(output, dtype=np.int_)

    @classmethod
    def from_numpy(cls, command_tensor, coord_tensor):
        commands = []
        command_keys = list(NodeCommand.grammar.keys())
        sos_index = command_keys.index("SOS")
        eos_index = command_keys.index("EOS")

        for i in range(command_tensor.shape[0]):
            command_index = np.argmax(command_tensor[i])

            if command_index == sos_index:
                continue

            if command_index == eos_index:
                break

            command_str = command_keys[command_index]
            # A zero vector may be fed to the model, which will be interpreted as 'SOS'
            # This is padding, so we should stop
            if command_str == 'SOS' and not np.any(coord_tensor[i]):
                break

            num_coords = NodeCommand.grammar[command_str]
            coords = coord_tensor[i, :num_coords].tolist()
            commands.append(NodeCommand(command_str, coords))
        return cls(commands)

    def to_svg_glyph(self) -> "SVGGlyph":
        svg_commands: List[SVGCommand] = []
        if not self.commands:
            return SVGGlyph([])

        start_of_contour = True
        for i, node_cmd in enumerate(self.commands):
            if node_cmd.command == "Z":
                svg_commands.append(SVGCommand("Z", []))
                start_of_contour = True
                continue

            node_pos = node_cmd.coordinates[0:2]
            if start_of_contour:
                svg_commands.append(SVGCommand("M", node_pos))
                start_of_contour = False
            else:
                prev_node_cmd = self.commands[i - 1]
                prev_pos = np.array(prev_node_cmd.coordinates[0:2])
                curr_pos = np.array(node_pos)

                # Outgoing handle from prev
                if prev_node_cmd.command in ["N", "NCO"]:
                    hout = np.array(
                        prev_node_cmd.coordinates[4:6]
                        if prev_node_cmd.command == "N"
                        else prev_node_cmd.coordinates[2:4]
                    )
                    c1 = prev_pos + hout
                else:
                    c1 = prev_pos

                # Incoming handle to curr
                if node_cmd.command in ["N", "NCI"]:
                    hin = np.array(node_cmd.coordinates[2:4])
                    c2 = curr_pos + hin
                else:
                    c2 = curr_pos

                is_line = np.array_equal(c1, prev_pos) and np.array_equal(c2, curr_pos)
                if is_line:
                    svg_commands.append(SVGCommand("L", list(map(int, curr_pos))))
                else:
                    svg_commands.append(
                        SVGCommand(
                            "C",
                            list(
                                map(
                                    int,
                                    [
                                        c1[0],
                                        c1[1],
                                        c2[0],
                                        c2[1],
                                        curr_pos[0],
                                        curr_pos[1],
                                    ],
                                )
                            ),
                        )
                    )
        return SVGGlyph(svg_commands)

    def to_debug_string(self):
        path_data = []
        for cmd in self.commands:
            path_data.append(cmd.command)
            path_data.extend(map(lambda x: str(int(x)), cmd.coordinates))

class SVGGlyph:
    commands: List[SVGCommand]

    def __init__(self, commands):
        if commands and commands[0].command != "M":
            raise ValueError("SVG path must start with a 'M' command")
        self.commands = commands

    def to_node_glyph(self) -> "NodeGlyph":
        if not self.commands:
            return NodeGlyph([])

        contours = []
        current_contour = []
        for command in self.commands:
            current_contour.append(command)
            if command.command == "Z":
                contours.append(current_contour)
                current_contour = []
        if current_contour:
            contours.append(current_contour)

        all_node_commands = []
        for contour in contours:
            node_positions = []
            segments = []

            def add_pos(p):
                if p not in node_positions:
                    node_positions.append(p)
                return node_positions.index(p)

            start_pos = tuple(contour[0].coordinates)
            start_node_idx = add_pos(start_pos)
            current_node_idx = start_node_idx

            for cmd in contour:
                if cmd.command == "M":
                    continue
                elif cmd.command == "L":
                    end_pos = tuple(cmd.coordinates)
                    end_node_idx = add_pos(end_pos)
                    segments.append(
                        {"start": current_node_idx, "end": end_node_idx, "type": "line"}
                    )
                    current_node_idx = end_node_idx
                elif cmd.command == "C":
                    end_pos = tuple(cmd.coordinates[4:6])
                    end_node_idx = add_pos(end_pos)
                    c1 = tuple(cmd.coordinates[0:2])
                    c2 = tuple(cmd.coordinates[2:4])
                    segments.append(
                        {
                            "start": current_node_idx,
                            "end": end_node_idx,
                            "type": "curve",
                            "c1": c1,
                            "c2": c2,
                        }
                    )
                    current_node_idx = end_node_idx
                elif cmd.command == "Z":
                    end_node_idx = start_node_idx
                    segments.append(
                        {"start": current_node_idx, "end": end_node_idx, "type": "line"}
                    )
                    current_node_idx = end_node_idx

            path_nodes = [{"pos": pos, "in": [], "out": []} for pos in node_positions]
            for i, seg in enumerate(segments):
                path_nodes[seg["start"]]["out"].append(i)
                path_nodes[seg["end"]]["in"].append(i)

            contour_node_commands = []
            # This assumes the nodes appear in order in node_positions, which should be true
            # for well-formed glyphs.
            for i, node in enumerate(path_nodes):
                pos = node["pos"]

                in_seg = segments[node["in"][0]] if node["in"] else None
                out_seg = segments[node["out"][0]] if node["out"] else None

                incoming_is_curve = in_seg and in_seg["type"] == "curve"
                outgoing_is_curve = out_seg and out_seg["type"] == "curve"

                hin, hout = np.array([0, 0]), np.array([0, 0])
                if incoming_is_curve:
                    hin = np.array(in_seg["c2"]) - np.array(pos)
                if outgoing_is_curve:
                    hout = np.array(out_seg["c1"]) - np.array(pos)

                coords = [pos[0], pos[1]]
                if incoming_is_curve and outgoing_is_curve:
                    command = "N"
                    coords.extend([hin[0], hin[1], hout[0], hout[1]])
                elif incoming_is_curve and not outgoing_is_curve:
                    command = "NCI"
                    coords.extend([hin[0], hin[1]])
                elif not incoming_is_curve and outgoing_is_curve:
                    command = "NCO"
                    coords.extend([hout[0], hout[1]])
                else:  # both lines
                    command = "L"

                contour_node_commands.append(
                    NodeCommand(command, list(map(int, coords)))
                )
            all_node_commands.extend(contour_node_commands)
            if contour[-1].command == "Z":
                all_node_commands.append(NodeCommand("Z", []))

        return NodeGlyph(all_node_commands)

    def to_svg_string(self):
        path_data = []
        for cmd in self.commands:
            path_data.append(cmd.command)
            path_data.extend(map(lambda x: str(int(x)), cmd.coordinates))
        return " ".join(path_data)


def quantize(p, mod=5):
    return int((p // mod) * mod)


class Glyph:
    font_file: Path
    unicode_id: int
    location: Dict[str, float]  # Coordinates in a variable font's designspace

    def __init__(self, font_file: Path, unicode_id: int, location: Dict[str, float]):
        self.font_file = font_file
        self.unicode_id = unicode_id
        self.location = location

    def rasterize(self, size: int) -> npt.NDArray[np.float64]:
        """
        Rasterizes the glyph at a given size.
        This method should be implemented to convert the glyph into a raster image.
        """
        return render(
            self.font_file,
            variation=self.location,
            text=chr(self.unicode_id),
            target_size=(size, size),
        )

    def vectorize(self) -> SVGGlyph:
        """
        Converts the glyph into a vector representation.
        This method should be implemented to return the glyph's path as an SVGGlyph object.
        """
        blob = hb.Blob.from_file_path(self.font_file)
        face = hb.Face(blob)
        font = hb.Font(face)
        svgpen = SVGPathPen({}, ntos=lambda f: str(quantize(f)))
        pen = Qu2CuPen(svgpen, max_err=5, all_cubic=True)
        if self.location:
            font.set_variations(self.location)
        glyph = font.get_nominal_glyph(self.unicode_id)
        path = []
        if glyph is None:
            return SVGGlyph([])
        font.draw_glyph_with_pen(glyph, pen)
        for command in svgpen._commands:
            cmd = command[0] if command[0] != " " else "L"
            coords = [int(p) for p in command[1:].split()]
            path.append(SVGCommand(cmd, coords))
        return SVGGlyph(path)
