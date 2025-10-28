from pathlib import Path
from time import sleep
from typing import Dict, List, Optional

import diskcache
import numpy as np
import numpy.typing as npt
from PIL import Image
import torch
import uharfbuzz as hb
from fontTools.pens.qu2cuPen import Qu2CuPen
from fontTools.pens.svgPathPen import SVGPathPen, pointToString
import pathops
from fontTools.ttLib import TTFont
from fontTools.ttLib.removeOverlaps import _simplify


from .hyperparameters import (
    BASE_DIR,
    MAX_COMMANDS,
    MAX_SEQUENCE_LENGTH,
    RASTER_IMG_SIZE,
)
from .rasterizer import rasterize_batch
from .rendering import render
from .command_defs import (
    BASIC_SVG_COMMANDS,
    NODE_GLYPH_COMMANDS,
    SVGCommand,
    NodeCommand,
    MAX_COORDINATE,
)
from .model import unroll_relative_coords


class AbsoluteSVGPathPen(SVGPathPen):
    def _lineTo(self, pt):
        x, y = pt
        # duplicate point
        if x == self._lastX and y == self._lastY:
            return
        # write the string
        t = "L" + " " + pointToString(pt, self._ntos)
        self._lastCommand = "L"
        self._commands.append(t)
        # store for future reference
        self._lastX, self._lastY = pt


class NodeContour:
    nodes: List["Node"]

    def __init__(self, nodes: List["Node"]):
        self.nodes = nodes

    @property
    def commands(self) -> List[NodeCommand]:
        if not self.nodes:
            return []

        node_commands = [self.nodes[0].command()]
        for i in range(1, len(self.nodes)):
            node_commands.append(self.nodes[i].command(previous_node=self.nodes[i - 1]))
        return node_commands

    @property
    def svg_commands(self) -> List[SVGCommand]:
        svg_commands: List[SVGCommand] = []
        if not self.nodes:
            return []

        svg_commands.append(SVGCommand("M", self.nodes[0].coordinates.tolist()))

        for i in range(len(self.nodes)):
            p1 = self.nodes[i]
            p2 = self.nodes[(i + 1) % len(self.nodes)]

            c1 = p1.out_handle if p1.out_handle is not None else p1.coordinates
            c2 = p2.in_handle if p2.in_handle is not None else p2.coordinates

            is_line = np.array_equal(c1, p1.coordinates) and np.array_equal(
                c2, p2.coordinates
            )
            if is_line:
                if i == len(self.nodes) - 1:
                    pass
                else:
                    svg_commands.append(SVGCommand("L", p2.coordinates.tolist()))
            else:
                svg_commands.append(
                    SVGCommand("C", np.concatenate((c1, c2, p2.coordinates)).tolist())
                )
        svg_commands.append(SVGCommand("Z", []))
        return svg_commands

    @classmethod
    def from_svg_contour(cls, contour: List[SVGCommand]) -> "NodeContour":
        contour = contour[:-1]  # Drop Z

        node_positions = []
        segments = []

        def add_pos(p):
            p = tuple(p)
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
            else:
                # Handle unexpected command types
                raise ValueError(f"Unexpected SVG command: {cmd.command}")

        segments.append(
            {"start": current_node_idx, "end": start_node_idx, "type": "line"}
        )

        path_nodes = [{"pos": pos, "in": [], "out": []} for pos in node_positions]
        for i, seg in enumerate(segments):
            path_nodes[seg["start"]]["out"].append(i)
            path_nodes[seg["end"]]["in"].append(i)

        nodes = []
        contour_obj = cls(nodes)
        for i, node_info in enumerate(path_nodes):
            pos = np.array(node_info["pos"])
            in_seg = segments[node_info["in"][0]] if node_info["in"] else None
            out_seg = segments[node_info["out"][0]] if node_info["out"] else None

            in_handle = None
            if in_seg and in_seg["type"] == "curve":
                in_handle = np.array(in_seg["c2"])

            out_handle = None
            if out_seg and out_seg["type"] == "curve":
                out_handle = np.array(out_seg["c1"])

            nodes.append(Node(pos, contour_obj, in_handle, out_handle))

        return contour_obj

    def push_command(self, command: NodeCommand):
        prev_node = self.nodes[-1] if len(self.nodes) > 0 else None
        position = np.array([0, 0])
        in_handle: Optional[np.ndarray] = None
        out_handle: Optional[np.ndarray] = None

        if command.command == "L":
            position = np.array(command.coordinates)
        elif command.command == "LH":
            if prev_node is not None:
                position = np.array(
                    [
                        prev_node.coordinates[0] + command.coordinates[0],
                        prev_node.coordinates[1],
                    ]
                )
            else:
                # We are lenient here because we are coming from the model and want to make as many
                # sequences as possible work
                return  # raise ValueError("Invalid LH command: no previous node")
        elif command.command == "LV":
            if prev_node is not None:
                position = np.array(
                    [
                        prev_node.coordinates[0],
                        prev_node.coordinates[1] + command.coordinates[0],
                    ]
                )
            else:
                # See above
                return  # raise ValueError("Invalid LV command: no previous node")
        elif command.command == "NH":
            position = np.array(command.coordinates[0:2])
            in_handle = position + np.array([command.coordinates[2], 0])
            out_handle = position + np.array([command.coordinates[3], 0])
        elif command.command == "NV":
            position = np.array(command.coordinates[0:2])
            in_handle = position + np.array([0, command.coordinates[2]])
            out_handle = position + np.array([0, command.coordinates[3]])
        elif command.command == "NCI":
            position = np.array(command.coordinates[0:2])
            in_handle = position + np.array(command.coordinates[2:4])
        elif command.command == "NCO":
            position = np.array(command.coordinates[0:2])
            out_handle = position + np.array(command.coordinates[2:4])
        elif command.command == "N":
            position = np.array(command.coordinates[0:2])
            in_handle = position + np.array(command.coordinates[2:4])
            out_handle = position + np.array(command.coordinates[4:6])
        else:
            # Shouldn't really be here
            return
        self.nodes.append(Node(position, self, in_handle, out_handle))

    def roll(self, shift: int):
        """
        Performs a circular shift on the nodes in the contour.
        """
        if not self.nodes:
            return
        shift = shift % len(self.nodes)
        self.nodes = self.nodes[shift:] + self.nodes[:shift]

    def normalize(self) -> None:
        index_of_bottom_left = min(
            range(len(self.nodes)),
            key=lambda i: (self.nodes[i].coordinates[0], self.nodes[i].coordinates[1]),
        )
        self.nodes = (
            self.nodes[index_of_bottom_left:] + self.nodes[:index_of_bottom_left]
        )


class Node:
    coordinates: npt.NDArray[np.int_]
    in_handle: Optional[npt.NDArray[np.int_]]
    out_handle: Optional[npt.NDArray[np.int_]]

    _contour: NodeContour

    ALIGNMENT_EPSILON = 3

    def __init__(self, coordinates, contour, in_handle=None, out_handle=None):
        self.coordinates = np.array(coordinates)
        self.in_handle = np.array(in_handle) if in_handle is not None else None
        self.out_handle = np.array(out_handle) if out_handle is not None else None
        self._contour = contour

    @property
    def index(self) -> int:
        return self._contour.nodes.index(self)

    @property
    def next(self) -> "Node":
        return self._contour.nodes[(self.index + 1) % len(self._contour.nodes)]

    @property
    def previous(self) -> "Node":
        return self._contour.nodes[self.index - 1]

    @property
    def handles_horizontal(self) -> bool:
        return (
            self.in_handle is not None
            and self.out_handle is not None
            and abs(self.in_handle[1] - self.coordinates[1]) <= self.ALIGNMENT_EPSILON
            and abs(self.coordinates[1] - self.out_handle[1]) <= self.ALIGNMENT_EPSILON
        )

    @property
    def handles_vertical(self) -> bool:
        return (
            self.in_handle is not None
            and self.out_handle is not None
            and abs(self.in_handle[0] - self.coordinates[0]) <= self.ALIGNMENT_EPSILON
            and abs(self.coordinates[0] - self.out_handle[0]) <= self.ALIGNMENT_EPSILON
        )

    # Convert to optimal command representation
    def command(self, previous_node: Optional["Node"] = None) -> NodeCommand:
        coords = self.coordinates
        if previous_node:
            coords = self.coordinates - previous_node.coordinates

        if self.in_handle is None and self.out_handle is None:
            # It's a line.
            if previous_node:  # Only check for LH/LV if coords are relative
                if abs(coords[1]) < self.ALIGNMENT_EPSILON:  # Horizontal line
                    return NodeCommand("LH", [coords[0].item(), 0])
                if abs(coords[0]) < self.ALIGNMENT_EPSILON:  # Vertical line
                    return NodeCommand("LV", [coords[1].item(), 0])
            return NodeCommand("L", coords.tolist())

        # Deal with line-to-curve and curve-to-line
        if self.in_handle is None and self.out_handle is not None:
            delta_hout = self.out_handle - self.coordinates
            return NodeCommand("NCO", np.concatenate((coords, delta_hout)).tolist())
        elif self.out_handle is None and self.in_handle is not None:
            delta_hin = self.in_handle - self.coordinates
            return NodeCommand("NCI", np.concatenate((coords, delta_hin)).tolist())

        # It's a curve. Can we do better?
        assert self.in_handle is not None and self.out_handle is not None
        delta_hin = self.in_handle - self.coordinates
        delta_hout = self.out_handle - self.coordinates

        if self.handles_horizontal:
            return NodeCommand(
                "NH",
                [coords[0], coords[1], delta_hin[0], delta_hout[0]],
            )
        elif self.handles_vertical:
            return NodeCommand(
                "NV",
                [coords[0], coords[1], delta_hin[1], delta_hout[1]],
            )
        else:
            return NodeCommand(
                "N", np.concatenate((coords, delta_hin, delta_hout)).tolist()
            )


class NodeGlyph:
    contours: List[NodeContour]

    def __init__(self, contours: List[NodeContour]):
        self.contours = contours

    @property
    def commands(self) -> List[NodeCommand]:
        cmds = []
        for contour in self.contours:
            cmds.append(NodeCommand("SOC", []))
            cmds.extend(contour.commands)
        return cmds

    def encode(self) -> Optional[npt.NDArray[np.float32]]:
        # We one-hot encode the commands
        command_width = len(NodeCommand.grammar.keys())
        # And we pad the coordinates to a fixed length
        max_coordinates = max(NodeCommand.grammar.values())
        output: List[np.ndarray] = []

        # Add SOS token
        sos_command_vector = np.zeros(command_width, dtype=np.float32)
        sos_command_vector[NodeCommand.encode_command("SOS")] = 1.0
        sos_coordinates = np.zeros(max_coordinates, dtype=np.float32)
        output.append(np.concatenate((sos_command_vector, sos_coordinates)))


        for command in self.commands:
            # One-hot encode the command
            command_vector = np.zeros(command_width, dtype=np.float32)
            command_vector[NodeCommand.encode_command(command.command)] = 1.0
            # Normalize and pad the coordinates
            norm_coords = (
                np.array(command.coordinates, dtype=np.float32) / MAX_COORDINATE
            )
            padded_coords = np.pad(
                norm_coords, (0, max_coordinates - len(command.coordinates))
            )
            output.append(np.concatenate((command_vector, padded_coords)))

        # Add EOS token
        eos_command_vector = np.zeros(command_width, dtype=np.float32)
        eos_command_vector[NodeCommand.encode_command("EOS")] = 1.0
        eos_coordinates = np.zeros(max_coordinates, dtype=np.float32)
        output.append(np.concatenate((eos_command_vector, eos_coordinates)))
        encoded_glyph = np.array(output, dtype=np.float32)

        # Pad vector representation or skip
        if encoded_glyph.shape[0] > MAX_SEQUENCE_LENGTH:
            return
        elif encoded_glyph.shape[0] < MAX_SEQUENCE_LENGTH:
            padding = np.zeros(
                (MAX_SEQUENCE_LENGTH - encoded_glyph.shape[0], encoded_glyph.shape[1]),
                dtype=np.float32,
            )
            encoded_glyph = np.vstack((encoded_glyph, padding))
        return encoded_glyph

    @classmethod
    def from_numpy(cls, command_tensor, coord_tensor, relative: bool = True):
        contours: List[NodeContour] = []
        command_keys = list(NodeCommand.grammar.keys())
        cur_contour = NodeContour([])
        current_pos = np.array([0, 0])
        is_first_node_in_contour = True

        for i in range(command_tensor.shape[0]):
            command_index = np.argmax(command_tensor[i])
            command_str = command_keys[command_index]

            if command_str == "EOS":
                break

            if command_str == "SOS":
                continue

            if command_str == "SOC":
                if cur_contour.nodes:
                    contours.append(cur_contour)
                cur_contour = NodeContour([])
                is_first_node_in_contour = True
                continue

            num_coords = NodeCommand.grammar[command_str]
            # Denormalize and convert to int
            coords_slice = coord_tensor[i, :num_coords]
            if torch.is_tensor(coords_slice):
                coords_slice = coords_slice.cpu().numpy()

            if relative:
                denorm_coords = (coords_slice * MAX_COORDINATE).round()

                if command_str == "LH":
                    coords = [int(denorm_coords[0]), 0]  # Only x-delta
                elif command_str == "LV":
                    coords = [
                        int(denorm_coords[0]),
                        0,
                    ]  # Only y-delta (coords_slice[0] is the y-delta)
                else:
                    # Handle relative coordinates. The coordinates for push_command must be absolute.
                    absolute_coords = np.copy(denorm_coords)
                    if is_first_node_in_contour:
                        current_pos = denorm_coords[0:2]
                        is_first_node_in_contour = False
                    else:
                        current_pos = current_pos + denorm_coords[0:2]
                    absolute_coords[0:2] = current_pos
                    coords = absolute_coords.astype(int).tolist()
            else:
                coords = coords_slice.round().astype(int).tolist()

            cur_contour.push_command(NodeCommand(command_str, coords))
        if cur_contour.nodes:
            contours.append(cur_contour)
        return cls(contours)

    def to_svg_glyph(self) -> "SVGGlyph":
        svg_commands: List[SVGCommand] = []
        for contour in self.contours:
            svg_commands.extend(contour.svg_commands)
        return SVGGlyph(svg_commands)

    def to_debug_string(self):
        path_data: List[str] = []
        for cmd in self.commands:
            path_data.append(cmd.command)
            path_data.extend(map(lambda x: str(int(x)), cmd.coordinates))
        return " ".join(path_data)

    def normalize(self):
        for contour in self.contours:
            contour.normalize()
        return self


class SVGGlyph:
    commands: List[SVGCommand]

    def __init__(self, commands):
        if commands and commands[0].command != "M":
            raise ValueError("SVG path must start with a 'M' command")
        self.commands = commands

    def to_node_glyph(self) -> "NodeGlyph":
        if not self.commands:
            return NodeGlyph([])

        svg_contours: List[List[SVGCommand]] = []
        current_contour = []
        for command in self.commands:
            current_contour.append(command)
            if command.command == "Z":
                svg_contours.append(current_contour)
                current_contour = []
        if current_contour:
            svg_contours.append(current_contour)

        node_glyph = NodeGlyph(
            [NodeContour.from_svg_contour(contour) for contour in svg_contours]
        )
        node_glyph.normalize()
        return node_glyph

    def to_svg_string(self):
        path_data: List[str] = []
        for cmd in self.commands:
            path_data.append(cmd.command)
            path_data.extend(map(lambda x: str(int(x)), cmd.coordinates))
        return " ".join(path_data)


cache_dir = Path("imgcache")


class Glyph:
    font_file: Path
    unicode_id: int
    location: Dict[str, float]  # Coordinates in a variable font's designspace

    def __init__(self, font_file: Path, unicode_id: int, location: Dict[str, float]):
        self.font_file = font_file
        self.unicode_id = unicode_id
        self.location = location

    def rasterize(self, size: int = RASTER_IMG_SIZE) -> npt.NDArray[np.float64]:
        """
        Rasterizes the glyph at a given size using the same pipeline as the model.
        """
        font_base = str(self.font_file).replace(BASE_DIR + "/", "").replace("/", "-")
        key = "-".join(
            [
                str(self.unicode_id),
                ",".join(
                    {f"{k}:{self.location[k]}" for k in sorted(self.location.keys())}
                ),
                str(size),
            ]
        )
        if (cache_dir / font_base / (key + ".png")).exists():
            # print("Loading", cache_dir / font_base / (key + ".png"))
            img = Image.open(cache_dir / font_base / (key + ".png")).convert("L")
            img = np.asarray(img, dtype=np.float32) / 255.0
            img = np.expand_dims(img, axis=-1)
        else:
            img = self._rasterize(size)
            pil_img = Image.fromarray(
                (img.squeeze(-1) * 255).astype(np.uint8), mode="L"
            )
            (cache_dir / font_base).mkdir(exist_ok=True)
            print("Saving", font_base, key)
            pil_img.save(cache_dir / font_base / (key + ".png"))
        return img

    def _rasterize(self, size: int) -> npt.NDArray[np.float64]:
        node_glyph = self.vectorize().to_node_glyph()
        encoded_glyph = node_glyph.encode()

        if encoded_glyph is None:
            # Handle glyphs that are too complex
            return np.zeros((size, size, 1), dtype=np.float64)

        encoded_tensor = torch.from_numpy(encoded_glyph).float()

        command_width = len(NodeCommand.grammar.keys())

        cmds_tensor = encoded_tensor[:, :command_width]
        coords_tensor = encoded_tensor[:, command_width:]

        # Add batch dimension
        cmds_tensor = cmds_tensor.unsqueeze(0)
        coords_tensor = coords_tensor.unsqueeze(0)

        coord_tensor_absolute = unroll_relative_coords(cmds_tensor, coords_tensor)

        # Rasterize
        image_tensor = rasterize_batch(
            cmds_tensor,
            coord_tensor_absolute,
            img_size=size,
            requires_grad=False,
            device=torch.device("cpu"),
        )

        numpy_image = image_tensor.squeeze(0).squeeze(0).cpu().numpy()
        return np.expand_dims(numpy_image, axis=-1).astype(np.float64)

    def vectorize(self, remove_overlaps: bool = True) -> SVGGlyph:
        """
        Converts the glyph into a vector representation.
        This method should be implemented to return the glyph's path as an SVGGlyph object.
        """
        scale = 1000 / TTFont(self.font_file)["head"].unitsPerEm
        blob = hb.Blob.from_file_path(self.font_file)
        face = hb.Face(blob)
        font = hb.Font(face)
        svgpen = AbsoluteSVGPathPen({}, ntos=lambda f: str(int(f * scale)))
        pen = Qu2CuPen(svgpen, max_err=5, all_cubic=True)
        if self.location:
            font.set_variations(self.location)
        glyph = font.get_nominal_glyph(self.unicode_id)
        path = []
        if glyph is None:
            return SVGGlyph([])

        if remove_overlaps:
            # Simplify and remove overlaps
            skpath = pathops.Path()
            pathPen = skpath.getPen()
            font.draw_glyph_with_pen(glyph, pathPen)
            skpath = _simplify(skpath, chr(self.unicode_id))
            skpath.draw(pen)
        else:
            font.draw_glyph_with_pen(glyph, pen)

        for command in svgpen._commands:
            cmd = command[0] if command[0] != " " else "L"
            coords = [int(p) for p in command[1:].split()]
            if "XAUG" in self.location:
                # Add XAUG to the x coordinates
                for i in range(0, len(coords), 2):
                    coords[i] += int(self.location["XAUG"])
            if "YAUG" in self.location:
                # Add YAUG to the y coordinates
                for i in range(1, len(coords), 2):
                    coords[i] += int(self.location["YAUG"])
            path.append(SVGCommand(cmd, coords))
        return SVGGlyph(path)


def simplify(glyph):
    glyph.draw(pathPen)
    path = _simplify(path)
