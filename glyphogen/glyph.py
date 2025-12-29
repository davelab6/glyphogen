from pathlib import Path
from time import sleep
from typing import Dict, List, Optional

import diskcache
from glyphogen.coordinate import get_bounds, to_image_space
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
from kurbopy import BezPath, Point


# No point cacheing as we are storing the PNGs in our dataset
CACHING = False

from PIL import Image, ImageDraw

from .hyperparameters import (
    BASE_DIR,
    GEN_IMAGE_SIZE,
    MAX_SEQUENCE_LENGTH,
    RASTER_IMG_SIZE,
)
from .rasterizer import rasterize_batch
from .command_defs import SVGCommand, NodeCommand


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

        # First command is an absolute Move to the first node's position
        node_commands = [NodeCommand("M", self.nodes[0].coordinates.tolist())]

        # Generate a command for each node
        for i in range(len(self.nodes)):
            p_curr = self.nodes[i]
            
            if i == 0:
                # First node's command has a zero delta from the 'M' position
                delta_pos = np.array([0, 0])
            else:
                p_prev = self.nodes[i-1]
                delta_pos = p_curr.coordinates - p_prev.coordinates

            # The command for node `i` describes the handles of node `i`
            # and the movement from node `i-1` to `i`.
            h_in_curr = p_curr.in_handle - p_curr.coordinates if p_curr.in_handle is not None else np.array([0,0])
            h_out_curr = p_curr.out_handle - p_curr.coordinates if p_curr.out_handle is not None else np.array([0,0])

            # For a pure line segment, the node has no handles.
            # We check the handles of the current node to decide the command type.
            is_line_node = np.array_equal(h_in_curr, [0,0]) and np.array_equal(h_out_curr, [0,0])

            if is_line_node:
                # For the first node (i=0), even if it's a line-node, we can't use LH/LV
                # as the delta is zero. We use a generic L.
                if abs(delta_pos[1]) < p_curr.ALIGNMENT_EPSILON and i > 0:
                    node_commands.append(NodeCommand("LH", [delta_pos[0]]))
                elif abs(delta_pos[0]) < p_curr.ALIGNMENT_EPSILON and i > 0:
                    node_commands.append(NodeCommand("LV", [delta_pos[1]]))
                else:
                    node_commands.append(NodeCommand("L", delta_pos.tolist()))
            else:  # It's a curve node, so encode the handles
                if p_curr.handles_horizontal:
                    coords = np.concatenate((delta_pos, [h_in_curr[0]], [h_out_curr[0]]))
                    node_commands.append(NodeCommand("NH", coords.tolist()))
                elif p_curr.handles_vertical:
                    coords = np.concatenate((delta_pos, [h_in_curr[1]], [h_out_curr[1]]))
                    node_commands.append(NodeCommand("NV", coords.tolist()))
                else:
                    coords = np.concatenate((delta_pos, h_in_curr, h_out_curr))
                    node_commands.append(NodeCommand("N", coords.tolist()))
        
        return node_commands

    def to_svg_commands(self) -> List["SVGCommand"]:
        svg_commands: List["SVGCommand"] = []
        # First command is always an absolute Move to the first node's position
        if not self.nodes:
            return []
        svg_commands.append(SVGCommand("M", self.nodes[0].coordinates.tolist()))

        # Add commands for each segment of the contour
        for i in range(len(self.nodes)):
            p_prev = self.nodes[i]
            p_curr = self.nodes[(i + 1) % len(self.nodes)]

            # Determine if the segment is a straight line
            is_line = p_prev.out_handle is None and p_curr.in_handle is None

            if is_line:
                svg_commands.append(SVGCommand("L", p_curr.coordinates.tolist()))
            else:  # It's a curve arriving at p_curr
                # For a curve, we need the control points.
                # p_prev.out_handle is c1, p_curr.in_handle is c2
                c1 = p_prev.out_handle.tolist() if p_prev.out_handle is not None else p_prev.coordinates.tolist()
                c2 = p_curr.in_handle.tolist() if p_curr.in_handle is not None else p_curr.coordinates.tolist()
                svg_commands.append(SVGCommand("C", c1 + c2 + p_curr.coordinates.tolist()))
        
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

        closing_segment = {"start": current_node_idx, "end": start_node_idx, "type": "line"}
        if len(segments) == 0 or segments[-1] != closing_segment:
             segments.append(closing_segment)

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
        in_handle: Optional[np.ndarray] = None

    def roll(self, shift: int):
        if not self.nodes:
            return
        shift = shift % len(self.nodes)
        self.nodes = self.nodes[shift:] + self.nodes[:shift]

    def normalize(self) -> None:
        if not self.nodes:
            return
        index_of_bottom_left = min(
            range(len(self.nodes)),
            key=lambda i: (self.nodes[i].coordinates[1], self.nodes[i].coordinates[0]),
        )
        self.nodes = (
            self.nodes[index_of_bottom_left:] + self.nodes[:index_of_bottom_left]
        )

    def __eq__(self, other):
        if not isinstance(other, NodeContour):
            return NotImplemented
        return len(self.nodes) == len(other.nodes) and all(
            n1 == n2 for n1, n2 in zip(self.nodes, other.nodes)
        )


class Node:
    coordinates: npt.NDArray[np.int_]
    in_handle: Optional[npt.NDArray[np.int_]]
    out_handle: Optional[npt.NDArray[np.int_]]
    _contour: "NodeContour"
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

    def __eq__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        
        coords_equal = np.allclose(self.coordinates, other.coordinates)
        
        in_handles_equal = (self.in_handle is None and other.in_handle is None) or \
                           (self.in_handle is not None and other.in_handle is not None and np.allclose(self.in_handle, other.in_handle))
                           
        out_handles_equal = (self.out_handle is None and other.out_handle is None) or \
                            (self.out_handle is not None and other.out_handle is not None and np.allclose(self.out_handle, other.out_handle))

        return coords_equal and in_handles_equal and out_handles_equal


class NodeGlyph:
    contours: List[NodeContour]
    origin: str

    def __init__(self, contours: List[NodeContour], origin="unknown"):
        self.contours = contours
        self.origin = origin

    def __eq__(self, other):
        if not isinstance(other, NodeGlyph):
            return NotImplemented
        return len(self.contours) == len(other.contours) and all(
            c1 == c2 for c1, c2 in zip(self.contours, other.contours)
        )

    @property
    def commands(self) -> List[NodeCommand]:
        cmds = []
        for contour in self.contours:
            cmds.extend(contour.commands)
        return cmds

    def encode(self) -> Optional[List[npt.NDArray[np.float32]]]:
        from .command_defs import NODE_COMMAND_WIDTH, COORDINATE_WIDTH
        contour_sequences = []

        for contour in self.contours:
            output: List[np.ndarray] = []
            sos_command_vector = np.zeros(NODE_COMMAND_WIDTH, dtype=np.float32)
            sos_command_vector[NodeCommand.encode_command("SOS")] = 1.0
            sos_coordinates = np.zeros(COORDINATE_WIDTH, dtype=np.float32)
            output.append(np.concatenate((sos_command_vector, sos_coordinates)))

            for command in contour.commands:
                command_vector = np.zeros(NODE_COMMAND_WIDTH, dtype=np.float32)
                command_vector[NodeCommand.encode_command(command.command)] = 1.0
                coords = np.array(command.coordinates, dtype=np.float32)
                padded_coords = np.pad(
                    coords, (0, COORDINATE_WIDTH - len(command.coordinates)) # type: ignore
                )
                output.append(np.concatenate((command_vector, padded_coords)))

            eos_command_vector = np.zeros(NODE_COMMAND_WIDTH, dtype=np.float32)
            eos_command_vector[NodeCommand.encode_command("EOS")] = 1.0
            eos_coordinates = np.zeros(COORDINATE_WIDTH, dtype=np.float32)
            output.append(np.concatenate((eos_command_vector, eos_coordinates)))

            encoded_contour = np.array(output, dtype=np.float32)

            if encoded_contour.shape[0] > MAX_SEQUENCE_LENGTH:
                return None

            contour_sequences.append(encoded_contour)

        return contour_sequences if contour_sequences else None

    @classmethod
    def from_numpy(cls, contour_sequences: List):
        from .command_defs import NODE_GLYPH_COMMANDS, NODE_COMMAND_WIDTH
        
        command_keys = list(NODE_GLYPH_COMMANDS.keys())
        glyph_commands = []

        for command_tensor, coord_tensor in contour_sequences:
            contour_commands = []
            for i in range(command_tensor.shape[0]):
                command_index = torch.argmax(command_tensor[i]).item()
                command_str = command_keys[command_index]
                
                if command_str == "SOS":
                    continue
                if command_str == "EOS":
                    break
                
                num_coords = NODE_GLYPH_COMMANDS[command_str]
                coords_slice = coord_tensor[i, :num_coords].cpu().numpy()
                contour_commands.append(NodeCommand(command_str, coords_slice.tolist()))
            glyph_commands.append(contour_commands)
        
        return cls.from_commands(glyph_commands)

    @classmethod
    def from_commands(cls, glyph_commands: List[List[NodeCommand]]):
        contours: List[NodeContour] = []
        for contour_cmds in glyph_commands:
            nodes: List[Node] = []
            contour_obj = NodeContour(nodes)
            
            if not contour_cmds or contour_cmds[0].command != "M":
                continue

            current_pos = np.array(contour_cmds[0].coordinates)

            for command in contour_cmds[1:]:
                in_handle: Optional[np.ndarray] = None
                out_handle: Optional[np.ndarray] = None
                
                delta_pos = np.array([0.0, 0.0])
                if command.command in ["L", "N", "NH", "NV"]:
                    delta_pos = command.coordinates[0:2]
                elif command.command == "LH":
                    delta_pos[0] = command.coordinates[0]
                elif command.command == "LV":
                    delta_pos[1] = command.coordinates[0]
                
                current_pos += delta_pos

                if command.command in ["N", "NH", "NV"]:
                    if command.command == "N":
                        in_handle = current_pos + command.coordinates[2:4]
                        out_handle = current_pos + command.coordinates[4:6]
                    elif command.command == "NH":
                        in_handle = current_pos + np.array([command.coordinates[2], 0.0])
                        out_handle = current_pos + np.array([command.coordinates[3], 0.0])
                    elif command.command == "NV":
                        in_handle = current_pos + np.array([0.0, command.coordinates[2]])
                        out_handle = current_pos + np.array([0.0, command.coordinates[3]])
                
                nodes.append(Node(current_pos.copy(), contour_obj, in_handle, out_handle))

            if nodes:
                contours.append(contour_obj)

        return cls(contours)

    def to_svg_glyph(self) -> "SVGGlyph":
        svg_commands: List[SVGCommand] = []
        for contour in self.contours:
            svg_commands.extend(contour.to_svg_commands())
        return SVGGlyph(svg_commands, self.origin)

    def to_debug_string(self):
        path_data: List[str] = []
        for cmd in self.commands:
            path_data.append(cmd.command)
            path_data.extend(map(lambda x: str(int(x)), cmd.coordinates))
        return " ".join(path_data)

    def normalize(self):
        for contour in self.contours:
            contour.normalize()
        self.contours.sort(
            key=lambda c: (
                (c.nodes[0].coordinates[1], c.nodes[0].coordinates[0])
                if c.nodes
                else (float("inf"), float("inf"))
            )
        )
        return self


class SVGGlyph:
    commands: List[SVGCommand]
    origin: str

    def __init__(self, commands, origin="unknown"):
        if commands and commands[0].command != "M":
            raise ValueError("SVG path must start with a 'M' command")
        self.commands = commands
        self.origin = origin

    def to_node_glyph(self) -> "NodeGlyph":
        if not self.commands:
            return NodeGlyph([], self.origin)

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
            [NodeContour.from_svg_contour(contour) for contour in svg_contours],
            self.origin,
        )
        node_glyph.normalize()
        return node_glyph

    def to_svg_string(self):
        path_data: List[str] = []
        for cmd in self.commands:
            path_data.append(cmd.command)
            path_data.extend(map(lambda x: str(int(x)), cmd.coordinates))
        return " ".join(path_data)

    def to_bezpaths(self) -> List[BezPath]:
        if not self.commands:
            return []

        svg_contours: List[List[SVGCommand]] = []
        current_contour: List[SVGCommand] = []
        for command in self.commands:
            current_contour.append(command)
            if command.command == "Z":
                svg_contours.append(current_contour)
                current_contour = []
        if current_contour:
            svg_contours.append(current_contour)

        kurbopy_contours = []

        for contour_cmds in svg_contours:
            path = BezPath()
            for cmd in contour_cmds:
                if cmd.command == "M":
                    path.move_to(Point(*cmd.coordinates))
                elif cmd.command == "L":
                    path.line_to(Point(*cmd.coordinates))
                elif cmd.command == "C":
                    path.curve_to(
                        Point(cmd.coordinates[0], cmd.coordinates[1]),
                        Point(cmd.coordinates[2], cmd.coordinates[3]),
                        Point(cmd.coordinates[4], cmd.coordinates[5]),
                    )
                elif cmd.command == "Z":
                    path.close_path()
            kurbopy_contours.append(path)
        return kurbopy_contours

    def get_segmentation_data(self):
        segmentation_data = []
        kurbopy_contours = self.to_bezpaths()
        for i, path_i in enumerate(kurbopy_contours):
            segs = path_i.segments()
            if not segs or not segs[0]:
                continue

            points = [(pt.x, pt.y) for pt in path_i.flatten(1.0)]
            if not points:
                continue

            bbox = get_bounds(points)

            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if width <= 0 or height <= 0:
                print(
                    f"Warning: Skipping contour with zero width/height in {self.origin}. Bbox: {bbox}"
                )
                continue

            containment_count = 0
            test_point = path_i.segments()[0].start()
            for j, path_j in enumerate(kurbopy_contours):
                if i == j:
                    continue
                if path_j.contains(test_point):
                    containment_count += 1
            is_hole = containment_count % 2 == 1

            img = Image.new("L", (GEN_IMAGE_SIZE[0], GEN_IMAGE_SIZE[1]), 0)
            draw = ImageDraw.Draw(img)
            draw.polygon(points, fill=1)
            mask = np.array(img, dtype=np.uint8)

            segmentation_data.append(
                {
                    "bbox": bbox,
                    "label": 1 if is_hole else 0,
                    "mask": mask,
                }
            )

        return segmentation_data


cache_dir = Path("imgcache")


class Glyph:
    font_file: Path
    unicode_id: int
    location: Dict[str, float]

    def __init__(self, font_file: Path, unicode_id: int, location: Dict[str, float]):
        self.font_file = font_file
        self.unicode_id = unicode_id
        self.location = location

    def rasterize(self, size: int = RASTER_IMG_SIZE) -> npt.NDArray[np.float64]:
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
        if CACHING and (cache_dir / font_base / (key + ".png")).exists():
            img = Image.open(cache_dir / font_base / (key + ".png")).convert("L")
            img = np.asarray(img, dtype=np.float32) / 255.0
            img = np.expand_dims(img, axis=-1)
        else:
            img = self._rasterize(size)
            if CACHING:
                pil_img = Image.fromarray(
                    (img.squeeze(-1) * 255).astype(np.uint8), mode="L"
                )
                (cache_dir / font_base).mkdir(exist_ok=True)
                print("Saving", font_base, key)
                pil_img.save(cache_dir / font_base / (key + ".png"))
        return img

    def _rasterize(self, size: int) -> npt.NDArray[np.float64]:
        node_glyph = self.vectorize().to_node_glyph()
        contour_sequences = node_glyph.encode()

        if contour_sequences is None:
            return np.zeros((size, size, 1), dtype=np.float64)

        from .command_defs import NODE_COMMAND_WIDTH
        contour_tensors = []
        for encoded_contour in contour_sequences:
            encoded_tensor = torch.from_numpy(encoded_contour).float()
            cmds_tensor = encoded_tensor[:, :NODE_COMMAND_WIDTH]
            coords_tensor = encoded_tensor[:, NODE_COMMAND_WIDTH:]

            contour_tensors.append((cmds_tensor, coords_tensor))

        image_tensor = rasterize_batch(
            [contour_tensors],
            img_size=size,
            requires_grad=False,
            device=torch.device("cpu"),
        )

        numpy_image = image_tensor.squeeze(0).squeeze(0).cpu().numpy()
        return np.expand_dims(numpy_image, axis=-1).astype(np.float64)

    def vectorize(self, remove_overlaps: bool = True) -> SVGGlyph:
        scale = 1000 / TTFont(self.font_file)['head'].unitsPerEm
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
                for i in range(0, len(coords), 2):
                    coords[i] += int(self.location["XAUG"])
            if "YAUG" in self.location:
                for i in range(1, len(coords), 2):
                    coords[i] += int(self.location["YAUG"])
            
            image_space_coords = []
            for x, y in zip(coords[0::2], coords[1::2]):
                ix, iy = to_image_space((x, y))
                image_space_coords.extend([ix, iy])
            path.append(SVGCommand(cmd, image_space_coords))
        return SVGGlyph(path, "%s, %s" % (self.font_file, chr(self.unicode_id)))