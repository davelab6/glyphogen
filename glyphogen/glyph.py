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

        node_commands = [self.nodes[0].command()]
        for i in range(1, len(self.nodes)):
            node_commands.append(self.nodes[i].command())
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
        in_handle: Optional[np.ndarray] = None
        out_handle: Optional[np.ndarray] = None

        if command.command == "L":
            position = np.array(command.coordinates)
        elif command.command == "N":
            position = np.array(command.coordinates[0:2])
            in_handle_delta = np.array(command.coordinates[2:4])
            # in_handle = position + in_handle_delta
            in_handle = in_handle_delta

            out_handle_delta = np.array(command.coordinates[4:6])
            # out_handle = position + out_handle_delta
            out_handle = out_handle_delta
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
        if not self.nodes:
            return
        index_of_bottom_left = min(
            range(len(self.nodes)),
            key=lambda i: (self.nodes[i].coordinates[1], self.nodes[i].coordinates[0]),
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
    def command(self) -> NodeCommand:
        coords = self.coordinates

        if self.in_handle is None and self.out_handle is None:
            # It's a line.
            return NodeCommand("L", coords.tolist())

        if self.in_handle is not None:
            hin = self.in_handle  # - self.coordinates
        else:
            hin = self.coordinates

        if self.out_handle is not None:
            hout = self.out_handle  # - self.coordinates
        else:
            hout = self.coordinates

        return NodeCommand("N", np.concatenate((coords, hin, hout)).tolist())


class NodeGlyph:
    contours: List[NodeContour]
    origin: str

    def __init__(self, contours: List[NodeContour], origin="unknown"):
        self.contours = contours
        self.origin = origin

    @property
    def commands(self) -> List[NodeCommand]:
        """
        Returns all commands for the entire glyph as a single stream.
        Deprecated: Use encode() which returns per-contour sequences.
        """
        cmds = []
        for contour in self.contours:
            cmds.extend(contour.commands)
        return cmds

    def encode(self) -> Optional[List[npt.NDArray[np.float32]]]:
        """
        Encodes the glyph as a list of sequences, one per contour.
        Each sequence contains: [SOS, N, N, ..., EOS]

        Returns:
            A list of numpy arrays, one per contour. Each array has shape (seq_len, command_width + coord_width).
            Returns None if any contour is too long.
        """
        command_width = len(NodeCommand.grammar.keys())
        max_coordinates = max(NodeCommand.grammar.values())

        contour_sequences = []

        for contour in self.contours:
            output: List[np.ndarray] = []

            # Add SOS token at the start of this contour
            sos_command_vector = np.zeros(command_width, dtype=np.float32)
            sos_command_vector[NodeCommand.encode_command("SOS")] = 1.0
            sos_coordinates = np.zeros(max_coordinates, dtype=np.float32)
            output.append(np.concatenate((sos_command_vector, sos_coordinates)))

            # Add all node commands for this contour
            for command in contour.commands:
                # One-hot encode the command
                command_vector = np.zeros(command_width, dtype=np.float32)
                command_vector[NodeCommand.encode_command(command.command)] = 1.0
                # Pad the coordinates
                norm_coords = np.array(command.coordinates, dtype=np.float32)
                padded_coords = np.pad(
                    norm_coords, (0, max_coordinates - len(command.coordinates))
                )
                output.append(np.concatenate((command_vector, padded_coords)))

            # Add EOS token at the end of this contour
            eos_command_vector = np.zeros(command_width, dtype=np.float32)
            eos_command_vector[NodeCommand.encode_command("EOS")] = 1.0
            eos_coordinates = np.zeros(max_coordinates, dtype=np.float32)
            output.append(np.concatenate((eos_command_vector, eos_coordinates)))

            encoded_contour = np.array(output, dtype=np.float32)

            # Check if contour is too long
            if encoded_contour.shape[0] > MAX_SEQUENCE_LENGTH:
                return None

            contour_sequences.append(encoded_contour)

        return contour_sequences if contour_sequences else None

    @classmethod
    def from_numpy(cls, contour_sequences: List):
        """
        Decode a list of command/coordinate sequences into a NodeGlyph.
        Coordinates are assumed to be absolute.

        Args:
            contour_sequences: List of (command_tensor, coord_tensor) tuples, one per contour

        Returns:
            NodeGlyph instance
        """
        contours: List[NodeContour] = []
        command_keys = list(NodeCommand.grammar.keys())

        for command_tensor, coord_tensor in contour_sequences:
            cur_contour = NodeContour([])

            # Pad coordinates to 6 dimensions if needed
            if coord_tensor.shape[1] == 2:
                if torch.is_tensor(coord_tensor):
                    padding = torch.zeros(
                        coord_tensor.shape[0],
                        4,
                        device=coord_tensor.device,
                        dtype=coord_tensor.dtype,
                    )
                    coord_tensor = torch.cat([coord_tensor, padding], dim=1)
                else:
                    padding = np.zeros(
                        (coord_tensor.shape[0], 4), dtype=coord_tensor.dtype
                    )
                    coord_tensor = np.concatenate([coord_tensor, padding], axis=1)

            for i in range(command_tensor.shape[0]):
                command_index = np.argmax(command_tensor[i].cpu().numpy())
                command_str = command_keys[command_index]

                if command_str in ["EOS", "SOS"]:
                    continue

                num_coords = NodeCommand.grammar[command_str]
                coords_slice = coord_tensor[i, :num_coords]
                if torch.is_tensor(coords_slice):
                    coords_slice = coords_slice.cpu().numpy()

                coords = coords_slice.round().astype(int).tolist()
                cur_contour.push_command(NodeCommand(command_str, coords))

            if cur_contour.nodes:
                contours.append(cur_contour)

        return cls(contours)

    def to_svg_glyph(self) -> "SVGGlyph":
        svg_commands: List[SVGCommand] = []
        for contour in self.contours:
            svg_commands.extend(contour.svg_commands)
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

        # 1. Split into contours
        svg_contours: List[List[SVGCommand]] = []
        current_contour: List[SVGCommand] = []
        for command in self.commands:
            current_contour.append(command)
            if command.command == "Z":
                svg_contours.append(current_contour)
                current_contour = []
        if current_contour:
            svg_contours.append(current_contour)

        # 2. Convert to kurbopy.BezPath objects
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

            # 3. Flatten path and transform to image space
            font_space_points = [(pt.x, pt.y) for pt in path_i.flatten(1.0)]
            if not font_space_points:
                continue

            points_tensor = torch.tensor(font_space_points)
            image_space_points_tensor = to_image_space(points_tensor)
            image_space_points = image_space_points_tensor.tolist()

            # 4. Get Bounding Box in image space
            bbox = get_bounds(image_space_points_tensor)

            # Check for zero-width or zero-height bounding boxes
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if width <= 0 or height <= 0:
                print(
                    f"Warning: Skipping contour with zero width/height in {self.origin}. Bbox: {bbox}"
                )
                continue

            # 5. Determine Contour Type (can be done in font space)
            containment_count = 0
            test_point = path_i.segments()[0].start()
            for j, path_j in enumerate(kurbopy_contours):
                if i == j:
                    continue
                if path_j.contains(test_point):
                    containment_count += 1
            is_hole = containment_count % 2 == 1

            # 6. Generate Mask in image space
            img = Image.new("L", (GEN_IMAGE_SIZE[0], GEN_IMAGE_SIZE[1]), 0)
            draw = ImageDraw.Draw(img)
            draw.polygon(image_space_points, fill=1)
            mask = np.array(img, dtype=np.uint8)

            segmentation_data.append(
                {
                    "bbox": bbox,
                    "label": 1 if is_hole else 0,  # 0 for outer, 1 for hole
                    "mask": mask,
                }
            )

        return segmentation_data


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
        contour_sequences = node_glyph.encode()

        if contour_sequences is None:
            # Handle glyphs that are too complex
            return np.zeros((size, size, 1), dtype=np.float64)

        command_width = len(NodeCommand.grammar.keys())

        # Convert each contour sequence to tensors and split into commands/coords
        contour_tensors = []
        for encoded_contour in contour_sequences:
            encoded_tensor = torch.from_numpy(encoded_contour).float()
            cmds_tensor = encoded_tensor[:, :command_width]
            coords_tensor = encoded_tensor[:, command_width:]

            # The coords_tensor now contains absolute coordinates.
            contour_tensors.append((cmds_tensor, coords_tensor))

        # Rasterize all contours as a single glyph
        image_tensor = rasterize_batch(
            [contour_tensors],
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
        return SVGGlyph(path, "%s, %s" % (self.font_file, chr(self.unicode_id)))
