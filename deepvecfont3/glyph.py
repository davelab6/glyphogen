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
    # 'Q': 4,  # Quadratic Bezier curve # Fonts don't use quads!
    "Z": 0,  # Close path (no coordinates)
}

EXTENDED_SVG_COMMANDS = {
    **BASIC_SVG_COMMANDS,
    "HS": 4,  # Horizontal smooth node (2 coordinates plus handle lengths)
    "VS": 4,  # Vertical smooth node (2 coordinates plus handle lengths)
    "G2": 2,  # G2 continuous node (2 coordinates)
}

EXTENDED_COMMAND_WIDTH = len(EXTENDED_SVG_COMMANDS.keys())  # 7
COORDINATE_WIDTH = max(EXTENDED_SVG_COMMANDS.values())  # 6


class SVGCommand:
    grammar = BASIC_SVG_COMMANDS
    command: str
    coordinates: List[int]

    def __init__(self, command: str, coordinates: List[int]):
        if command not in self.grammar:
            raise ValueError(f"Invalid SVG command: {command}")
        if len(coordinates) != self.grammar[command]:
            raise ValueError(
                f"Invalid number of coordinates for command {command}: expected {SVGCommand.grammar[command]}, got {len(coordinates)}"
            )
        self.command = command
        self.coordinates = coordinates


class ExtendedCommand(SVGCommand):
    grammar = EXTENDED_SVG_COMMANDS


class RelaxedSVG:
    commands: List[ExtendedCommand]

    def __init__(self, commands: List[ExtendedCommand]):
        self.commands = commands

    def encode(self) -> npt.NDArray[np.int_]:
        # We one-hot encode the commands
        command_width = len(ExtendedCommand.grammar.keys())
        # And we pad the coordinates to a fixed length
        max_coordinates = max(ExtendedCommand.grammar.values())
        output: List[List[int]] = []
        for command in self.commands:
            # One-hot encode the command
            command_vector = [0] * command_width
            command_vector[
                list(ExtendedCommand.grammar.keys()).index(command.command)
            ] = 1
            # Pad the coordinates
            coordinates = command.coordinates + [0] * (
                max_coordinates - len(command.coordinates)
            )
            output.append(command_vector + coordinates)
        return np.array(output, dtype=np.int_)


class UnrelaxedSVG:
    commands: List[SVGCommand]

    def __init__(self, commands):
        self.commands = commands

    def relax(self) -> RelaxedSVG:
        """
        Converts the UnrelaxedSVG to a RelaxedSVG.
        This method should be implemented to convert the SVG commands into a relaxed format.
        """
        # XXX Later we'll implement this to convert the SVG commands into a relaxed format.
        # For now, just use plain SVG commands.
        return RelaxedSVG(
            [ExtendedCommand(cmd.command, cmd.coordinates) for cmd in self.commands]
        )


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

    def vectorize(self) -> UnrelaxedSVG:
        """
        Converts the glyph into a vector representation.
        This method should be implemented to return the glyph's path as an UnrelaxedSVG object.
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
            return UnrelaxedSVG([])
        font.draw_glyph_with_pen(glyph, pen)
        for command in svgpen._commands:
            cmd = command[0] if command[0] != " " else "L"
            coords = [int(p) for p in command[1:].split()]
            path.append(SVGCommand(cmd, coords))
        return UnrelaxedSVG(path)
