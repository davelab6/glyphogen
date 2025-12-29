from typing import List

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
    "M": 2,  # Absolute move to (x, y)
    "L": 2,  # Relative line to (dx, dy)
    "LH": 1, # Relative horizontal line to (dx)
    "LV": 1, # Relative vertical line to (dy)
    "N": 6,  # Relative node with two handles (dx, dy, dhix, dhiy, dhox, dhoy)
    "NH": 4, # Relative horizontal-handle node (dx, dy, dh_in_x, dh_out_x)
    "NV": 4, # Relative vertical-handle node (dx, dy, dh_in_y, dh_out_y)
    "EOS": 0,
}

MAX_COORDINATE = 1000


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

    @classmethod
    def encode_command(cls, s: str) -> int:
        return list(cls.grammar.keys()).index(s)


NODE_COMMAND_WIDTH = len(NODE_GLYPH_COMMANDS.keys())
COORDINATE_WIDTH = max(NODE_GLYPH_COMMANDS.values())
PREDICTED_COORDINATE_WIDTH = COORDINATE_WIDTH
