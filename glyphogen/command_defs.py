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
    "SOC": 0,  # Start of Contour
    "N": 6,  # Node with two handles (x, y, delta_hin_x, delta_hin_y, delta_hout_x, delta_hout_y)
    "NH": 4,  # Node with horizontal handles (x, y, delta_hin_x, delta_hout_x)
    "NV": 4,  # Node with vertical handles (x, y, delta_hin_y, delta_hout_y)
    "NCI": 4,  # Node with curve in, line out (x, y, delta_hin_x, delta_hin_y)
    "NCO": 4,  # Node with line in, curve out (x, y, delta_hout_x, delta_hout_y)
    "L": 2,  # Line node (x, y)
    "LH": 2,  # Horizontal line (1 relative arg, but 2 absolute coords checked in loss)
    "LV": 2,  # Vertical line (1 relative arg, but 2 absolute coords checked in loss)
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
