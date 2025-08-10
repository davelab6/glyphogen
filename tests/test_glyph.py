from pathlib import Path
from deepvecfont3.glyph import (
    Glyph,
    UnrelaxedSVG,
    RelaxedSVG,
    ExtendedCommand,
    EXTENDED_COMMAND_WIDTH,
)
from deepvecfont3.hyperparameters import GEN_IMAGE_SIZE
import numpy as np


def test_glyph_extraction():
    font_path = Path("tests/data/NotoSans[wdth,wght].ttf")
    location = {"wght": 400.0}
    codepoint = ord("l")

    glyph = Glyph(font_path, codepoint, location)

    # Rasterize the glyph
    rasterized_glyph = glyph.rasterize(GEN_IMAGE_SIZE[0])

    # Check the size of the rasterized numpy array
    assert rasterized_glyph.shape == (*GEN_IMAGE_SIZE, 1)

    # import matplotlib.pyplot as plt

    # plt.imshow(rasterized_glyph.squeeze(), cmap="gray")
    # plt.show()

    # Vectorize the glyph
    vectorized_glyph = glyph.vectorize()

    assert isinstance(vectorized_glyph, UnrelaxedSVG)
    assert len(vectorized_glyph.commands) > 0
    assert vectorized_glyph.commands[0].command == "M"
    relaxed = vectorized_glyph.relax()
    assert relaxed.to_svg_string() == "M 170 0 H 85 V 760 H 170 V 0 Z"


def test_relaxed_svg_conversion():
    commands = [
        ExtendedCommand("M", [10, 20]),
        ExtendedCommand("L", [30, 40]),
        ExtendedCommand("C", [50, 60, 70, 80, 90, 100]),
        ExtendedCommand("Z", []),
    ]
    relaxed_svg = RelaxedSVG(commands)

    encoded_svg = relaxed_svg.encode()

    # Add a batch dimension and convert to tensor
    encoded_svg_batch = np.expand_dims(encoded_svg, axis=0)

    command_tensor = encoded_svg_batch[:, :, :EXTENDED_COMMAND_WIDTH]
    coord_tensor = encoded_svg_batch[:, :, EXTENDED_COMMAND_WIDTH:]

    # Remove the batch dimension for from_numpy
    decoded_svg = RelaxedSVG.from_numpy(command_tensor[0], coord_tensor[0])

    assert len(decoded_svg.commands) == len(commands)
    for i, command in enumerate(decoded_svg.commands):
        assert command.command == commands[i].command
        assert command.coordinates == commands[i].coordinates

    svg_string = decoded_svg.to_svg_string()
    assert svg_string == "M 10 20 L 30 40 C 50 60 70 80 90 100 Z"
