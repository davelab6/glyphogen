import torch

from glyphogen_torch.command_defs import COORDINATE_WIDTH, NODE_COMMAND_WIDTH
from glyphogen_torch.hyperparameters import (
    BATCH_SIZE,
    D_MODEL,
    GEN_IMAGE_SIZE,
    LATENT_DIM,
    MAX_COMMANDS,
    NUM_GLYPHS,
    RATE,
)
from glyphogen_torch.model import GlyphGenerator


def test_output_shapes():
    model = GlyphGenerator(
        num_glyphs=NUM_GLYPHS,
        d_model=D_MODEL,
        latent_dim=LATENT_DIM,
        rate=RATE,
    )

    # Dummy input data
    style_image_input = torch.rand(BATCH_SIZE, 1, 40, 168)
    glyph_id_input = torch.nn.functional.one_hot(
        torch.randint(0, NUM_GLYPHS, (BATCH_SIZE,)),
        NUM_GLYPHS,
    ).float()
    target_sequence_input = torch.rand(
        BATCH_SIZE, MAX_COMMANDS, NODE_COMMAND_WIDTH + COORDINATE_WIDTH
    )

    # Get model output
    outputs = model((style_image_input, glyph_id_input, target_sequence_input))
    generated_glyph_raster = outputs["raster"]
    command_output = outputs["command"]
    coord_output = outputs["coord"]

    # Check output shapes
    assert generated_glyph_raster.shape == (BATCH_SIZE, 1, *GEN_IMAGE_SIZE)
    assert command_output.shape == (BATCH_SIZE, MAX_COMMANDS, NODE_COMMAND_WIDTH)
    assert coord_output.shape == (BATCH_SIZE, MAX_COMMANDS, COORDINATE_WIDTH)
