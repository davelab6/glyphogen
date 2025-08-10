import tensorflow as tf

from deepvecfont3.glyph import COORDINATE_WIDTH, EXTENDED_COMMAND_WIDTH
from deepvecfont3.hyperparameters import (
    BATCH_SIZE,
    D_MODEL,
    DFF,
    GEN_IMAGE_SIZE,
    LATENT_DIM,
    MAX_COMMANDS,
    NUM_GLYPHS,
    NUM_HEADS,
    NUM_TRANSFORMER_LAYERS,
    RATE,
)
from deepvecfont3.model import GlyphGenerator


def test_output_shapes():
    model = GlyphGenerator(
        num_glyphs=NUM_GLYPHS,
        num_transformer_layers=NUM_TRANSFORMER_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dff=DFF,
        latent_dim=LATENT_DIM,
        rate=RATE,
    )

    # Dummy input data
    style_image_input = tf.random.uniform(shape=(BATCH_SIZE, 40, 168, 1))
    glyph_id_input = tf.one_hot(
        tf.random.uniform(
            shape=(BATCH_SIZE,), minval=0, maxval=NUM_GLYPHS, dtype=tf.int32
        ),
        NUM_GLYPHS,
    )
    target_sequence_input = tf.random.uniform(
        shape=(BATCH_SIZE, MAX_COMMANDS, EXTENDED_COMMAND_WIDTH + COORDINATE_WIDTH)
    )

    # Get model output
    outputs = model(
        (style_image_input, glyph_id_input, target_sequence_input)
    )
    generated_glyph_raster = outputs["raster"]
    command_output = outputs["command"]
    coord_output = outputs["coord"]

    # Check output shapes
    assert generated_glyph_raster.shape == (BATCH_SIZE, *GEN_IMAGE_SIZE, 1)
    assert command_output.shape == (BATCH_SIZE, MAX_COMMANDS, EXTENDED_COMMAND_WIDTH)
    assert coord_output.shape == (BATCH_SIZE, MAX_COMMANDS, COORDINATE_WIDTH)
