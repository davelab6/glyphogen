# import torch

# from glyphogen.command_defs import COORDINATE_WIDTH, NODE_COMMAND_WIDTH
# from glyphogen.hyperparameters import (
#     BATCH_SIZE,
#     D_MODEL,
#     GEN_IMAGE_SIZE,
#     LATENT_DIM,
#     MAX_COMMANDS,
#     RATE,
# )
# from glyphogen.model import VectorizationGenerator

# NOTE: This test is obsolete as of 2026-01-01.
# It was written for a previous version of the model's API which took a simple
# dictionary of tensors. The new model uses a `CollatedGlyphData` object
# from a custom collate_fn, and the `forward` method's signature and return
# value have changed completely. The output shapes are implicitly tested
# by the more meaningful round-trip tests.

# def test_output_shapes():
#     segmenter_state = None  # For this test, we don't need a pre-trained segmenter
#     model = VectorizationGenerator(
#         segmenter_state=segmenter_state,
#         d_model=D_MODEL,
#         latent_dim=LATENT_DIM,
#         rate=RATE,
#     )

#     # Dummy input data
#     raster_image_input = torch.rand(BATCH_SIZE, 1, *GEN_IMAGE_SIZE)
#     target_sequence_input = torch.rand(
#         BATCH_SIZE, MAX_COMMANDS, NODE_COMMAND_WIDTH + COORDINATE_WIDTH
#     )

#     # Get model output
#     inputs = {
#         "raster_image": raster_image_input,
#         "target_sequence": target_sequence_input,
#     }
#     outputs = model(inputs)
#     command_output = outputs["command"]
#     coord_output = outputs["coord_absolute"]

#     # Check output shapes
#     assert command_output.shape == (BATCH_SIZE, MAX_COMMANDS, NODE_COMMAND_WIDTH)
#     assert coord_output.shape == (BATCH_SIZE, MAX_COMMANDS, COORDINATE_WIDTH)
