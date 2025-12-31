from glyphogen.svgglyph import SVGGlyph
from glyphogen.command_defs import SVGCommand
from glyphogen.rasterizer import rasterize_batch
import torch


def test_rasterizer():
    g = SVGGlyph.from_svg_string("M 81 343 L 62 343 L 62 98 L 81 98 L 81 343 Z")
    ng = g.to_node_glyph()
    assert len(ng.contours) == 1
    assert len(ng.contours[0].nodes) == 4

    encoded = ng.encode(SVGCommand)
    assert encoded is not None
    assert len(encoded) == 1  # One contour
    inputs = [SVGCommand.split_tensor(torch.tensor(c)) for c in encoded]
    raster_img = rasterize_batch(
        [inputs],
        SVGCommand,
        img_size=64,
        device=torch.device("cpu"),
    )
