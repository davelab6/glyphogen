from glyphogen.glyph import Glyph
from pathlib import Path

glyph = Glyph(
    Path("/home/simon/others-repos/fonts/ofl/mali/Mali-Light.ttf"),
    ord("z"),
    location={},
)

# Generate vector data first to ensure it's valid
node_glyph = glyph.vectorize().to_node_glyph()
sv_glyph = glyph.vectorize().to_node_glyph()
node_glyph.normalize()  # IMPORTANT: ensure canonical order
encoded_svg = node_glyph.encode()

svg_glyph = glyph.vectorize()
print(svg_glyph.to_svg_string())
segmentation_data = svg_glyph.get_segmentation_data()
print(segmentation_data)
