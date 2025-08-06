from pathlib import Path
from deepvecfont3.glyph import Glyph, UnrelaxedSVG
from deepvecfont3.hyperparameters import GEN_IMAGE_SIZE


def test_glyph_extraction():
    font_path = Path("tests/data/NotoSans[wdth,wght].ttf")
    location = {"wght": 400.0}
    codepoint = 97  # 'a'

    glyph = Glyph(font_path, codepoint, location)

    # Rasterize the glyph
    rasterized_glyph = glyph.rasterize(GEN_IMAGE_SIZE[0])

    # Check the size of the rasterized numpy array
    assert rasterized_glyph.shape == (*GEN_IMAGE_SIZE, 1)

    # import matplotlib.pyplot as plt

    # plt.imshow(rasterized_glyph.squeeze(), cmap="gray")
    # plt.axis("off")
    # plt.show()

    # Vectorize the glyph
    vectorized_glyph = glyph.vectorize()

    assert isinstance(vectorized_glyph, UnrelaxedSVG)
    assert len(vectorized_glyph.commands) > 0
    assert vectorized_glyph.commands[0].command == "M"
