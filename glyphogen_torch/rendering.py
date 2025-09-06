import io
import subprocess
from typing import Dict

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageChops
import diskcache
from glyphogen_torch.hyperparameters import GEN_IMAGE_SIZE, STYLE_IMAGE_SIZE

cache_dir = "imgcache"
cache = diskcache.Cache(cache_dir, size_limit=4 * 2**30)  # 4 GB


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    return im


@cache.memoize()
def _render(vars_text, font, text):
    return subprocess.run(
        [
            "hb-view",
            "-o",
            "-",
            "-O",
            "png",
            vars_text,
            "--font-size=1024",
            font,
            text,
        ],
        check=True,
        capture_output=True,
    ).stdout


def render(
    font,
    variation: Dict[str, float] = {},
    text="hamburgefontsiv",
    target_size=GEN_IMAGE_SIZE,
    do_trim=False,
):
    if not variation:
        vars_text = "--variations=wght=400"
    else:
        vars_text = "--variations=" + ",".join(
            [f"{k}={v}" for k, v in variation.items()]
        )
    image = _render(vars_text, font, text)
    image = Image.open(io.BytesIO(image))
    width, height = image.size
    if do_trim:
        image = trim(image)
        width, height = image.size
        scale = min(target_size[0] / width, target_size[1] / height)
        new_img = image.resize((int(width * scale), int(height * scale)))
        width, height = new_img.size
    else:
        new_img = image

    new_img2 = Image.new("L", target_size)
    # White background
    new_img2.paste(255, (0, 0, target_size[0], target_size[1]))
    if do_trim:
        new_img2.paste(
            new_img,
            (int((target_size[0] - width) / 2), int((target_size[1] - height) / 2)),
        )
    else:
        # Paste it at known coordinates. This is the bit that nobody
        # thinks about. The X coordinate is easy, we'll put it at 0.
        # The Y coordinate is a bit more tricky, because we want the
        # glyph to be aligned on the baseline, regardless of the font's
        # vertical metrics.
        ttFont = TTFont(font)
        upem = ttFont["head"].unitsPerEm
        ascent = ttFont["hhea"].ascent
        # We've received a glyph at upem scale, which means that the height of
        # the glyph image is ascent - descent. We want to scale and position the
        # glyph such that (a) all glyphs are scaled by the same amount, regardless
        # of their vertical metrics, and (b) the baseline is three quarters of the way
        # up the image. When we scale it let's assume that all glyphs fit inside
        # 1.5x the upem. So we have one upem worth of ascent above the baseline
        # and 0.5 upem worth of descent below the baseline.
        scale = target_size[1] / (1.5 * upem)
        new_img = new_img.resize(
            (int(new_img.width * scale), int(new_img.height * scale))
        )
        scaled_ascent = ascent * scale
        baseline_y = int(target_size[1] * 0.66)
        new_img2.paste(new_img, (0, int(baseline_y - scaled_ascent)))
    new_img2 = np.asarray(new_img2, dtype=np.float32)
    new_img2 = new_img2.reshape((new_img2.shape[0], new_img2.shape[1], 1))
    return new_img2 / 255.0


def get_style_image(font, variation):
    return render(
        font,
        variation=variation,
        text="hamburgefonsiv",
        target_size=STYLE_IMAGE_SIZE,
        do_trim=True,
    )
