import io
import subprocess
from typing import Dict

from keras.utils import img_to_array
from PIL import Image, ImageChops

from deepvecfont3.hyperparameters import STYLE_IMAGE_SIZE, GEN_IMAGE_SIZE


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    return im


def render(
    font,
    variation: Dict[str, float] = {},
    text="hamburgefontsiv",
    target_size=GEN_IMAGE_SIZE,
):
    if not variation:
        vars_text = "--variations=wght=400"
    else:
        vars_text = "--variations=" + ",".join(
            [f"{k}={v}" for k, v in variation.items()]
        )
    image = subprocess.run(
        [
            "hb-view",
            "-o",
            "-",
            "-O",
            "png",
            vars_text,
            "--font-size=128",
            font,
            text,
        ],
        check=True,
        capture_output=True,
    ).stdout
    image = trim(Image.open(io.BytesIO(image)))
    width, height = image.size
    scale = min(target_size[0] / width, target_size[1] / height)
    new_img = image.resize((int(width * scale), int(height * scale)))
    width, height = new_img.size

    new_img2 = Image.new("L", target_size)
    # White background
    new_img2.paste(255, (0, 0, target_size[0], target_size[1]))
    new_img2.paste(
        new_img,
        (int((target_size[0] - width) / 2), int((target_size[1] - height) / 2)),
    )
    return img_to_array(new_img2) / 255.0


def get_style_image(font, variation):
    return render(
        font, variation=variation, text="hamburgefonsiv", target_size=STYLE_IMAGE_SIZE
    )
