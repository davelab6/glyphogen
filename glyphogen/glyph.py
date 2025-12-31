from pathlib import Path
from typing import Dict

import numpy as np
import numpy.typing as npt
import pathops
import torch
import uharfbuzz as hb
from fontTools.pens.qu2cuPen import Qu2CuPen
from fontTools.pens.svgPathPen import SVGPathPen, pointToString
from fontTools.ttLib import TTFont
from fontTools.ttLib.removeOverlaps import _simplify
from PIL import Image

from glyphogen.coordinate import to_image_space
from glyphogen.svgglyph import SVGGlyph

from .command_defs import SVGCommand
from .hyperparameters import BASE_DIR, RASTER_IMG_SIZE
from .rasterizer import rasterize_batch

# No point cacheing as we are storing the PNGs in our dataset
CACHING = False


class AbsoluteSVGPathPen(SVGPathPen):
    def _lineTo(self, pt):
        x, y = pt
        # duplicate point
        if x == self._lastX and y == self._lastY:
            return
        # write the string
        t = "L" + " " + pointToString(pt, self._ntos)
        self._lastCommand = "L"
        self._commands.append(t)
        # store for future reference
        self._lastX, self._lastY = pt


cache_dir = Path("imgcache")


class Glyph:
    """A glyph defined by a font file, unicode ID, and location in design space.

    We don't store any vector representation here; that is generated on demand.
    We are simply representing the concept of a source glyph, with the ability
    to extract its vector representation or rasterized image.

    See also SVGGlyph for a vector representation, and NodeGlyph for a
    "designer-like" representation we will use for our model.
    """

    font_file: Path
    unicode_id: int
    location: Dict[str, float]

    def __init__(self, font_file: Path, unicode_id: int, location: Dict[str, float]):
        self.font_file = font_file
        self.unicode_id = unicode_id
        self.location = location

    def rasterize(self, size: int = RASTER_IMG_SIZE) -> npt.NDArray[np.float64]:
        font_base = str(self.font_file).replace(BASE_DIR + "/", "").replace("/", "-")
        key = "-".join(
            [
                str(self.unicode_id),
                ",".join(
                    {f"{k}:{self.location[k]}" for k in sorted(self.location.keys())}
                ),
                str(size),
            ]
        )
        if CACHING and (cache_dir / font_base / (key + ".png")).exists():
            img = Image.open(cache_dir / font_base / (key + ".png")).convert("L")
            img = np.asarray(img, dtype=np.float32) / 255.0
            img = np.expand_dims(img, axis=-1)
        else:
            img = self._rasterize(size)
            if CACHING:
                pil_img = Image.fromarray(
                    (img.squeeze(-1) * 255).astype(np.uint8), mode="L"
                )
                (cache_dir / font_base).mkdir(exist_ok=True)
                print("Saving", font_base, key)
                pil_img.save(cache_dir / font_base / (key + ".png"))
        return img

    def _rasterize(self, size: int) -> npt.NDArray[np.float64]:
        node_glyph = self.vectorize().to_node_glyph()
        contour_sequences = node_glyph.encode(SVGCommand)

        if contour_sequences is None:
            return np.zeros((size, size, 1), dtype=np.float64)

        contour_tensors = []
        for encoded_contour in contour_sequences:
            encoded_tensor = torch.from_numpy(encoded_contour).float()
            cmds_tensor, coords_tensor = SVGCommand.split_tensor(encoded_tensor)

            contour_tensors.append((cmds_tensor, coords_tensor))

        image_tensor = rasterize_batch(
            [contour_tensors],
            SVGCommand,
            seed=42,
            img_size=size,
            requires_grad=False,
            device=torch.device("cpu"),
        )

        numpy_image = image_tensor.squeeze(0).squeeze(0).cpu().numpy()
        return np.expand_dims(numpy_image, axis=-1).astype(np.float64)

    def vectorize(self, remove_overlaps: bool = True) -> SVGGlyph:
        scale = 1000 / TTFont(self.font_file)["head"].unitsPerEm
        blob = hb.Blob.from_file_path(self.font_file)
        face = hb.Face(blob)
        font = hb.Font(face)
        svgpen = AbsoluteSVGPathPen({}, ntos=lambda f: str(int(f * scale)))
        pen = Qu2CuPen(svgpen, max_err=5, all_cubic=True)
        if self.location:
            font.set_variations(self.location)
        glyph = font.get_nominal_glyph(self.unicode_id)
        path = []
        if glyph is None:
            return SVGGlyph([])

        if remove_overlaps:
            skpath = pathops.Path()
            pathPen = skpath.getPen()
            font.draw_glyph_with_pen(glyph, pathPen)
            skpath = _simplify(skpath, chr(self.unicode_id))
            skpath.draw(pen)
        else:
            font.draw_glyph_with_pen(glyph, pen)

        for command in svgpen._commands:
            cmd = command[0] if command[0] != " " else "L"
            coords = [int(p) for p in command[1:].split()]
            if "XAUG" in self.location:
                for i in range(0, len(coords), 2):
                    coords[i] += int(self.location["XAUG"])
            if "YAUG" in self.location:
                for i in range(1, len(coords), 2):
                    coords[i] += int(self.location["YAUG"])

            image_space_coords = []
            for x, y in zip(coords[0::2], coords[1::2]):
                ix, iy = to_image_space((x, y))
                image_space_coords.extend([ix, iy])
            path.append(SVGCommand(cmd, image_space_coords))
        return SVGGlyph(path, "%s, %s" % (self.font_file, chr(self.unicode_id)))
