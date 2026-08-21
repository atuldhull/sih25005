"""Render an NDDB-style ear tag for a given 12-digit animal id.

    python scripts/make_tag.py 356279812347 out.jpg

WHY THIS EXISTS
The demo needs one tag per animal, and the tag is not decoration: it is the
only object of known size in a capture, so every centimetre trait and the
weight is derived from its printed digit rows. A tag whose geometry drifts
changes the measurements.

So the layout here is not chosen, it is inherited. The band positions and
heights are exactly 3x the asset the pipeline was tuned against, and the ratio
that actually reaches the body measurements is

    panel_height_cm = yellow_panel_px * (1.8 / tall_ink_band_px)

which comes out at 6.70 cm for any uniform scaling of this layout. Change a
band height and you change every animal's weight.

THE DETECTED BANDS ARE NOT THE DRAWN BANDS. Glyphs with curved tops (8, 2, 3,
6, 9) are anti-aliased, and their outermost rows land on the wrong side of the
reader's Otsu split, so a row drawn 180px tall is detected as 177-178. The
barcode, with flat crisp bar tops, measures exactly. This script therefore
renders, measures what the reader would actually see, corrects, and re-renders
until the DETECTED heights match the target. Skipping that loop drifts the
scale by about 1%.

The barcode is a genuine Code 128 subset C encoding of the id, with a real
mod-103 check character - and it is decoded back out of the finished JPEG
before the file is accepted, so a mistyped pattern cannot ship silently.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- Code 128, values 0..106. Each string is bar/space widths in modules. ---
_PATTERNS = (
    "212222 222122 222221 121223 121322 131222 122213 122312 132212 221213 "
    "221312 231212 112232 122132 122231 113222 123122 123221 223211 221132 "
    "221231 213212 223112 312131 311222 321122 321221 312212 322112 322211 "
    "212123 212321 232121 111323 131123 131321 112313 132113 132311 211313 "
    "231113 231311 112133 112331 132131 113123 113321 133121 313121 211331 "
    "231131 213113 213311 213131 311123 311321 331121 312113 312311 332111 "
    "314111 221411 431111 111224 111422 121124 121421 141122 141221 112214 "
    "112412 122114 122411 142112 142211 241211 221114 413111 241112 134111 "
    "111242 121142 121241 114212 124112 124211 411212 421112 421211 212141 "
    "214121 412121 111143 111341 131141 114113 114311 411113 411311 113141 "
    "114131 311141 411131 211412 211214 211232 2331112"
).split()

START_C, STOP = 105, 106


def code128c(digits: str) -> list[int]:
    """Bar/space module widths for a Code 128 subset C symbol."""
    if len(digits) % 2:
        raise ValueError("subset C encodes digit PAIRS - length must be even")
    values = [int(digits[i:i + 2]) for i in range(0, len(digits), 2)]
    checksum = (START_C + sum((i + 1) * v for i, v in enumerate(values))) % 103
    widths = []
    for value in [START_C, *values, checksum, STOP]:
        widths.extend(int(c) for c in _PATTERNS[value])
    return widths


def _font(size: int, narrow: bool = False):
    for name in (("arialnb.ttf", "arialbd.ttf") if narrow else ("arialbd.ttf",)):
        try:
            return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)
        except OSError:
            continue
    return ImageFont.load_default()


def _fit_text(draw, text, box, font_size, narrow):
    """Draw text stretched to exactly fill box=(x0, y0, x1, y1)."""
    x0, y0, x1, y1 = box
    font = _font(font_size, narrow)
    tmp = Image.new("L", (1, 1))
    l, t, r, b = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=font)
    tw, th = max(1, r - l), max(1, b - t)
    layer = Image.new("L", (tw, th), 0)
    ImageDraw.Draw(layer).text((-l, -t), text, font=font, fill=255)
    layer = layer.resize((x1 - x0, y1 - y0), Image.LANCZOS)
    return layer, (x0, y0)


# Geometry, 3x the tuned asset. Do not edit without re-reading the docstring.
W, H = 1440, 1638
PANEL = (216, 216, 1224, 1422)              # x0, y0, x1, y1
BANDS = [(252, 180), (504, 180), (756, 324)]  # (y_top, height) barcode, 10mm, 18mm
MARGIN = 36


def render(animal_id: str, dy=(0, 0, 0)) -> np.ndarray:
    px0, py0, px1, py1 = PANEL
    pw = px1 - px0

    img = Image.new("RGB", (W, H), (214, 214, 212))
    d = ImageDraw.Draw(img)

    # backdrop: a light pool behind the object, then a vignette
    d.ellipse([W * 0.10, H * 0.06, W * 0.90, H * 0.94], fill=(232, 232, 230))

    # the plastic: marigold at the top easing to mustard at the bottom
    plate = Image.new("RGB", (pw, py1 - py0))
    pd = ImageDraw.Draw(plate)
    for y in range(py1 - py0):
        f = y / (py1 - py0)
        pd.line([(0, y), (pw, y)], fill=(int(255 - 29 * f), int(201 - 45 * f), int(26 - 14 * f)))
    # satin sheen across the upper third
    sheen = Image.new("L", (pw, py1 - py0), 0)
    ImageDraw.Draw(sheen).ellipse([-pw * 0.3, -(py1 - py0) * 0.5, pw * 1.3, (py1 - py0) * 0.42], fill=54)
    plate = Image.composite(Image.new("RGB", plate.size, (255, 236, 150)), plate, sheen.point(lambda v: v))

    mask = Image.new("L", (pw, py1 - py0), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, pw - 1, py1 - py0 - 1], radius=66, fill=255)
    img.paste(plate, (px0, py0), mask)

    # printed content
    ink = Image.new("L", (W, H), 0)
    idraw = ImageDraw.Draw(ink)

    # band 1: the barcode
    by, bh = BANDS[0][0], BANDS[0][1] + dy[0]
    widths = code128c(animal_id)
    module = (pw - 2 * MARGIN) / sum(widths)
    x = px0 + MARGIN
    for i, w in enumerate(widths):
        if i % 2 == 0:                              # even index = bar
            idraw.rectangle([round(x), by, round(x + w * module) - 1, by + bh], fill=255)
        x += w * module

    # bands 2 and 3: 6 digits at 10 mm, 6 digits at 18 mm (NDDB layout)
    for idx, (text, narrow) in enumerate(((animal_id[:6], False), (animal_id[6:], True)), start=1):
        y, h = BANDS[idx][0], BANDS[idx][1] + dy[idx]
        layer, at = _fit_text(idraw, text, (px0 + MARGIN, y, px1 - MARGIN, y + h), 300, narrow)
        ink.paste(layer, at, layer)

    img = Image.composite(Image.new("RGB", img.size, (18, 18, 18)), img, ink)

    # the moulded fastening boss a front-on photo actually shows - NOT the
    # 27mm rear button, which would merge with the 18mm digit row and make the
    # digit-row scale method refuse outright
    d = ImageDraw.Draw(img)
    cx, cy = (px0 + px1) // 2, 1230
    d.ellipse([cx - 84, cy - 84, cx + 84, cy + 84], fill=(226, 165, 16))
    d.ellipse([cx - 66, cy - 66, cx + 66, cy + 66], fill=(243, 190, 40))
    d.ellipse([cx - 30, cy - 30, cx + 30, cy + 30], fill=(28, 24, 12))

    out = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # contact shadow, clipped to below the tag so it cannot darken the panel
    # corners into a spurious fourth ink band
    sh = np.zeros((H, W), np.float32)
    cv2.ellipse(sh, (W // 2, py1 + 40), (pw // 2 - 20, 40), 0, 0, 360, 1.0, -1)
    sh = cv2.GaussianBlur(sh, (0, 0), 26)
    sh[:py1] = 0
    out = np.clip(out.astype(np.float32) * (1 - 0.34 * sh[..., None]), 0, 255).astype(np.uint8)

    grain = np.random.default_rng(11).normal(0, 2.1, out.shape)
    return np.clip(out.astype(np.float32) + grain, 0, 255).astype(np.uint8)


def detected_bands(bgr) -> list[int]:
    """The band heights the pipeline's reader would actually find."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = (th > 0).mean(axis=1)
    runs, start = [], None
    for y, v in enumerate(dark):
        if v > 0.10 and start is None:
            start = y
        elif v <= 0.10 and start is not None:
            runs.append(y - start)
            start = None
    if start is not None:
        runs.append(len(dark) - start)
    return [r for r in runs if r > 20]


def make(animal_id: str, out_path: str) -> dict:
    target = [b[1] for b in BANDS]
    dy = [0, 0, 0]
    for _ in range(6):                       # converge on the DETECTED heights
        img = render(animal_id, tuple(dy))
        got = detected_bands(img)
        if len(got) != 3:
            raise RuntimeError(f"expected 3 ink bands, reader found {got}")
        if got == target:
            break
        dy = [dy[i] + (target[i] - got[i]) for i in range(3)]

    cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 94])

    # verify from the FILE, not from memory
    saved = cv2.imread(out_path)
    bands = detected_bands(saved)
    decoded = None
    try:
        from pyzbar.pyzbar import decode
        hits = decode(Image.fromarray(cv2.cvtColor(saved, cv2.COLOR_BGR2RGB)))
        decoded = hits[0].data.decode() if hits else None
    except Exception:
        decoded = "pyzbar unavailable"
    return {"path": out_path, "bands": bands, "target": target,
            "barcode_decodes_to": decoded, "matches_id": decoded == animal_id}


if __name__ == "__main__":
    if len(sys.argv) != 3 or not sys.argv[1].isdigit() or len(sys.argv[1]) != 12:
        sys.exit("usage: make_tag.py <12-digit-animal-id> <out.jpg>")
    animal_id, out = sys.argv[1], sys.argv[2]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    r = make(animal_id, out)
    print(f"  {out}")
    print(f"    detected bands {r['bands']}  target {r['target']}"
          f"  {'OK' if r['bands'] == r['target'] else 'DRIFTED'}")
    print(f"    barcode decodes to {r['barcode_decodes_to']!r}"
          f"  {'OK' if r['matches_id'] else 'MISMATCH'}")
