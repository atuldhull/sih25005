"""Render explainability overlay images.

For a scored trait, draw its keypoints, the connecting line and a
label (trait, score, measured value) on the session photo. This is
the 'proof' screen: exact measurements, not AI heatmaps. Drawn once
per trait, then served from disk cache.

If the photo can't be opened (e.g. test uploads), we draw on a
placeholder canvas so the flow still works end-to-end.
"""
from PIL import Image, ImageDraw, ImageFont

ACCENT = (255, 140, 0)
POINT_RADIUS = 9


def _font(size=26):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render_overlay(photo_path, trait, out_path):
    try:
        img = Image.open(photo_path).convert("RGB")
    except Exception:
        img = Image.new("RGB", (1000, 700), (55, 58, 64))
        ImageDraw.Draw(img).text(
            (30, 330), "photo preview unavailable - keypoints shown on placeholder",
            fill=(200, 200, 200), font=_font(22))

    draw = ImageDraw.Draw(img)
    w, h = img.size
    points = [(min(max(int(x), 0), w - 1), min(max(int(y), 0), h - 1))
              for x, y in trait.get("overlay_points", [])]

    if len(points) >= 2:
        draw.line(points, fill=ACCENT, width=5)
    for x, y in points:
        draw.ellipse([x - POINT_RADIUS, y - POINT_RADIUS,
                      x + POINT_RADIUS, y + POINT_RADIUS],
                     fill=ACCENT, outline=(255, 255, 255), width=2)

    label = trait["name"]
    if trait.get("score") is not None:
        label += f"  |  score {trait['score']}"
    if trait.get("measured_value"):
        label += f"  |  {trait['measured_value']}"

    font = _font()
    box = draw.textbbox((0, 0), label, font=font)
    pad = 10
    draw.rectangle([10, 10, 10 + (box[2] - box[0]) + 2 * pad,
                    10 + (box[3] - box[1]) + 2 * pad], fill=(0, 0, 0))
    draw.text((10 + pad, 10 + pad), label, fill=(255, 255, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "JPEG", quality=88)
