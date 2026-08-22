"""Carry a scale from the close-up tag photo to the side photograph.

THE PROBLEM
Centimetres come from the ear tag, and the tag is legible only in the app's
dedicated close-up. But every body measurement is made in the SIDE
photograph's pixels, and the two shots were taken from different distances, so
the close-up's centimetres-per-pixel means nothing there. Applying it directly
multiplies every class-C trait by the ratio of the two distances - silently,
producing numbers that still look like plausible centimetres.

THE BRIDGE
One physical object appears in both photographs: the tag itself. So

    1. the close-up gives centimetres-per-pixel, from the 18 mm digit row or
       the 27 mm button - features of KNOWN size;
    2. measuring the tag's panel in those same close-up pixels therefore gives
       the panel's true height in centimetres, for THIS tag;
    3. finding that panel in the side photograph and measuring its height in
       side-photo pixels gives that photograph's own scale.

Step 2 is what makes this legitimate. The panel is normally useless as a ruler
because it is 55-69 mm depending on the supplier - a ~20% unknown that
tag_ruler.py refuses to bake into every measurement. Here it is not assumed:
it is MEASURED on the actual tag, in the close-up, against a feature of known
size. The panel becomes a ruler of known length only because the close-up
calibrated it first.

WHAT IT COSTS
The tag hangs from the ear, which is nearer the camera than the animal's flank
on any real side view. A scale taken at the ear therefore slightly overstates
the scale at the barrel, by roughly the ratio of the two distances - a few per
cent for a photograph taken from any sensible range, more for a close one. That
is carried in the error fraction rather than ignored, and it is the same
limitation the original design had when it measured a tag in the side photo.

WHAT IT REFUSES
Lemon yellow is a common colour on a farm - a plastic drum, a jerrycan, the
back of a truck. A wrong panel gives a wrong scale, and a wrong scale is worse
than no scale, so the gates below are deliberately strict: shape, size,
position relative to the animal, and agreement between candidates. When more
than one plausible panel is found and they disagree, this refuses rather than
picking one.
"""
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from ml.detection.detector import DetectionBackendError


def _get_cv2():
    try:
        import cv2
        return cv2
    except ImportError as exc:
        raise DetectionBackendError(
            "'opencv-python' (cv2) is not installed.") from exc


# What is measured here is the tag's whole VISIBLE YELLOW EXTENT, not the
# printed panel. On a real tag the moulded tab that passes through the ear is
# the same colour and the same piece of plastic, so a colour mask returns
# panel plus tab together: measured on a real photograph, 57 x 97 px, an
# aspect of 1.70 where the published panel alone is 1.18.
#
# That turns out not to matter, and is arguably better. The transfer needs the
# SAME physical extent measured the SAME way in both photographs - its ratio
# is what carries the scale. Whether that extent is the panel or the panel
# plus its tab is irrelevant as long as it is consistent, and the tab is
# present in both shots. Insisting on the published panel ratio instead
# rejected a tag that was correctly found and correctly segmented.
#
# The ranges are therefore loose enough to admit a tag with its tab, and the
# work of rejecting a yellow drum is done by position, fill and agreement
# instead of by a ratio the object does not actually have.
PANEL_ASPECT_RANGE = (0.95, 2.10)          # height / width, tab included
PANEL_HEIGHT_CM_RANGE = (5.0, 12.0)        # panel 5.5-6.9 cm, plus the tab

# Lemon yellow, in HSV. Matches the range tag_ruler already uses so the two
# agree about what a tag looks like.
#
# The lower hue bound is the load-bearing one, because a red Zebu coat is the
# nearest thing on the animal to a lemon tag. Measured on a Sahiwal, which is
# about as red as Indian cattle get:
#
#     tag panel          H 24  S 202  V 237   -> 80.5% inside the gate
#     coat, barrel       H 10  S 205  V 143   ->  1.7%
#     coat, neck         H  8  S 173  V  98   ->  1.2%
#     coat, shoulder     H 11  S 170  V 176   ->  0.0%
#
# So a bound of 15 sits in the gap, with roughly 13 hue units of margin. That
# is real but not generous, and a yellower coat would narrow it - which is why
# a merged blob is caught downstream by the shape and fill gates rather than
# being trusted because it was the largest yellow thing near the head.
YELLOW_LO = (15, 70, 70)
YELLOW_HI = (45, 255, 255)

# Below this the panel is too few pixels for its height to mean anything. At
# 40 px a 2 px edge error is 5%, and weight goes as the cube of the scale.
MIN_PANEL_PX = 40

# The tag is in an ear. Ears are at the head end, in the upper part of the
# animal - never under the belly, never at the tail.
HEAD_FRACTION = 0.45          # of the animal box, from the head end
MAX_DEPTH_FRACTION = 0.62     # of the animal box height, from the top

# Two candidate panels whose implied scales differ by more than this cannot
# both be the tag, and there is no basis for choosing.
CANDIDATE_AGREEMENT = 0.15

# The ear is nearer the camera than the flank. A few per cent on a photograph
# taken from a sensible distance; carried rather than ignored.
PARALLAX_ERROR_FRAC = 0.06


@dataclass
class TransferredScale:
    cm_per_px: float
    panel_height_cm: float
    panel_px_side: float
    error_frac: float
    method: str = "panel-transfer-from-closeup"
    note: str = ""


@dataclass
class TransferRefusal:
    reason: str
    detail: str = ""


def _yellow_boxes(image_bgr, region: Optional[Sequence[float]] = None):
    """Bounding boxes of lemon-yellow blobs, largest first."""
    cv2 = _get_cv2()
    img = image_bgr
    ox = oy = 0
    if region is not None:
        x1, y1, x2, y2 = (int(round(float(v))) for v in region)
        h, w = image_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return []
        img = image_bgr[y1:y2, x1:x2]
        ox, oy = x1, y1
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        out.append((x + ox, y + oy, w, h, float(cv2.contourArea(c))))
    return sorted(out, key=lambda b: -b[4])


def panel_height_cm(closeup_bgr, tag_bbox: Sequence[float],
                    cm_per_px_closeup: float) -> Optional[float]:
    """This tag's actual panel height, in centimetres.

    The one measurement that makes the whole transfer legitimate: the panel is
    not assumed to be any particular size, it is measured against a feature of
    known size in the same photograph.
    """
    if cm_per_px_closeup is None or cm_per_px_closeup <= 0:
        return None
    boxes = _yellow_boxes(closeup_bgr, tag_bbox)
    if not boxes:
        return None
    _x, _y, w, h, _a = boxes[0]
    if h < MIN_PANEL_PX or w <= 0:
        return None
    aspect = h / float(w)
    if not (PANEL_ASPECT_RANGE[0] <= aspect <= PANEL_ASPECT_RANGE[1]):
        return None
    cm = h * cm_per_px_closeup
    lo, hi = PANEL_HEIGHT_CM_RANGE
    return cm if lo <= cm <= hi else None


def _head_region(animal_bbox: Sequence[float], facing: int):
    """Where an ear can be: the head end of the animal, upper part."""
    x1, y1, x2, y2 = (float(v) for v in animal_bbox)
    w, h = x2 - x1, y2 - y1
    if facing < 0:                      # faces left: head at low x
        rx1, rx2 = x1, x1 + HEAD_FRACTION * w
    else:
        rx1, rx2 = x2 - HEAD_FRACTION * w, x2
    return (rx1, y1, rx2, y1 + MAX_DEPTH_FRACTION * h)


def find_panel_px(side_bgr, animal_bbox: Sequence[float], facing: int,
                  expected_aspect: bool = True) -> "float | TransferRefusal":
    """Height in side-photo pixels of the tag panel, or a refusal.

    Restricted to the head end of the animal, which is what keeps a yellow
    drum in the background or a truck across the field from being measured as
    an ear tag.
    """
    region = _head_region(animal_bbox, facing)
    boxes = _yellow_boxes(side_bgr, region)
    if not boxes:
        return TransferRefusal(
            reason="No yellow ear tag was visible in the side photo, so its "
                   "scale could not be established.",
            detail="no lemon-yellow region at the head end of the animal")

    good = []
    for x, y, w, h, area in boxes:
        if h < MIN_PANEL_PX or w <= 0:
            continue
        aspect = h / float(w)
        if expected_aspect and not (
                PANEL_ASPECT_RANGE[0] <= aspect <= PANEL_ASPECT_RANGE[1]):
            continue
        # a tag is a solid panel; a ragged blob of similar colour is not
        if area < 0.55 * w * h:
            continue
        good.append(float(h))

    if not good:
        return TransferRefusal(
            reason="A yellow region was found near the head but it is not "
                   "shaped like an ear tag - move closer so the tag is "
                   "clearly visible in the side photo.",
            detail=f"{len(boxes)} yellow blobs, none panel-shaped and "
                   f"at least {MIN_PANEL_PX}px tall")

    if len(good) > 1:
        lo, hi = min(good), max(good)
        if (hi - lo) / hi > CANDIDATE_AGREEMENT:
            return TransferRefusal(
                reason="More than one possible ear tag was found in the side "
                       "photo and they disagree, so no scale could be trusted.",
                detail=f"candidate panel heights {sorted(good)}")
    return float(np.median(good))


def transfer(side_bgr, animal_bbox: Sequence[float], facing: Optional[int],
             closeup_bgr, tag_bbox: Sequence[float],
             cm_per_px_closeup: float, closeup_error_frac: float = 0.056
             ) -> "TransferredScale | TransferRefusal":
    """Turn a close-up's scale into the side photograph's scale."""
    if facing is None:
        return TransferRefusal(
            reason="The animal's facing direction could not be established, "
                   "so the ear could not be located.",
            detail="facing_sign returned None")

    height_cm = panel_height_cm(closeup_bgr, tag_bbox, cm_per_px_closeup)
    if height_cm is None:
        return TransferRefusal(
            reason="The tag panel could not be measured in the close-up, so "
                   "there is nothing to carry across to the side photo.",
            detail="no panel-shaped yellow region of a plausible size")

    px = find_panel_px(side_bgr, animal_bbox, facing)
    if isinstance(px, TransferRefusal):
        return px

    cm_per_px = height_cm / px
    # the close-up's own error, plus the edge of a small panel, plus parallax
    edge = 2.0 / max(px, 1.0)
    err = float(np.sqrt(closeup_error_frac ** 2 + edge ** 2
                        + PARALLAX_ERROR_FRAC ** 2))
    return TransferredScale(
        cm_per_px=cm_per_px,
        panel_height_cm=height_cm,
        panel_px_side=px,
        error_frac=err,
        note=(f"panel measured at {height_cm:.1f} cm in the close-up, "
              f"{px:.0f} px in the side photo"),
    )
