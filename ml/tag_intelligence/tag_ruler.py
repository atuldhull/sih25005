"""Ear-tag-as-ruler: convert pixels to centimetres from the tag alone.

This is the project's differentiator. Everything measured in centimetres -
stature, body length, rump width, udder depth, teat length - depends on one
number produced here: centimetres per pixel.

WHAT IT MEASURES AGAINST, AND WHY
The NDDB tag's OUTER PANEL is not a ruler. Panel size varies by
manufacturer (roughly 55-69 mm), so calibrating on it bakes a
supplier-dependent error into every measurement. The invariant features are
the printed ones:

    button diameter      27 mm     <- primary reference
    barcode line height  10 mm
    digit line height    18 mm

WHY THE BUTTON, AND WHY AN ELLIPSE
The button is a circle. Photographed at an angle a circle projects to an
ellipse, and to first order the ellipse's MAJOR AXIS still equals the
circle's true diameter in pixels - the foreshortening happens along the
minor axis. So fitting an ellipse to the button and reading its major axis
gives a scale that is naturally robust to the tilt you get from a farmer
holding a phone, WITHOUT needing camera intrinsics at all.

That matters, because the intrinsics available to us are a guess. The
previous config hard-coded focal_length_px = 700, which describes an
800x600 image; on a 4000 px phone photo that is wrong by 4-5x, and any
solvePnP scale built on it inherits the error.

REFUSAL IS A FEATURE
If no button-like ellipse is found with acceptable circularity and size,
this returns None with a reason rather than guessing. A wrong scale is
worse than no scale: it silently corrupts every centimetre trait, whereas
a missing scale makes them refuse honestly.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

# real-world tag dimensions in millimetres (NDDB spec)
BUTTON_DIAMETER_MM = 27.0
BARCODE_LINE_MM = 10.0
DIGIT_LINE_MM = 18.0

# an accepted button must be reasonably round and a sensible share of the tag
MIN_CIRCULARITY = 0.72     # minor/major axis ratio; 1.0 is face-on
# The button is ~27 mm on a panel of ~55-69 mm, so it occupies roughly half
# the panel's short side. Allowing anything near the full crop lets the
# PANEL itself win the fit - which is exactly the failure the synthetic test
# caught: it produced scale errors of 13-75%.
MIN_BUTTON_FRAC = 0.12     # of the tag crop's shorter side
MAX_BUTTON_FRAC = 0.62
MIN_BUTTON_PX = 8.0        # below this, quantisation dominates
# A rectangle can be fitted with an ellipse too. Comparing the contour's own
# area against the fitted ellipse's area rejects it: a real circle matches
# closely, a rectangle over-fills.
MIN_FIT_QUALITY = 0.80
MAX_FIT_QUALITY = 1.20


@dataclass
class ScaleResult:
    cm_per_px: float
    confidence: float          # 0..1
    method: str                # which feature produced it
    button_axes_px: Tuple[float, float]
    circularity: float
    note: str = ""

    @property
    def px_per_cm(self) -> float:
        return 1.0 / self.cm_per_px if self.cm_per_px else float("inf")


@dataclass
class ScaleRefusal:
    reason: str                # farmer-facing explanation
    detail: str = ""


def _crop(image_bgr: np.ndarray, bbox: Sequence[float], pad: float = 0.25):
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - pad * bw))
    y1 = max(0, int(y1 - pad * bh))
    x2 = min(w, int(x2 + pad * bw))
    y2 = min(h, int(y2 + pad * bh))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None, (0, 0)
    return image_bgr[y1:y2, x1:x2], (x1, y1)


def find_button_ellipse(image_bgr: np.ndarray, tag_bbox: Sequence[float]
                        ) -> Optional[Tuple[float, float, float]]:
    """Return (major_px, minor_px, circularity) for the best button candidate.

    Works on the tag crop only, so the search space is tiny and the whole
    thing costs a millisecond.
    """
    import cv2

    crop, _ = _crop(image_bgr, tag_bbox)
    if crop is None:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # light blur only: a heavy kernel spreads the button's edge outward and
    # the fitted ellipse comes back systematically too large
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    short_side = min(crop.shape[0], crop.shape[1])
    best = None

    # Threshold both ways - the button may be lighter OR darker than the
    # panel - and take the FILLED region each time.
    #
    # Canny was tried here and removed: it traces the button's printed RIM,
    # so the fitted ellipse came back ~6 px too large at every size, which
    # is a 30% scale error on a small button. The filled disc is the feature
    # whose diameter is actually 27 mm.
    candidates: List[np.ndarray] = []
    for mode in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):
        mask = cv2.adaptiveThreshold(gray, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     mode, 21, 4)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones((3, 3), np.uint8))
        # RETR_LIST, not RETR_EXTERNAL: the button sits INSIDE the panel,
        # so as a nested contour it is invisible to an external-only search.
        cnts, _ = cv2.findContours(mask, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_NONE)
        candidates.extend(cnts)

    for c in candidates:
        if len(c) < 5:                       # fitEllipse needs 5 points
            continue
        (_, _), (a1, a2), _ = cv2.fitEllipse(c)
        major, minor = max(a1, a2), min(a1, a2)
        if major < MIN_BUTTON_PX:
            continue
        if not (MIN_BUTTON_FRAC * short_side <= major
                <= MAX_BUTTON_FRAC * short_side):
            continue
        circ = minor / major if major else 0.0
        if circ < MIN_CIRCULARITY:
            continue
        # Does the contour actually FILL its fitted ellipse? A rectangle does
        # not, so this is what keeps the panel from being read as the button.
        ellipse_area = np.pi * (major / 2.0) * (minor / 2.0)
        if ellipse_area <= 0:
            continue
        fit_quality = float(cv2.contourArea(c)) / ellipse_area
        if not (MIN_FIT_QUALITY <= fit_quality <= MAX_FIT_QUALITY):
            continue
        # Rank by how circular AND how ellipse-like it is. Deliberately NO
        # size term: an earlier version added one and it made the detector
        # prefer the panel over the button.
        score = circ * min(fit_quality, 1.0 / max(fit_quality, 1e-6))
        if best is None or score > best[0]:
            best = (score, major, minor, circ)

    if best is None:
        return None
    _, major, minor, circ = best
    return major, minor, circ


def estimate_scale(image_bgr: np.ndarray, tag_bbox: Sequence[float]
                   ) -> "ScaleResult | ScaleRefusal":
    """Centimetres per pixel from the ear tag, or a refusal with a reason."""
    found = find_button_ellipse(image_bgr, tag_bbox)
    if found is None:
        return ScaleRefusal(
            reason="Ear tag found but its round button could not be measured "
                   "- retake the photo with the tag facing the camera.",
            detail="no contour passed the circularity and size checks")
    major, minor, circ = found
    cm_per_px = (BUTTON_DIAMETER_MM / 10.0) / major

    # confidence: roundness says how face-on the tag is, size says how much
    # quantisation error we are carrying. A 12 px button measured to +/-1 px
    # is an 8% scale error, and that lands on every centimetre trait.
    size_conf = float(np.clip((major - MIN_BUTTON_PX) / 40.0, 0.0, 1.0))
    round_conf = float(np.clip((circ - MIN_CIRCULARITY)
                               / (1.0 - MIN_CIRCULARITY), 0.0, 1.0))
    confidence = round(0.5 * size_conf + 0.5 * round_conf, 3)

    # SAME PANEL GATE THE DIGIT METHOD USES.
    #
    # The button method had no cross-check at all, and on a real ear-tag
    # photograph it returned a scale implying an 11.4 cm panel - a real NDDB
    # tag is 6.5-6.9 cm. It had fitted an ellipse to something that was not
    # the 27 mm button. A scale that wrong silently multiplies every
    # centimetre trait, so it has to be caught here rather than trusted.
    #
    # Height, not width, for the same reason as the digit method: a tag
    # turned away from the camera keeps its height and loses its width.
    tag_h_px = abs(float(tag_bbox[3]) - float(tag_bbox[1]))
    panel_cm = tag_h_px * cm_per_px
    if tag_h_px > 0 and not (5.0 <= panel_cm <= 9.5):
        return ScaleRefusal(
            reason="The ear tag scale did not check out against the tag's "
                   "own size - retake the photo straight on.",
            detail=f"button gave a scale implying a {panel_cm:.1f} cm tag "
                   f"height; a real NDDB tag is 6.5-6.9 cm, so the measured "
                   f"circle was probably not the 27 mm button")

    note = ""
    if major < 20:
        note = ("button is only {:.0f} px across - scale error is roughly "
                "{:.0f}%; move closer for centimetre traits"
                .format(major, 100.0 / major))
    return ScaleResult(cm_per_px=cm_per_px, confidence=confidence,
                       method="button_ellipse_27mm",
                       button_axes_px=(major, minor), circularity=circ,
                       note=note)


# Edge localisation on a blurred, thresholded disc is good to roughly two
# pixels, not one - measured against synthetic buttons of known size. And no
# scale is claimed to better than 4%, because that is the level at which the
# method has actually been validated; on real NDDB tags it has not yet been
# validated at all.
DEFAULT_PX_UNCERTAINTY = 2.0
MIN_SCALE_ERROR_FRAC = 0.04


def scale_error_fraction(res: ScaleResult,
                         px_uncertainty: float = DEFAULT_PX_UNCERTAINTY
                         ) -> float:
    """Relative error in the scale itself, from +/-px on the major axis.

    This is what makes an honest confidence interval possible: a trait's
    total error is its keypoint error PLUS this scale error, and pretending
    the scale is exact is how a system ends up quoting false precision.
    """
    return max(px_uncertainty / max(res.button_axes_px[0], 1e-6),
               MIN_SCALE_ERROR_FRAC)


def measure_cm(pixel_distance: float, res: ScaleResult,
               keypoint_err_frac: float = 0.0,
               px_uncertainty: float = DEFAULT_PX_UNCERTAINTY
               ) -> Tuple[float, float]:
    """Convert a pixel distance to (centimetres, +/- centimetres).

    Combines the two independent error sources in quadrature: how well the
    keypoints were located, and how well the scale itself is known.
    """
    cm = pixel_distance * res.cm_per_px
    rel = float(np.hypot(keypoint_err_frac, scale_error_fraction(
        res, px_uncertainty)))
    return cm, cm * rel


# ---------------------------------------------------------------------------
# Scale from the DIGIT ROWS - the method that works on a front-facing photo
# ---------------------------------------------------------------------------
# The 27 mm button is on the REAR of the ear (NDDB Technical Specifications of
# Eartag & Ear Tag Applicator, 15-02-2016, which lists it as "cross-check /
# occlusion fallback"). A field officer photographs the FRONT, so the button is
# not in the picture and the button method correctly refuses.
#
# The same spec gives the front-facing features:
#
#     Digit rows    6 @ 10 mm + 6 @ 18 mm, +/-1 mm   "different glyph heights"
#     Barcode line  10 mm tall, +/-1 mm
#     Female panel  55x65 to 58x69 mm  -> "do not key on the edge" (varies ~5%)
#
# So scale comes from the GLYPH HEIGHT of the 18 mm digit row. That is a
# tighter tolerance than the button: +/-1 mm on 18 mm is 5.6%, against +/-2 mm
# on 27 mm at 7.4%. The front-facing feature is the better ruler.
#
# The two rows also give a free validity check. Their true heights are 18 mm
# and 10 mm, so the measured heights must come out near 1.8:1. If they do not,
# something other than the digit rows has been found, and we refuse rather
# than scale the whole scorecard off a shadow.

# CONFIRMED against the primary source: "TECHNICAL SPECIFICATIONS OF EARTAG &
# EAR TAG APPLICATOR", NDDB, 15-02-2016. Item 4, Printing (Laser):
#
#     1st Line : One dimensional Barcode with encoding 128, 10mm high (+/-1mm)
#     2nd Line : A row of 6 digits, 10mm high (+/-1mm)
#     3rd Line : A row of 6 digits, 18mm high (+/-1mm)
#
# So a conformant tag shows THREE ink bands at 10, 10 and 18 mm - not two.
# That is a much stronger check than a single ratio: two of the three bands
# must measure the SAME height, and the third must be 1.8x them. Random ink,
# a shadow, or the dark edge of an ear will not satisfy both at once.
#
# Item 5 gives one more invariant: "Numbers and bar code should be covering
# full size of the female tag and leaving 2 mm margin on all sides", so the
# printed block spans the panel width minus 4 mm.
TAG_MARGIN_MM = 2.0
EQUAL_ROWS_TOL = 0.25      # the two 10 mm rows, measured on ink
DIGIT_ROW_RATIO = DIGIT_LINE_MM / BARCODE_LINE_MM      # 1.8
# Tightened from 0.35 after a real tag slipped through at ratio 2.20: the
# 'tall row' was the dark EAR above the panel, not a digit row, and the
# resulting scale was 25% small. Antialiasing costs about a pixel an edge,
# so 0.20 still has room for that on a legible row.
DIGIT_ROW_RATIO_TOL = 0.20
MIN_GLYPH_PX = 6.0             # below this, +/-1 px is already a 17% error
DIGIT_SCALE_MIN_ERROR_FRAC = 0.056   # the spec's own +/-1 mm on 18 mm


def _ink_bands(panel_gray: np.ndarray, min_frac: float = 0.10
               ) -> List[Tuple[int, int, int]]:
    """Horizontal bands of dark ink: [(top, bottom, width_px), ...].

    Rows are 'inky' when enough of the row is dark. Printed text gives a solid
    band; a shadow or an ear edge does not hold across the panel width, which
    is what min_frac filters out.
    """
    if panel_gray.size == 0:
        return []
    # Threshold against the PANEL's own brightness, not a fixed cutoff. A
    # fixed "< 100" counted shadowed yellow as ink, which merged the top of
    # the tag into one 77px "row" and produced a scale 25% small. Otsu splits
    # ink from panel on whatever exposure the photo happens to have.
    import cv2
    _, binary = cv2.threshold(panel_gray, 0, 1,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = binary.astype(np.uint8)
    h, w = dark.shape
    rowsum = dark.sum(axis=1)
    bands, start = [], None
    for i in range(h):
        on = rowsum[i] > min_frac * w
        if on and start is None:
            start = i
        elif not on and start is not None:
            bands.append((start, i))
            start = None
    if start is not None:
        bands.append((start, h))
    out = []
    for a, b in bands:
        cols = np.where(dark[a:b].sum(axis=0) > 0)[0]
        width = int(cols.max() - cols.min() + 1) if len(cols) else 0
        out.append((a, b, width))
    return out


def estimate_scale_from_digits(image_bgr: np.ndarray,
                               tag_bbox: Sequence[float]
                               ) -> "ScaleResult | ScaleRefusal":
    """Centimetres per pixel from the 18 mm digit row on the tag front."""
    import cv2

    crop, _origin = _crop(image_bgr, tag_bbox, pad=0.02)
    if crop is None or crop.size == 0:
        return ScaleRefusal(reason="Ear tag crop was empty.",
                            detail="bbox outside the image")

    # isolate the lemon-yellow panel; the spec calls out high chroma contrast
    # against every coat colour, which is exactly what makes this findable
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (15, 70, 70), (45, 255, 255))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return ScaleRefusal(
            reason="The yellow tag panel could not be found - retake the "
                   "photo with the tag facing the camera.",
            detail="no lemon-yellow region inside the tag box")
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    if w < 20 or h < 20:
        return ScaleRefusal(
            reason="The ear tag is too small in this photo to measure - "
                   "move closer.",
            detail=f"panel only {w}x{h}px")

    panel = cv2.cvtColor(crop[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    bands = [b for b in _ink_bands(panel) if (b[1] - b[0]) >= MIN_GLYPH_PX]
    if len(bands) < 2:
        return ScaleRefusal(
            reason="The printed rows on the tag could not be measured - "
                   "retake the photo straight on, in better light.",
            detail=f"found {len(bands)} ink band(s), need at least 2")

    # A conformant tag prints three lines: barcode 10 mm, digits 10 mm,
    # digits 18 mm. Take the three widest ink bands and sort by height.
    bands.sort(key=lambda b: -b[2])
    cand = sorted(bands[:3], key=lambda b: -(b[1] - b[0]))
    if len(cand) < 2:
        return ScaleRefusal(reason="Only one printed row was found on the tag.",
                            detail="need at least two rows to cross-check")
    tall_px = float(cand[0][1] - cand[0][0])
    short_px = float(cand[1][1] - cand[1][0])

    # With all three visible, the two 10 mm lines must measure the same. That
    # is an independent check the single ratio cannot give: it rejects a case
    # where one "row" is actually a shadow that happens to sit at 1.8x.
    if len(cand) >= 3:
        a = float(cand[1][1] - cand[1][0])
        b = float(cand[2][1] - cand[2][0])
        if max(a, b) > 0 and abs(a - b) / max(a, b) > EQUAL_ROWS_TOL:
            return ScaleRefusal(
                reason="The tag's printed rows do not match the NDDB layout "
                       "- the scale would not be trustworthy.",
                detail=f"the two 10 mm lines measured {a:.0f}px and {b:.0f}px; "
                       f"a conformant tag prints barcode 10 mm, digits 10 mm "
                       f"and digits 18 mm")
        short_px = (a + b) / 2.0     # average the two 10 mm lines

    # the free validity check: 18 mm over 10 mm must measure near 1.8:1
    ratio = tall_px / max(short_px, 1e-6)
    if abs(ratio - DIGIT_ROW_RATIO) > DIGIT_ROW_RATIO_TOL * DIGIT_ROW_RATIO:
        return ScaleRefusal(
            reason="The tag's printed rows do not match the expected NDDB "
                   "layout - the scale would not be trustworthy.",
            detail=f"row height ratio {ratio:.2f}, expected "
                   f"{DIGIT_ROW_RATIO:.2f} +/-"
                   f"{DIGIT_ROW_RATIO_TOL * DIGIT_ROW_RATIO:.2f}")

    cm_per_px = (DIGIT_LINE_MM / 10.0) / tall_px

    # HARD GATE: does the panel come out the size a real tag is?
    #
    # This is the check that caught a wrong answer during development. The
    # ratio test passed at 2.20 because the "tall row" was the dark ear above
    # the panel rather than a digit row, and the scale came out 25% small.
    # The panel width is a completely independent quantity, so it catches the
    # class of error the ratio cannot.
    #
    # NDDB gives 55-69 mm. Allowing 45-85 mm leaves room for the detector's
    # box padding and a tilted tag while still rejecting anything that would
    # corrupt a scorecard.
    # Check the panel HEIGHT, not its width.
    #
    # This gate originally used the width and it was wrong: a tag turned away
    # from the camera loses width to the cosine while keeping its height, so
    # a perfectly good reading from a tilted tag was rejected as "wrong by
    # 46%". That defeated the entire reason the glyph HEIGHT is the ruler.
    # The cross-check has to be tilt-invariant for the same reason the ruler
    # is. NDDB item 1 gives the female piece as 55x65 to 58x69 mm, so the
    # height is 6.5-6.9 cm; 5.0-9.5 cm leaves room for the detector's box
    # padding and vertical foreshortening.
    panel_cm = h * cm_per_px
    if not (5.0 <= panel_cm <= 9.5):
        return ScaleRefusal(
            reason="The ear tag scale did not check out against the tag's "
                   "own size - retake the photo straight on.",
            detail=f"implied panel height {panel_cm:.1f} cm; a real NDDB tag "
                   f"is 6.5-6.9 cm, so this scale would be wrong by roughly "
                   f"{abs(100 * (panel_cm - 6.7) / 6.7):.0f}%")
    # confidence from glyph size (quantisation) and how well the ratio held
    size_conf = float(np.clip((tall_px - MIN_GLYPH_PX) / 30.0, 0.0, 1.0))
    ratio_conf = float(np.clip(
        1.0 - abs(ratio - DIGIT_ROW_RATIO) / (DIGIT_ROW_RATIO_TOL
                                              * DIGIT_ROW_RATIO), 0.0, 1.0))
    return ScaleResult(
        cm_per_px=cm_per_px,
        confidence=round(0.5 * size_conf + 0.5 * ratio_conf, 3),
        method="digit_row_18mm",
        button_axes_px=(tall_px, short_px),
        circularity=ratio,
        note=("digit row only {:.0f}px tall - scale error is roughly {:.0f}%; "
              "move closer".format(tall_px, 100.0 / tall_px)
              if tall_px < 18 else ""),
    )
