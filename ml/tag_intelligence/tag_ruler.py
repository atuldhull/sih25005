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
