"""Stage 3: turn the ear tag into a centimetre scale.

WHY NOT solvePnP ON THE TAG CORNERS
The original plan was corner detection plus solvePnP. That needs the outer
panel's dimensions, and the outer panel is 55-69 mm depending on which
supplier printed it. Using it as a ruler bakes an unknown 20% error into
every class-C trait, and the error is invisible - the number just comes out
wrong. So scale comes from the round BUTTON instead, whose diameter is a
fixed 27 mm.

WHY THE BUTTON WORKS
Under perspective a circle projects to an ellipse, and the ellipse's MAJOR
axis stays equal to the true diameter to first order however the tag is
tilted. That gives a scale without knowing anything about the camera. The
minor axis shrinks with tilt, so the ratio of the two also tells us how
face-on the tag is, which becomes the confidence.

WHAT THIS DOES NOT DO
OCR and identity resolution are not implemented. The animal_id still comes
from the BPA record the server passes in. This module answers one question -
how many centimetres is a pixel - and refuses when it cannot answer it.

Three bugs found by synthetic tests before this was trusted, all of which
produced plausible-looking wrong answers rather than failures:
  * scoring candidates by circularity plus a size bonus made the detector
    prefer the whole panel over the button (13-75% scale error)
  * Canny traced both edges of the button rim, adding a constant +6 px
  * RETR_EXTERNAL could not see the button nested inside the panel outline
Fixed with a fit-quality gate, filled-disc thresholding, and RETR_LIST.
19/19 synthetic checks pass. On real NDDB tags it is not yet validated,
which is why no scale is claimed to better than 4%.
"""
from typing import Any, Dict, Optional, Sequence

from ml.config.models import TAG_SCALE_METHOD


# KNOWN LIMITATION, found by running this on a real NDDB tag photo
# ---------------------------------------------------------------
# The 27 mm button is on the BACK of the tag - the disc that sits behind the
# ear. A photo of the tag FRONT (the yellow panel with "IND 23" and the
# 12-digit number, which is what a field officer naturally photographs) shows
# only the small attachment stud, roughly 6 mm. Measuring that stud as if it
# were the 27 mm button would report a scale about 4.5x too small, and every
# centimetre trait would inherit it.
#
# So on a front-facing tag photo this module correctly REFUSES rather than
# returning a confident wrong number. That refusal is why class-C traits
# currently report not_scored_reason; the class-A angle traits are unaffected
# because angles need no scale.
#
# Two ways forward, in order of preference:
#
#   1. Use an invariant PRINTED feature, which is visible from the front.
#      config/tag_spec.py already carries digit_line_cm = 1.8 and
#      barcode_line_cm = 1.0, and its comment states these do not vary by
#      manufacturer while the outer panel does (55-69 mm). This is the right
#      answer - but it is NOT implemented here, because "digit_line_cm = 1.8"
#      is ambiguous between the height of the digit row and its width, and
#      those differ by roughly 5x. Getting it wrong would be indistinguishable
#      from getting it right, which is exactly the failure this module exists
#      to avoid. Confirm which dimension it means before implementing.
#
#   2. Capture the tag back. Reliable, since the 27 mm button is then face-on,
#      but it asks the officer to photograph the far side of the ear.
#
# Detection itself is fine: the tag is found on a close-up at 0.84 confidence.
# It is only the SCALE that is unavailable from the front.


class TagScaleRefused(Exception):
    """Carries the reason a scale could not be measured, for the farmer."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def measure_scale(image_bgr, tag_bbox: Sequence[float]) -> Dict[str, Any]:
    """Centimetres per pixel from the ear-tag button.

    Returns {cm_per_px, confidence, error_frac, method, note}. The caller
    passes cm_per_px straight to measure_all_traits() as scale_factor, which
    multiplies pixel distances by it - so the units line up with no
    conversion, and a units mistake here would be caught by chain_test.py's
    "is this measurement biologically possible" check.

    Raises TagScaleRefused when the button cannot be measured. That is a
    designed outcome, not a failure: without a scale the class-C traits
    report not_scored_reason and the class-A angle traits still work, because
    angles need no scale at all.
    """
    from ml.tag_intelligence.tag_ruler import (ScaleResult, estimate_scale,
                                               scale_error_fraction)

    res = estimate_scale(image_bgr, tag_bbox)
    if not isinstance(res, ScaleResult):
        raise TagScaleRefused(getattr(res, "reason", "scale unavailable"),
                              getattr(res, "detail", ""))
    return {
        "cm_per_px": float(res.cm_per_px),
        "confidence": float(res.confidence),
        "error_frac": float(scale_error_fraction(res)),
        "method": TAG_SCALE_METHOD,
        "button_major_px": float(res.button_axes_px[0]),
        "circularity": float(res.circularity),
        "note": res.note or "",
    }


def read_tag(image_bgr, tag_bbox: Sequence[float]) -> Dict[str, Any]:
    """Everything we can currently get from the tag.

    identity is None on purpose - OCR is not implemented, and inventing an
    ear-tag number would be far worse than admitting we cannot read one.
    """
    out: Dict[str, Any] = {"identity": None, "scale": None,
                           "refused_reason": None}
    try:
        out["scale"] = measure_scale(image_bgr, tag_bbox)
    except TagScaleRefused as exc:
        out["refused_reason"] = exc.reason
    return out


def scale_factor_from(tag_result: Optional[Dict[str, Any]]):
    """(scale_factor, confidence) for the measurement stage, or (None, 0.0).

    None means "no centimetre scale". Measurement already refuses class-C
    traits on a None scale, so this is the single place that decides whether
    the run produces centimetres or only angles and ratios.
    """
    if not tag_result or not tag_result.get("scale"):
        return None, 0.0
    s = tag_result["scale"]
    return s["cm_per_px"], s["confidence"]
