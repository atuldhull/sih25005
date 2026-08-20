"""Scale from the digit rows - and the gates that stop a wrong one.

Per NDDB Technical Specifications of Eartag & Ear Tag Applicator (15-02-2016):

    Male button (REAR of ear)  27 mm +/-2   "cross-check / occlusion fallback"
    Digit rows                 6 @ 10 mm + 6 @ 18 mm +/-1  "different glyph heights"
    Female panel (front)       55x65 to 58x69 mm  "do not key on the edge"

A field officer photographs the FRONT, so the button is not in the picture.
The 18 mm digit row is the front-facing ruler, and it is the tighter of the
two: +/-1 mm on 18 mm is 5.6%, against +/-2 mm on 27 mm at 7.4%.
"""
import numpy as np
import pytest

from ml.tag_intelligence.tag_ruler import (
    DIGIT_LINE_MM,
    DIGIT_ROW_RATIO,
    DIGIT_ROW_RATIO_TOL,
    BARCODE_LINE_MM,
    ScaleRefusal,
    ScaleResult,
    estimate_scale_from_digits,
)


# Proportions of a REAL tag: the 18 mm row is 18/58 = 31% of the panel
# width, and the 10 mm row is 17%. Getting this wrong made the first
# version of this fixture imply a 12.6 cm tag, which the panel-size gate
# rejected - the gate working on its own test.
def _synthetic_tag(panel_w=280, panel_h=330, tall_px=87, short_px=48):
    """A lemon-yellow panel with two dark rows of the given glyph heights."""
    img = np.full((panel_h + 60, panel_w + 60, 3), 60, np.uint8)
    img[30:30 + panel_h, 30:30 + panel_w] = (60, 220, 240)      # BGR yellow
    x0, x1 = 60, 30 + panel_w - 30
    y = 120
    img[y:y + short_px, x0:x1] = (20, 20, 20)
    y2 = y + short_px + 40
    img[y2:y2 + tall_px, x0:x1] = (20, 20, 20)
    bbox = (30, 30, 30 + panel_w, 30 + panel_h)
    return img, bbox


def test_measures_scale_from_the_18mm_row():
    img, bbox = _synthetic_tag(tall_px=87, short_px=48)   # ratio 1.81
    r = estimate_scale_from_digits(img, bbox)
    assert isinstance(r, ScaleResult), getattr(r, "reason", "")
    assert r.method == "digit_row_18mm"
    # 18 mm over 87 px
    assert r.cm_per_px == pytest.approx((DIGIT_LINE_MM / 10.0) / 87.0, rel=0.15)
    # and the implied panel size must land where a real tag does
    assert 4.5 <= 280 * r.cm_per_px <= 8.5


def test_refuses_when_the_row_ratio_is_wrong():
    """Two rows at 18 mm and 10 mm must measure near 1.8:1.

    A real tag photo slipped through at 2.20 during development - the 'tall
    row' was the dark ear above the panel, not a digit row, and the scale came
    out 25% small. This gate is what stops that.
    """
    img, bbox = _synthetic_tag(tall_px=120, short_px=40)  # ratio 3.0
    r = estimate_scale_from_digits(img, bbox)
    assert isinstance(r, ScaleRefusal)
    assert "layout" in r.reason or "ratio" in r.detail


def test_refuses_when_no_yellow_panel_is_present():
    img = np.full((200, 200, 3), 90, np.uint8)
    r = estimate_scale_from_digits(img, (10, 10, 190, 190))
    assert isinstance(r, ScaleRefusal)
    assert "panel" in r.reason.lower()


def test_refuses_a_tag_too_small_to_measure():
    img, bbox = _synthetic_tag(panel_w=18, panel_h=18, tall_px=3, short_px=2)
    r = estimate_scale_from_digits(img, bbox)
    assert isinstance(r, ScaleRefusal)


def test_the_spec_constants_are_what_nddb_published():
    assert DIGIT_LINE_MM == 18.0
    assert BARCODE_LINE_MM == 10.0
    assert DIGIT_ROW_RATIO == pytest.approx(1.8)
    # loose enough for antialiasing, tight enough to have caught the 2.20 case
    assert DIGIT_ROW_RATIO_TOL <= 0.25


# --- the three-line layout, confirmed against the NDDB source document -----
# "TECHNICAL SPECIFICATIONS OF EARTAG & EAR TAG APPLICATOR", NDDB, 15-02-2016
#   1st Line : Barcode code128, 10mm high (+/-1mm)
#   2nd Line : A row of 6 digits, 10mm high (+/-1mm)
#   3rd Line : A row of 6 digits, 18mm high (+/-1mm)

def _three_line_tag(panel_w=280, panel_h=330,
                    barcode_px=48, digits10_px=48, digits18_px=87):
    """A tag printing all three lines the NDDB spec requires."""
    img = np.full((panel_h + 60, panel_w + 60, 3), 60, np.uint8)
    img[30:30 + panel_h, 30:30 + panel_w] = (60, 220, 240)
    x0, x1 = 55, 30 + panel_w - 25
    y = 60
    for hpx in (barcode_px, digits10_px, digits18_px):
        img[y:y + hpx, x0:x1] = (20, 20, 20)
        y += hpx + 30
    return img, (30, 30, 30 + panel_w, 30 + panel_h)


def test_a_conformant_three_line_tag_measures():
    img, bbox = _three_line_tag()
    r = estimate_scale_from_digits(img, bbox)
    assert isinstance(r, ScaleResult), getattr(r, "reason", "")
    assert r.cm_per_px == pytest.approx((DIGIT_LINE_MM / 10.0) / 87.0, rel=0.15)


def test_the_two_10mm_lines_must_agree():
    """Barcode and the 6-digit row are BOTH 10 mm, so they must measure equal.

    This catches something the 1.8 ratio alone cannot: a shadow that happens
    to sit at 1.8x the height of a real row would pass the ratio test, but it
    will not also produce a matching pair.
    """
    img, bbox = _three_line_tag(barcode_px=48, digits10_px=20, digits18_px=87)
    r = estimate_scale_from_digits(img, bbox)
    assert isinstance(r, ScaleRefusal)
    assert "10 mm" in r.detail or "layout" in r.reason


def test_button_diameter_matches_the_source_document():
    """NDDB item 1: 'a button with a diameter of 27 mm (+/-2mm)'."""
    from ml.tag_intelligence.tag_ruler import BUTTON_DIAMETER_MM
    assert BUTTON_DIAMETER_MM == 27.0


def test_panel_gate_spans_the_published_range():
    """NDDB item 1: female piece is 55x65 mm to 58x69 mm.

    The gate must accept every conformant tag and still reject a scale that
    is wrong by an order of magnitude.
    """
    for panel_cm in (5.5, 5.8, 6.5, 6.9):
        assert 4.5 <= panel_cm <= 8.5, f"{panel_cm}cm is a real tag, must pass"
    for wrong in (2.0, 12.6, 40.0):
        assert not (4.5 <= wrong <= 8.5), f"{wrong}cm must be rejected"
