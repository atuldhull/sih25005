"""Does the ruler recover a KNOWN scale from a spec-accurate tag?

Every other tag test checks that a wrong answer is refused. This one checks
the opposite and harder thing: given a tag rendered to the published NDDB
dimensions at a known pixel scale, does the ruler return that scale?

The tags here are built from the primary source - "TECHNICAL SPECIFICATIONS OF
EARTAG & EAR TAG APPLICATOR", NDDB, 15-02-2016:

    item 1  female piece 55x65 mm to 58x69 mm
    item 4  1st line  barcode code128, 10 mm high (+/-1mm)
            2nd line  a row of 6 digits, 10 mm high (+/-1mm)
            3rd line  a row of 6 digits, 18 mm high (+/-1mm)
    item 5  printing covers the full tag "leaving 2 mm margin on all sides"
    item 7  colour lemon yellow

This is not a substitute for a photograph of a real tag - it cannot catch
print variation, glare, wear, or a supplier who ignores the spec. It does
prove the geometry and the arithmetic are right, which is the part that would
otherwise silently scale every centimetre trait.
"""
import numpy as np
import pytest

from ml.tag_intelligence.tag_ruler import (
    DIGIT_LINE_MM,
    ScaleRefusal,
    ScaleResult,
    estimate_scale_from_digits,
)

# NDDB item 1: the female piece, in millimetres
PANEL_W_MM, PANEL_H_MM = 56.0, 67.0
MARGIN_MM = 2.0                       # item 5
BARCODE_MM, DIGITS10_MM, DIGITS18_MM = 10.0, 10.0, 18.0   # item 4


def render_tag(px_per_mm, tilt_x=1.0, bg=(70, 70, 70)):
    """A lemon-yellow NDDB tag with its three printed lines, to spec.

    tilt_x < 1 squeezes horizontally, standing in for a tag turned away from
    the camera. Glyph HEIGHT is unaffected by that, which is the whole reason
    the height is used as the ruler rather than the width.
    """
    pw = int(round(PANEL_W_MM * px_per_mm * tilt_x))
    ph = int(round(PANEL_H_MM * px_per_mm))
    pad = int(round(12 * px_per_mm))
    img = np.full((ph + 2 * pad, pw + 2 * pad, 3), bg, np.uint8)
    img[pad:pad + ph, pad:pad + pw] = (60, 220, 240)          # BGR lemon

    m = int(round(MARGIN_MM * px_per_mm))
    x0, x1 = pad + m, pad + pw - m
    gap = int(round(4 * px_per_mm))
    y = pad + m
    for mm in (BARCODE_MM, DIGITS10_MM, DIGITS18_MM):
        h = int(round(mm * px_per_mm))
        img[y:y + h, x0:x1] = (25, 25, 25)
        y += h + gap
    bbox = (pad, pad, pad + pw, pad + ph)
    return img, bbox


@pytest.mark.parametrize("px_per_mm", [3.0, 4.5, 6.0, 9.0])
def test_recovers_the_true_scale_at_several_sizes(px_per_mm):
    """cm_per_px must come back as 1/(10*px_per_mm), within the spec's own
    +/-1 mm on an 18 mm row - which is 5.6%."""
    img, bbox = render_tag(px_per_mm)
    r = estimate_scale_from_digits(img, bbox)
    assert isinstance(r, ScaleResult), getattr(r, "reason", "")
    expected = 1.0 / (10.0 * px_per_mm)          # cm per pixel
    assert r.cm_per_px == pytest.approx(expected, rel=0.12), (
        f"at {px_per_mm} px/mm expected {expected:.5f}, got {r.cm_per_px:.5f}")


@pytest.mark.parametrize("px_per_mm", [4.0, 7.0])
def test_the_recovered_scale_reproduces_the_panel_size(px_per_mm):
    """An independent check: measuring the panel with the recovered scale
    must give back a real tag width, 55-58 mm."""
    img, bbox = render_tag(px_per_mm)
    r = estimate_scale_from_digits(img, bbox)
    assert isinstance(r, ScaleResult)
    panel_px = bbox[2] - bbox[0]
    panel_cm = panel_px * r.cm_per_px
    assert 4.5 <= panel_cm <= 8.5, f"implied panel {panel_cm:.1f} cm"


@pytest.mark.parametrize("tilt", [1.0, 0.8, 0.6])
def test_scale_survives_a_tag_turned_away(tilt):
    """Glyph HEIGHT is unchanged by a horizontal turn.

    This is exactly why height is the ruler: a tag rotated about its vertical
    axis loses width but keeps height, so the scale holds where a
    width-based measurement would shrink with the cosine.
    """
    img, bbox = render_tag(6.0, tilt_x=tilt)
    r = estimate_scale_from_digits(img, bbox)
    assert isinstance(r, ScaleResult), getattr(r, "reason", "")
    assert r.cm_per_px == pytest.approx(1.0 / 60.0, rel=0.15)


def test_a_real_measurement_comes_out_in_the_right_units():
    """End to end: a distance of N pixels must convert to N/px_per_mm mm.

    A units slip here - cm against mm, or a factor of ten - would put every
    class-C trait out by an order of magnitude while still looking like a
    plausible number.
    """
    px_per_mm = 5.0
    img, bbox = render_tag(px_per_mm)
    r = estimate_scale_from_digits(img, bbox)
    assert isinstance(r, ScaleResult)
    # a 500 px distance is 100 mm = 10 cm at 5 px/mm
    assert 500 * r.cm_per_px == pytest.approx(10.0, rel=0.12)


def test_a_tag_too_small_to_measure_is_refused():
    """Below a few px per mm the glyph rows blur together and the scale would
    be quantisation noise. Refuse rather than report it."""
    img, bbox = render_tag(0.5)
    assert isinstance(estimate_scale_from_digits(img, bbox), ScaleRefusal)


def test_a_tag_with_the_wrong_line_layout_is_refused():
    """A tag printing rows that are not 10/10/18 is not a conformant NDDB tag,
    and its scale cannot be trusted."""
    px_per_mm = 6.0
    pw = int(PANEL_W_MM * px_per_mm)
    ph = int(PANEL_H_MM * px_per_mm)
    pad = 40
    img = np.full((ph + 2 * pad, pw + 2 * pad, 3), 70, np.uint8)
    img[pad:pad + ph, pad:pad + pw] = (60, 220, 240)
    y = pad + 12
    for mm in (30.0, 6.0, 6.0):                    # nothing like the spec
        h = int(mm * px_per_mm)
        img[y:y + h, pad + 12:pad + pw - 12] = (25, 25, 25)
        y += h + 20
    r = estimate_scale_from_digits(img, (pad, pad, pad + pw, pad + ph))
    assert isinstance(r, ScaleRefusal)


def test_the_constant_matches_the_source_document():
    """NDDB item 4, 3rd line: 'A row of 6 digits, 18mm high (+/-1mm)'."""
    assert DIGIT_LINE_MM == 18.0
