"""The scale path must not depend on the ear-tag detector.

Everything measured in centimetres - five contract traits, heart girth, and
the weight estimate - hangs off one call to detect_ear_tag. On a 2560x1700
photograph with a plainly legible yellow tag that call returns None, and still
returns None with the threshold dropped to 0.02: the class does not fire at
all. See FINDINGS-ear-tag-in-the-field.md.

On the app's dedicated close-up the detector is not needed. The photograph is
OF the tag, so the frame is the region of interest, and the ruler is the real
validator - it gates on the panel's height against the published 55-69 mm, on
the round button, and on the 10/10/18 mm printed rows. A wrong box produces a
refusal, not a wrong scale.

The second half of this is the digit-row method, which was written, validated
against the published NDDB dimensions, covered by tests - and never called,
because measure_scale only ever tried the button. Per the spec the button is
on the REAR of the ear while the barcode and digits are printed on the FRONT,
so a farmer photographing a tag head-on gets a picture with no button in it.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_tag_scale_recovery import render_tag  # noqa: E402

from ml.tag_intelligence.tag_reader import (  # noqa: E402
    TagScaleRefused,
    measure_scale,
    read_tag,
    scale_factor_from,
)


@pytest.mark.parametrize("px_per_mm", [4.0, 6.0, 9.0])
def test_the_whole_frame_works_as_well_as_a_perfect_box(px_per_mm):
    """A close-up needs no detector: the frame IS the tag."""
    img, bbox = render_tag(px_per_mm)
    h, w = img.shape[:2]
    from_box = measure_scale(img, bbox)["cm_per_px"]
    from_frame = measure_scale(img, (0.0, 0.0, float(w), float(h)))["cm_per_px"]
    true = 1.0 / (10.0 * px_per_mm)
    assert from_frame == pytest.approx(true, rel=0.12)
    assert from_frame == pytest.approx(from_box, rel=0.02), (
        "the detector's box bought nothing on a close-up")


def test_a_tag_with_no_visible_button_still_yields_a_scale():
    """The case the digit method exists for.

    render_tag draws the three printed rows and no button, which is what a
    head-on photograph of a real tag looks like - the button is moulded on the
    back of the ear. Before the fallback was wired this refused outright.
    """
    img, bbox = render_tag(6.0)
    out = measure_scale(img, bbox)
    assert out["cm_per_px"] == pytest.approx(1.0 / 60.0, rel=0.12)
    assert "digit" in out["method"], (
        f"expected the printed-row method, got {out['method']}")


def test_the_digit_method_reports_its_own_larger_error():
    """It is a less direct ruler than a 27 mm button and must say so, because
    every class-C trait inherits this error."""
    img, bbox = render_tag(6.0)
    out = measure_scale(img, bbox)
    from ml.tag_intelligence.tag_ruler import DIGIT_SCALE_MIN_ERROR_FRAC
    assert out["error_frac"] == pytest.approx(DIGIT_SCALE_MIN_ERROR_FRAC)
    assert out["error_frac"] > 0.04, "the button method's floor"


def test_a_photograph_with_no_tag_in_it_is_refused():
    """Handing the ruler a whole frame must not make it credulous."""
    noise = np.random.default_rng(0).integers(0, 255, (500, 700, 3),
                                              dtype=np.uint8)
    with pytest.raises(TagScaleRefused):
        measure_scale(noise, (0.0, 0.0, 700.0, 500.0))


def test_a_yellow_rectangle_that_is_not_a_tag_is_refused():
    """Lemon yellow alone is not a tag. Without the printed rows at 10/10/18
    mm there is nothing of known size to measure, and a plain panel would give
    a scale wrong by however much this supplier's panel differs."""
    img = np.full((500, 700, 3), 70, np.uint8)
    img[120:380, 200:520] = (60, 220, 240)          # blank lemon panel
    with pytest.raises(TagScaleRefused):
        measure_scale(img, (0.0, 0.0, 700.0, 500.0))


def test_the_refusal_reason_tells_the_farmer_what_to_do():
    noise = np.random.default_rng(1).integers(0, 255, (400, 400, 3),
                                              dtype=np.uint8)
    r = read_tag(noise, (0.0, 0.0, 400.0, 400.0))
    assert scale_factor_from(r) == (None, 0.0)
    reason = str(r.get("refused_reason", "")).lower()
    assert any(w in reason for w in ("retake", "closer", "facing")), (
        f"a refusal has to be actionable, got: {reason}")
