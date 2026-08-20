"""Carrying a scale from the close-up to the side photograph.

The two shots are taken from different distances, so the close-up's
centimetres-per-pixel is meaningless in the side photo. One physical object
appears in both, though - the tag - so it can act as a bridge:

    the close-up gives cm/px from a feature of KNOWN size (the 18 mm digit row)
    -> the tag's own visible height, measured in the close-up, is known in cm
    -> that same height measured in the side photo gives the side photo's scale

What makes this legitimate is that the tag's size is never ASSUMED. The panel
is 55-69 mm depending on the supplier, which is exactly why tag_ruler refuses
to use it as a ruler; here it is measured on the actual tag first.

These check the arithmetic against a synthetic pair where the true answer is
known, and then check that it refuses when it should - which matters more,
because a wrong scale is worse than no scale. Weight goes as its cube.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_tag_scale_recovery import render_tag  # noqa: E402

from ml.tag_intelligence.panel_transfer import (  # noqa: E402
    MIN_PANEL_PX,
    TransferRefusal,
    TransferredScale,
    panel_height_cm,
    transfer,
)

# A cow facing LEFT (facing_sign -1), so the head end is the low-x side.
ANIMAL = (100.0, 80.0, 1700.0, 1150.0)


def side_photo_with_tag(tag_h_px=90, x=260, y=250, extra_yellow=True):
    """A side photograph with a yellow tag near the head, and distractions.

    The distractions are the point. Lemon yellow is common on a farm - the
    photograph this was built from has a yellow truck across the field and
    sunlit ground - and picking one of those would give a confidently wrong
    scale.
    """
    img = np.full((1250, 1800, 3), 90, np.uint8)
    # A red Zebu coat, BGR for H=10 S=200 - the hardest real case, because a
    # red coat is the nearest thing on the animal to a lemon tag. Measured on
    # a Sahiwal the coat sits at H 8-11 and the tag at H 24, so the gate's
    # lower bound of 15 separates them by about 13 hue units.
    img[80:1150, 100:1700] = (32, 72, 150)               # the animal
    w = int(round(tag_h_px / 1.7))                        # tag with its tab
    img[y:y + tag_h_px, x:x + w] = (60, 220, 240)         # lemon yellow
    if extra_yellow:
        img[900:1150, 1300:1650] = (60, 215, 235)         # a truck, far side
        img[1160:1240, 200:900] = (70, 210, 230)          # sunlit ground
    return img


def test_the_transfer_recovers_a_known_scale():
    """The whole chain against a pair whose true answer is known.

    The close-up is rendered at 6 px/mm; the same tag is 90 px tall in the side
    photo. Whatever the tag's real height turns out to be, the side photo's
    scale must be that height divided by 90.
    """
    closeup, tag_bbox = render_tag(6.0)
    cm_per_px_closeup = 1.0 / 60.0

    height_cm = panel_height_cm(closeup, tag_bbox, cm_per_px_closeup)
    assert height_cm is not None, "the tag was not measurable in the close-up"

    side = side_photo_with_tag(tag_h_px=90)
    got = transfer(side, ANIMAL, -1, closeup, tag_bbox, cm_per_px_closeup)
    assert isinstance(got, TransferredScale), getattr(got, "reason", "")
    assert got.panel_px_side == pytest.approx(90, abs=2)
    assert got.cm_per_px == pytest.approx(height_cm / 90.0, rel=0.03)


def test_a_tag_further_away_gives_a_coarser_scale():
    """Monotonic and in the right direction: the same tag appearing smaller in
    the side photo means each pixel covers MORE centimetres."""
    closeup, bbox = render_tag(6.0)
    near = transfer(side_photo_with_tag(120), ANIMAL, -1, closeup, bbox, 1 / 60.0)
    far = transfer(side_photo_with_tag(60), ANIMAL, -1, closeup, bbox, 1 / 60.0)
    assert isinstance(near, TransferredScale) and isinstance(far, TransferredScale)
    assert far.cm_per_px > near.cm_per_px
    assert far.cm_per_px / near.cm_per_px == pytest.approx(2.0, rel=0.08)


def test_the_tag_size_is_measured_not_assumed():
    """The reason this is allowed at all.

    A panel is 55-69 mm depending on the supplier, so the same physical tag
    photographed at two different close-up scales must come back the same size
    in centimetres. If it did not, the supplier variation would be silently
    baked into every measurement.
    """
    small = panel_height_cm(*render_tag(5.0), 1.0 / 50.0)
    large = panel_height_cm(*render_tag(9.0), 1.0 / 90.0)
    assert small is not None and large is not None
    assert small == pytest.approx(large, rel=0.10)


# --- refusals: a wrong scale is worse than no scale ----------------------

def test_yellow_elsewhere_in_the_frame_is_not_mistaken_for_the_tag():
    """A truck and sunlit ground are both lemon yellow and both much larger."""
    closeup, bbox = render_tag(6.0)
    got = transfer(side_photo_with_tag(tag_h_px=90), ANIMAL, -1,
                   closeup, bbox, 1 / 60.0)
    assert isinstance(got, TransferredScale)
    assert got.panel_px_side == pytest.approx(90, abs=2), (
        f"measured {got.panel_px_side} px - a background object was picked")


def test_no_tag_in_the_side_photo_is_refused():
    closeup, bbox = render_tag(6.0)
    bare = np.full((1250, 1800, 3), 90, np.uint8)
    bare[80:1150, 100:1700] = (70, 110, 150)
    got = transfer(bare, ANIMAL, -1, closeup, bbox, 1 / 60.0)
    assert isinstance(got, TransferRefusal)
    assert "side photo" in got.reason


def test_a_tag_too_small_to_measure_is_refused():
    """At MIN_PANEL_PX a two-pixel edge error is already 5%, and that error is
    cubed by the time it reaches a weight."""
    closeup, bbox = render_tag(6.0)
    side = side_photo_with_tag(tag_h_px=MIN_PANEL_PX - 12)
    assert isinstance(transfer(side, ANIMAL, -1, closeup, bbox, 1 / 60.0),
                      TransferRefusal)


def test_an_unknown_facing_direction_is_refused():
    """Without knowing which end is the head the ear cannot be located, and
    the search would have to cover the whole animal - including the udder,
    which is not where ear tags are."""
    closeup, bbox = render_tag(6.0)
    got = transfer(side_photo_with_tag(), ANIMAL, None, closeup, bbox, 1 / 60.0)
    assert isinstance(got, TransferRefusal)
    assert "facing" in got.reason.lower()


def test_a_tag_at_the_wrong_end_of_the_animal_is_not_used():
    """Anything yellow at the TAIL end is not an ear tag."""
    closeup, bbox = render_tag(6.0)
    side = side_photo_with_tag(tag_h_px=90, x=1500, y=250, extra_yellow=False)
    got = transfer(side, ANIMAL, -1, closeup, bbox, 1 / 60.0)
    assert isinstance(got, TransferRefusal)


def test_two_disagreeing_candidates_refuse_rather_than_choose():
    closeup, bbox = render_tag(6.0)
    side = side_photo_with_tag(tag_h_px=90, extra_yellow=False)
    side[400:400 + 150, 500:500 + 88] = (60, 220, 240)     # a second "tag"
    got = transfer(side, ANIMAL, -1, closeup, bbox, 1 / 60.0)
    assert isinstance(got, TransferRefusal)
    assert "disagree" in got.reason.lower()


def test_the_error_carries_parallax_and_the_closeup_error():
    """The tag hangs from the ear, nearer the camera than the flank, so this
    scale slightly overstates the scale at the barrel. Ignoring that would be
    false precision."""
    closeup, bbox = render_tag(6.0)
    got = transfer(side_photo_with_tag(90), ANIMAL, -1, closeup, bbox,
                   1 / 60.0, closeup_error_frac=0.056)
    assert isinstance(got, TransferredScale)
    assert got.error_frac > 0.056, (
        "the transferred scale cannot be more certain than the close-up it "
        "came from")
    assert got.error_frac < 0.30


def test_a_missing_closeup_scale_is_refused():
    closeup, bbox = render_tag(6.0)
    assert panel_height_cm(closeup, bbox, None) is None
    assert panel_height_cm(closeup, bbox, 0.0) is None
