"""An angle is only as good as the segment it is measured across.

Keypoint error is a fixed fraction of the ANIMAL's size - roughly 1.56% of its
longer side - and it does not shrink just because two joints happen to be close
together. So an angle taken across a short segment inherits an enormous error
while looking exactly as confident as one taken across a long one.

Measured over 55 photographs:

    rump_angle      segment 33.1% of the animal  ->  +/-4 deg,  band 15 deg
    shoulder_angle  segment  4.8%                ->  +/-27 deg, band 20 deg
    foot_angle      segment  3.9%                ->  +/-25 deg, band 25 deg

The last two were being scored on a quantity whose error bar covers the whole
range the 1-9 score is drawn from. Every bin sits inside the uncertainty, so
which bin comes out is decided by noise - and the farmer is shown a number.
"""
import math

import pytest

from ml.common.schemas import MeasurementResult
from ml.measurement.traits import (
    KEYPOINT_ERR_FRAC,
    _angle_uncertainty,
    _animal_scale,
    measure_trait,
)
from ml.scoring.scorer import UNCERTAINTY_BAND_FRACTION, score_trait


def test_a_short_segment_is_far_less_certain_than_a_long_one():
    scale = 1000.0
    long_seg = _angle_uncertainty([(0.0, 0.0), (400.0, 0.0)], scale, "angle")
    short_seg = _angle_uncertainty([(0.0, 0.0), (40.0, 0.0)], scale, "angle")
    assert short_seg > long_seg * 5, (
        f"a segment a tenth as long must be far less certain: "
        f"{short_seg:.1f} vs {long_seg:.1f} degrees")


def test_the_uncertainty_matches_the_geometry_it_claims():
    """atan(error / segment), with the error from two independent endpoints."""
    scale, seg = 1000.0, 100.0
    got = _angle_uncertainty([(0.0, 0.0), (seg, 0.0)], scale, "angle")
    expected = math.degrees(math.atan2(
        KEYPOINT_ERR_FRAC * scale * math.sqrt(2.0), seg))
    assert got == pytest.approx(expected, rel=1e-9)


def test_the_leg_set_geometry_measures_the_LEG_not_the_gap_between_legs():
    """The four-point traits compare a left leg with a right leg.

    Each leg's direction comes from its own upper-to-lower span. On a side-on
    photograph the two sides overlap almost exactly, so the gap BETWEEN them is
    a few pixels - and treating that as the baseline implies a huge error where
    there is none. Doing so suppressed fore_leg_set from 19 scored to 9.
    """
    scale = 1000.0
    pts = [(100.0, 100.0), (104.0, 102.0),      # upper left, upper right
           (100.0, 500.0), (105.0, 503.0)]      # lower left, lower right
    leg = _angle_uncertainty(pts, scale, "leg_set")
    naive = _angle_uncertainty(pts, scale, "angle")
    assert leg is not None and naive is not None
    assert leg < 10.0, f"the legs are 400 px long; {leg:.0f} deg is wrong"
    assert naive > leg * 4, "the naive reduction should be much worse here"


def test_the_animal_scale_comes_from_the_joints_themselves():
    kps = {"a": (100.0, 100.0, 0.9), "b": (900.0, 300.0, 0.9),
           "c": (500.0, 200.0, 0.0)}          # refused, must not count
    assert _animal_scale(kps) == pytest.approx(800.0)


def test_one_joint_gives_no_scale_rather_than_zero():
    assert _animal_scale({"a": (1.0, 2.0, 0.9)}) == 0.0
    assert _angle_uncertainty([(0.0, 0.0), (10.0, 0.0)], 0.0, "angle") is None


# --- the scoring gate -----------------------------------------------------

def _measurement(trait_id, value, uncertainty):
    return MeasurementResult(trait_id=trait_id, trait_class="A", value=value,
                             unit="degrees", confidence=0.8, flags=[],
                             uncertainty=uncertainty)


def test_an_error_bar_that_covers_the_band_refuses_to_score():
    """foot_angle's band is 40-65: 25 degrees wide. An uncertainty of 25
    degrees means every one of the nine bins sits inside the error bar."""
    m = _measurement("foot_angle", 52.0, 25.0)
    assert score_trait(m, "cattle").score_1_9 is None


def test_a_tight_measurement_in_the_same_band_still_scores():
    m = _measurement("foot_angle", 52.0, 2.0)
    assert score_trait(m, "cattle").score_1_9 is not None


def test_the_threshold_is_a_fraction_of_the_band_not_an_absolute():
    """A wide band tolerates more absolute error than a narrow one, which is
    the whole point of comparing against the band rather than a fixed number."""
    band = 25.0                                    # foot_angle, 40..65
    just_under = band * UNCERTAINTY_BAND_FRACTION * 0.95
    just_over = band * UNCERTAINTY_BAND_FRACTION * 1.05
    assert score_trait(_measurement("foot_angle", 52.0, just_under),
                       "cattle").score_1_9 is not None
    assert score_trait(_measurement("foot_angle", 52.0, just_over),
                       "cattle").score_1_9 is None


def test_a_measurement_without_an_uncertainty_is_unaffected():
    """Class B and C traits do not compute one, and must not be refused."""
    m = MeasurementResult(trait_id="foot_angle", trait_class="A", value=52.0,
                          unit="degrees", confidence=0.8, flags=[])
    assert m.uncertainty is None
    assert score_trait(m, "cattle").score_1_9 is not None


def test_real_traits_carry_an_uncertainty_through_measurement():
    """End to end: measure_trait must attach it, or the gate never fires."""
    kps = {"pastern_left": (500.0, 900.0, 0.8), "hoof_left": (520.0, 950.0, 0.8),
           "withers": (200.0, 200.0, 0.8), "tail_head": (1400.0, 250.0, 0.8)}
    m = measure_trait("foot_angle", kps)
    assert m.value is not None
    assert m.uncertainty is not None and m.uncertainty > 0
    # pastern to hoof is ~54 px on a ~1200 px animal: a very short lever
    assert m.uncertainty > 15.0, (
        f"a 4% segment should be highly uncertain, got {m.uncertainty:.1f}")
    assert score_trait(m, "cattle").score_1_9 is None, (
        "and that uncertainty must actually stop the score")


# --- the reason has to distinguish the two failures -----------------------
# "Outside the calibrated range" reads as "this animal is unusual". When the
# measurement is tight and still far outside anatomy, the opposite is true:
# the landmarks are in the wrong place. Saying the wrong one sends whoever
# reads it looking at the animal instead of at the photograph.

def test_an_imprecise_refusal_says_it_was_imprecise():
    m = _measurement("foot_angle", 52.0, 25.0)
    r = score_trait(m, "cattle")
    assert r.score_1_9 is None
    assert "imprecise" in r.not_scored_reason
    assert "25" in r.not_scored_reason, "it should quote the actual figure"


def test_a_tight_value_far_outside_its_band_blames_the_landmarks():
    """Precise and impossible means bad inputs, not a strange animal."""
    m = _measurement("foot_angle", 140.0, 1.0)          # band is 40..65
    r = score_trait(m, "cattle")
    assert r.score_1_9 is None
    assert "landmarks" in r.not_scored_reason
    assert "more likely wrong than the animal unusual" in r.not_scored_reason


def test_a_loose_value_outside_its_band_makes_no_such_claim():
    """With a wide error bar there is no basis for blaming either one."""
    m = _measurement("foot_angle", 140.0, 12.0)
    r = score_trait(m, "cattle")
    assert r.score_1_9 is None
    assert "landmarks" not in r.not_scored_reason


def test_a_scored_trait_carries_no_reason():
    r = score_trait(_measurement("foot_angle", 52.0, 1.0), "cattle")
    assert r.score_1_9 is not None
    assert r.not_scored_reason is None
