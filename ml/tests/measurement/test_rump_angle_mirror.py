"""Regression tests for rump_angle mirror-invariance (REVIEW-ml-dev.md Important Fix #1).

_compute_angle()'s 2-point branch (ml/measurement/traits.py) used to return a
raw atan2(dy, dx). Horizontally mirroring the same physical geometry flips dx
but not dy, sending the raw angle to (180-theta) instead of the equivalent
-theta, which fell outside the calibrated [0, 15] band and clamped to the
opposite extreme score (5.71deg/score 6 -> 174.29deg/score 1). The fix uses
atan2(dy, abs(dx)) so the result depends only on dy (unaffected by a
horizontal mirror), not on which way dx happens to point.

Scope: only rump_angle is exercised, via measure_trait()/score_trait(). The
rump_angle scoring bins (ml/config/rules.py) are NOT modified by this change
and are not modified by these tests.
"""

import math

import pytest

from ml.measurement.traits import measure_trait
from ml.scoring.scorer import score_trait


def _kp(x: float, y: float, confidence: float = 1.0):
    return (x, y, confidence)


def _mirror(keypoints: dict) -> dict:
    """Flip every keypoint horizontally (x -> -x), leaving y and confidence untouched."""
    return {name: (-x, y, c) for name, (x, y, c) in keypoints.items()}


def _measure_and_score(keypoints: dict, species: str = "cattle"):
    measurement = measure_trait("rump_angle", keypoints)
    score = score_trait(measurement, species)
    return measurement.value, score.score_1_9


def test_normal_orientation_matches_known_value_and_score():
    """Canonical orientation: hooks level, pins lower-right by a small amount."""
    keypoints = {
        "hook_left": _kp(0, 0),
        "hook_right": _kp(0, 0),
        "pin_left": _kp(10, 1),
        "pin_right": _kp(10, 1),
    }
    value, score = _measure_and_score(keypoints)
    assert value == pytest.approx(5.710593137499642, abs=1e-6)
    assert score == 4


def test_mirrored_orientation_produces_same_value_and_score():
    """The bug this fix targets: mirroring used to flip the score (verified pre-Phase-4:
    5.71deg/score 6 vs 174.29deg/score 1). Current convention (post-Phase-4, reverse=True
    removed): both orientations must produce 5.71deg/score 4.
    """
    normal = {
        "hook_left": _kp(0, 0),
        "hook_right": _kp(0, 0),
        "pin_left": _kp(10, 1),
        "pin_right": _kp(10, 1),
    }
    mirrored = _mirror(normal)

    normal_value, normal_score = _measure_and_score(normal)
    mirrored_value, mirrored_score = _measure_and_score(mirrored)

    assert mirrored_value == pytest.approx(normal_value, abs=1e-9)
    assert mirrored_score == normal_score
    # Pin down the previously-buggy behavior explicitly so a regression is obvious.
    assert mirrored_value == pytest.approx(5.710593137499642, abs=1e-6)
    assert mirrored_score == 4


def test_flat_rump_dy_near_zero_is_mirror_invariant():
    """dy ~= 0 (hooks and pins level) should read ~0deg regardless of facing direction."""
    normal = {
        "hook_left": _kp(0, 0),
        "hook_right": _kp(0, 0),
        "pin_left": _kp(10, 0.001),
        "pin_right": _kp(10, 0.001),
    }
    mirrored = _mirror(normal)

    normal_value, normal_score = _measure_and_score(normal)
    mirrored_value, mirrored_score = _measure_and_score(mirrored)

    assert normal_value == pytest.approx(0.0, abs=1e-2)
    assert mirrored_value == pytest.approx(normal_value, abs=1e-9)
    assert mirrored_score == normal_score


def test_bin_boundary_case_is_mirror_invariant():
    """A value sitting on a rule-table bin boundary must score identically both ways.

    rump_angle bins span [0, 15] in 9 equal-width bins (width = 15/9 = 1.6667),
    so 5.0deg sits exactly on the boundary between bin index 2 ([3.33, 5.0))
    and bin index 3 ([5.0, 6.67)).

    This only checks that both orientations land in the same bin as each other,
    not which bin that is.
    """
    dx = 10.0
    dy = dx * math.tan(math.radians(5.0))
    normal = {
        "hook_left": _kp(0, 0),
        "hook_right": _kp(0, 0),
        "pin_left": _kp(dx, dy),
        "pin_right": _kp(dx, dy),
    }
    mirrored = _mirror(normal)

    normal_value, normal_score = _measure_and_score(normal)
    mirrored_value, mirrored_score = _measure_and_score(mirrored)

    assert normal_value == pytest.approx(5.0, abs=1e-6)
    assert mirrored_value == pytest.approx(normal_value, abs=1e-9)
    assert mirrored_score == normal_score