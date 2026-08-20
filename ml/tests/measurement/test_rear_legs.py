"""Regression tests for the Rear Legs Set / Rear Legs Rear View swap
(Rev 2 audit item 1 / implementation item 10).

Rear Legs Set is the SIDE-VIEW hock angle (knee_left -> hock_left ->
pastern_left, same geometry as the internal "hock_angle" trait, calibrated
130-160 deg cattle / 132-162 deg buffalo). Rear Legs Rear View is the
REAR-VIEW hip->hock cow-hock deviation-from-vertical angle (previously
mis-attached to Rear Legs Set) - calibrated -10 to 10 deg for both species.

Both scoring calibration bins in ml/config/rules.py were migrated to follow
the geometry (not independently re-derived), since inventing new calibration
ranges was out of scope.
"""

import pytest

from ml.measurement.traits import measure_trait
from ml.scoring.scorer import score_trait


def _kp(x: float, y: float, confidence: float = 1.0):
    return (x, y, confidence)


SIDE_VIEW_KP = {
    "knee_left": _kp(100.0, 400.0),
    "hock_left": _kp(95.0, 480.0),
    "pastern_left": _kp(130.0, 550.0),
}

REAR_VIEW_KP = {
    "hip_bone_left": _kp(200.0, 100.0),
    "hip_bone_right": _kp(260.0, 100.0),
    "hock_left": _kp(210.0, 300.0),
    "hock_right": _kp(250.0, 300.0),
}


@pytest.mark.parametrize("species", ["cattle", "buffalo"])
def test_rear_legs_set_uses_side_view_hock_angle(species):
    result = measure_trait("rear_legs_set", SIDE_VIEW_KP, scale_factor=None)
    assert result.trait_class == "A"
    assert result.value is not None
    # Calibrated range is ~130-162 deg for a real hock angle - the old
    # deviation-from-vertical geometry produced values near 0, which would
    # never land in this range.
    assert 100.0 < result.value < 200.0

    score = score_trait(result, species)
    assert score.score_1_9 is not None, (
        "measured value must fall within the migrated calibration bin, "
        "not the old -10..10 deviation-angle bin"
    )


@pytest.mark.parametrize("species", ["cattle", "buffalo"])
def test_rear_legs_rear_view_uses_cow_hock_deviation(species):
    result = measure_trait("rear_legs_rear_view", REAR_VIEW_KP, scale_factor=None)
    assert result.trait_class == "A"
    assert result.value is not None
    # Deviation-from-vertical angles are small - the old ratio/0-1 bin or a
    # 130-160 hock-angle bin would both be wrong ranges for this.
    assert -90.0 < result.value < 90.0

    score = score_trait(result, species)
    assert score.score_1_9 is not None, (
        "measured value must fall within the migrated -10..10 calibration bin"
    )


def test_rear_legs_set_and_rear_view_are_independent():
    """Feeding rear-view-shaped keypoints to rear_legs_set (which now expects
    3 side-view points) must not silently succeed using leftover keypoints -
    it only has knee_left/hock_left/pastern_left available in SIDE_VIEW_KP,
    confirming the two traits no longer share required_keypoints."""
    from ml.config.traits import CONTRACT_TRAITS

    by_id = {t["trait_id"]: t for t in CONTRACT_TRAITS}
    rls = set(by_id["rear_legs_set"]["required_keypoints"])
    rlrv = set(by_id["rear_legs_rear_view"]["required_keypoints"])
    assert rls == {"knee_left", "hock_left", "pastern_left"}
    assert rlrv == {"hip_bone_left", "hip_bone_right", "hock_left", "hock_right"}
    assert rls != rlrv
