"""Regression tests for the Rear Legs Set / Rear Legs Rear View swap
(Rev 2 audit item 1 / implementation item 10).

Rear Legs Set is the SIDE-VIEW hock angle (hip_bone_left -> hock_left ->
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


# A rear leg seen from the side: hip above, hock at the bend, pastern below
# and forward. The upper landmark is hip_bone rather than knee because the
# knee (carpus) is a FORE-leg joint - see ml/config/traits.py for the
# measurements that established this.
SIDE_VIEW_KP = {
    "hip_bone_left": _kp(100.0, 300.0),
    "hock_left": _kp(95.0, 480.0),
    "pastern_left": _kp(130.0, 550.0),
}

# REAR-frame landmarks. This trait compares the animal's left side with its
# right, and a side photograph shows the two sides 1.7% of the animal apart -
# on top of each other - so it can only be measured from a rear view.
REAR_VIEW_KP = {
    "rear_hip_bone_left": _kp(200.0, 100.0),
    "rear_hip_bone_right": _kp(260.0, 100.0),
    "rear_hock_left": _kp(210.0, 300.0),
    "rear_hock_right": _kp(250.0, 300.0),
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
    """The two traits must not collapse back into measuring the same thing.

    They were once the same geometry under two names. What keeps them distinct
    is that one is a side-view angle along a single leg and the other a
    rear-view comparison between both legs, so rear_legs_set must be a
    three-point chain and rear_legs_rear_view a four-point left/right set.
    """
    from ml.config.traits import CONTRACT_TRAITS

    by_id = {t["trait_id"]: t for t in CONTRACT_TRAITS}
    rls = set(by_id["rear_legs_set"]["required_keypoints"])
    rlrv = set(by_id["rear_legs_rear_view"]["required_keypoints"])
    assert rls != rlrv
    assert len(rls) == 3 and len(rlrv) == 4
    assert rlrv == {"rear_hip_bone_left", "rear_hip_bone_right",
                    "rear_hock_left", "rear_hock_right"}
    assert all(j.startswith("rear_") for j in rlrv), (
        "a left-versus-right comparison cannot come from a side photograph")
    # A single leg's chain: one side only, and no left/right pairing.
    assert not any(j.endswith("_right") for j in rls)


def test_rear_legs_set_measures_one_leg_not_two_different_ones():
    """The hock is a REAR-leg joint; the knee (carpus) is a FORE-leg one.

    knee -> hock -> pastern spanned both legs and measured nothing anatomical.
    Over 74 photographs it produced a median of 67.1 degrees against a band of
    130-160 and could only be computed on 11 of them, because knee_left is one
    of the joints that most often collapses onto a neighbour. The rear-leg
    chain gave a median of 153.5 on 40 of them.
    """
    from ml.config.traits import CONTRACT_TRAITS, TRAIT_REGISTRY

    by_id = {t["trait_id"]: t for t in TRAIT_REGISTRY}
    for tid in ("rear_legs_set", "hock_angle"):
        joints = by_id[tid]["required_keypoints"]
        assert "knee_left" not in joints and "knee_right" not in joints, (
            f"{tid} must not mix the fore-leg knee into a rear-leg angle")
        assert joints[1] == "hock_left", f"{tid}'s vertex must be the hock"
