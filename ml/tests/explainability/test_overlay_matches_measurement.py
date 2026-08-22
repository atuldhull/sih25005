"""The overlay must show the points the measurement actually used.

The overlay is what a judge or a vet officer taps to see WHY a trait scored
what it did. If it draws a different set of points from the ones the
measurement was computed on, it is proof of something else.

The two had drifted: generate_overlay_data hardcoded a 0.3 confidence
threshold while measurement moved to 0.10, so a trait computed from a joint at
0.15 had that joint quietly missing from its own picture.
"""
import pytest

from ml.config.traits import get_trait
from ml.explainability.explainer import generate_overlay_data
from ml.measurement.traits import KEYPOINT_CONFIDENCE_THRESHOLD, measure_trait


def _leg(conf_hock):
    return {"hip_bone_left": (100.0, 300.0, 0.8),
            "hock_left": (95.0, 480.0, conf_hock),
            "pastern_left": (130.0, 550.0, 0.8)}


def test_a_joint_good_enough_to_measure_is_good_enough_to_draw():
    """A joint just above the measurement threshold. If the overlay used a
    stricter one, this trait would measure and then show an incomplete
    picture of how."""
    conf = KEYPOINT_CONFIDENCE_THRESHOLD + 0.02
    kps = _leg(conf)
    assert measure_trait("rear_legs_set", kps).value is not None
    overlay = generate_overlay_data("rear_legs_set", kps)
    assert len(overlay["points"]) == 3, (
        f"measured from three joints but drew {len(overlay['points'])}")


def test_a_joint_too_weak_to_measure_is_not_drawn_either():
    """The other direction: the overlay must not imply a joint was used when
    the measurement refused it."""
    kps = _leg(KEYPOINT_CONFIDENCE_THRESHOLD - 0.02)
    assert measure_trait("rear_legs_set", kps).value is None
    overlay = generate_overlay_data("rear_legs_set", kps)
    assert len(overlay["points"]) == 2


def test_the_two_thresholds_are_the_same_object():
    """Not merely equal today. Two constants that happen to match will drift,
    which is exactly what happened."""
    import ml.explainability.explainer as explainer
    import ml.measurement.traits as traits
    assert (explainer.KEYPOINT_CONFIDENCE_THRESHOLD
            is traits.KEYPOINT_CONFIDENCE_THRESHOLD)


@pytest.mark.parametrize("trait_id", ["rear_legs_set", "fore_leg_set",
                                      "foot_angle", "shoulder_angle"])
def test_the_overlay_draws_the_traits_own_landmarks(trait_id):
    """Not some other trait's. A mislabelled overlay is worse than none."""
    required = get_trait(trait_id)["required_keypoints"]
    kps = {name: (100.0 + 40 * i, 200.0 + 30 * i, 0.9)
           for i, name in enumerate(required)}
    overlay = generate_overlay_data(trait_id, kps)
    assert overlay["trait_id"] == trait_id
    assert len(overlay["points"]) == len(required)
    for name in required:
        assert (kps[name][0], kps[name][1]) in overlay["points"]
