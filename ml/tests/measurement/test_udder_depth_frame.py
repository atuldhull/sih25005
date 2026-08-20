"""Udder depth: one frame, and a sign that means something.

Two bugs met in this trait, and neither raised anything.

FRAME. ICAR measures udder depth from the udder floor to the hock, and a hock
is not an udder landmark. So the trait asked for udder_floor - which is merged
into the keypoints from the REAR photograph - together with hock_left, which
is not merged and stayed in SIDE-photo pixels. The distance between them was
computed across two coordinate frames, from two shots taken at two distances.
It produced 74.7 cm against a calibrated band of -10 to 25, which reads as an
animal with a remarkable udder rather than as a units error.

SIGN. The band runs from -10 to 25 because an udder floor ABOVE the hock is a
well-attached udder and a different animal from one hanging below it. A
Euclidean distance is a magnitude and can never be negative, so the whole
negative half of that band was unreachable and a high udder was reported as a
low one of the same size.
"""
import pytest

from ml.config.traits import get_trait
from ml.measurement.traits import measure_trait
from ml.pose_features.silhouette_landmarks import (REAR_FRAME_ALIASES,
                                                   add_rear_view_landmarks)

CM_PER_PX = 0.2


def _kps(udder_y, hock_y, hock_name="rear_hock_left"):
    return {"udder_floor": (500.0, udder_y, 0.6),
            hock_name: (520.0, hock_y, 0.7)}


# --- the sign ------------------------------------------------------------

def test_an_udder_below_the_hock_is_positive():
    m = measure_trait("udder_depth", _kps(udder_y=400.0, hock_y=300.0),
                      scale_factor=CM_PER_PX)
    assert m.value is not None
    assert m.value == pytest.approx(100.0 * CM_PER_PX)
    assert m.value > 0


def test_an_udder_above_the_hock_is_NEGATIVE():
    """The half of the band a magnitude could never reach."""
    m = measure_trait("udder_depth", _kps(udder_y=260.0, hock_y=300.0),
                      scale_factor=CM_PER_PX)
    assert m.value is not None
    assert m.value < 0, (
        "an udder floor above the hock must read negative - that is what the "
        "-10 end of the calibrated band is for")
    assert m.value == pytest.approx(-40.0 * CM_PER_PX)


def test_the_two_cases_are_distinguishable():
    """The actual failure: same distance, opposite anatomy, one number."""
    below = measure_trait("udder_depth", _kps(400.0, 300.0), scale_factor=CM_PER_PX)
    above = measure_trait("udder_depth", _kps(200.0, 300.0), scale_factor=CM_PER_PX)
    assert below.value != above.value
    assert below.value == pytest.approx(-above.value)


def test_it_is_vertical_only():
    """A horizontal offset between the udder and the hock is the animal's
    width, not its udder depth, and must not inflate the reading."""
    near = measure_trait("udder_depth",
                         {"udder_floor": (500.0, 400.0, 0.6),
                          "rear_hock_left": (505.0, 300.0, 0.7)},
                         scale_factor=CM_PER_PX)
    far = measure_trait("udder_depth",
                        {"udder_floor": (500.0, 400.0, 0.6),
                         "rear_hock_left": (900.0, 300.0, 0.7)},
                        scale_factor=CM_PER_PX)
    assert near.value == pytest.approx(far.value)


# --- the frame -----------------------------------------------------------

def test_the_trait_asks_for_the_REAR_frame_hock():
    joints = get_trait("udder_depth")["required_keypoints"]
    assert "rear_hock_left" in joints
    assert "hock_left" not in joints, (
        "hock_left is the SIDE photograph's hock; pairing it with udder_floor "
        "measures across two coordinate frames")


def test_the_side_view_hock_is_not_overwritten_by_the_rear_one():
    """The leg traits depend on the side-view hock. Merging the rear one over
    it would fix udder depth by breaking rear_legs_set and hock_angle."""
    side = {"hock_left": (100.0, 200.0, 0.62)}
    rear = {"hock_left": (900.0, 800.0, 0.7)}
    out, prov = add_rear_view_landmarks(side, rear, None, None)
    assert out["hock_left"] == (100.0, 200.0, 0.62)
    assert out["rear_hock_left"] == (900.0, 800.0, 0.7)
    assert prov["rear_hock_left"] == "detected_in_rear_view"


def test_every_left_right_pair_a_rear_trait_needs_is_aliased():
    """Hocks for udder depth; hip bones and hooks for the traits that compare
    the two sides of the animal, which a side photograph cannot show."""
    assert REAR_FRAME_ALIASES == {
        "hock_left": "rear_hock_left",
        "hock_right": "rear_hock_right",
        "hip_bone_left": "rear_hip_bone_left",
        "hip_bone_right": "rear_hip_bone_right",
        "hook_left": "rear_hook_left",
        "hook_right": "rear_hook_right",
        "hoof_left": "rear_hoof_left",
        "hoof_right": "rear_hoof_right",
    }


def test_a_refused_rear_hock_is_not_merged():
    out, prov = add_rear_view_landmarks(
        {}, {"hock_left": (900.0, 800.0, 0.0)}, None, None)
    assert "rear_hock_left" not in out
    assert prov == {}


def test_without_a_rear_photo_the_trait_simply_refuses():
    """No rear view means no rear hock, and the trait must say so rather than
    reaching for the side-view one."""
    m = measure_trait("udder_depth",
                      {"udder_floor": (500.0, 400.0, 0.6),
                       "hock_left": (100.0, 300.0, 0.7)},
                      scale_factor=CM_PER_PX)
    assert m.value is None
    assert "not_measurable" in m.flags


def test_no_scale_still_refuses():
    m = measure_trait("udder_depth", _kps(400.0, 300.0), scale_factor=None)
    assert m.value is None
    assert "no_scale" in m.flags
