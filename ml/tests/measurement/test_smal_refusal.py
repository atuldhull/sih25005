"""Regression tests for SMAL-class trait refusal (audit Finding 6 / Rev 2 item 5).

Heart Girth and Body Condition Score are registered with trait_class "SMAL"
(require a 3D mesh fit) but measure_trait()'s branch dispatch used to have no
SMAL case, so they fell into the Class C (distance) branch: crashing with
TypeError when unscaled, or returning a wrong-shape value (a 2D chord as a
circumference, a length as a 1-9 index) when scaled. Both must now refuse
honestly with flags=["not_measurable", "requires_3d_model"] regardless of
scale.
"""

import pytest

from ml.measurement.traits import measure_trait


def _kp(x: float, y: float, confidence: float = 1.0):
    return (x, y, confidence)


HEART_GIRTH_KP = {
    "withers": _kp(100.0, 50.0),
    "chest_bottom": _kp(105.0, 120.0),
}

BCS_KP = {
    "pin_left": _kp(10.0, 10.0),
    "pin_right": _kp(20.0, 10.0),
    "tail_head": _kp(15.0, 5.0),
}


@pytest.mark.parametrize("scale_factor", [None, 2.5])
def test_heart_girth_refuses_regardless_of_scale(scale_factor):
    result = measure_trait("heart_girth", HEART_GIRTH_KP, scale_factor=scale_factor)
    assert result.trait_class == "SMAL"
    assert result.value is None
    assert result.confidence == 0.0
    assert "not_measurable" in result.flags
    assert "requires_3d_model" in result.flags


@pytest.mark.parametrize("scale_factor", [None, 2.5])
def test_body_condition_score_refuses_regardless_of_scale(scale_factor):
    result = measure_trait("body_condition_score", BCS_KP, scale_factor=scale_factor)
    assert result.trait_class == "SMAL"
    assert result.value is None
    assert result.confidence == 0.0
    assert "not_measurable" in result.flags
    assert "requires_3d_model" in result.flags


def test_heart_girth_does_not_raise_without_scale():
    """The original bug: distance * None -> TypeError, uncaught anywhere in
    the call chain. Must not raise at all now."""
    measure_trait("heart_girth", HEART_GIRTH_KP, scale_factor=None)
