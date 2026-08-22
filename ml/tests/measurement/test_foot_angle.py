"""Regression test for Foot Angle geometry (Rev 2 audit item 4 / implementation item 9).

Foot Angle must measure the pastern->hoof vector's angle against horizontal,
not the interior 3-point pastern joint angle. required_keypoints was reduced
from 3 (hock_left, pastern_left, hoof_left) to 2 (pastern_left, hoof_left) so
measure_trait() routes through _compute_angle()'s 2-point horizontal-angle
branch instead of its 3-point interior-vertex-angle branch.
"""

from ml.config.traits import CONTRACT_TRAITS
from ml.measurement.traits import measure_trait


def _kp(x: float, y: float, confidence: float = 1.0):
    return (x, y, confidence)


def test_foot_angle_requires_exactly_two_keypoints():
    by_id = {t["trait_id"]: t for t in CONTRACT_TRAITS}
    assert by_id["foot_angle"]["required_keypoints"] == ["pastern_left", "hoof_left"]


def test_foot_angle_measures_vs_horizontal_not_interior_joint():
    # Near-vertical pastern->hoof line: horizontal-angle branch should return
    # something close to 90 degrees (vertical), not an arbitrary interior
    # joint angle that would depend on a third (hock) point we no longer pass.
    kp = {
        "pastern_left": _kp(100.0, 500.0),
        "hoof_left": _kp(105.0, 550.0),
    }
    result = measure_trait("foot_angle", kp)
    assert result.trait_class == "A"
    assert result.value is not None
    assert 75.0 < result.value < 90.0
