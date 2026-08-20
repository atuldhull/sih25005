"""The reported interval must come from a real uncertainty.

ci was null everywhere, with a comment saying no CI estimate existed and that
null was the honest value. It was - fabricating a range would have been worse.
But measurement now computes an uncertainty from each trait's own geometry: an
angle taken across a segment a few percent of the animal long inherits a much
wider one than the same angle across its whole body. So there is a real
interval to report.

Null still means "we do not know". It must never come to mean "exact".
"""
import pytest

from ml.common.schemas import MeasurementResult, ScoreResult, ScoringResult
from ml.explainability.result_builder import to_contract_dict


def _result(uncertainty, value=142.0, unit="degrees"):
    m = MeasurementResult(trait_id="rear_legs_set", trait_class="A",
                          value=value, unit=unit, confidence=0.7, flags=[],
                          uncertainty=uncertainty)
    r = ScoringResult(
        animal_id="T1", species="cattle", status="SCORED", tag=None,
        traits=[ScoreResult(trait_id="rear_legs_set", score_1_9=5,
                            confidence=0.6)])
    out = to_contract_dict(r, [m], {}, breed_registered="Gir")
    return next(t for t in out["traits"] if t["name"] == "Rear Legs Set")


def test_the_interval_is_the_value_plus_and_minus_its_uncertainty():
    t = _result(4.5)
    assert t["ci"] == "137.5-146.5 degrees"


def test_no_uncertainty_means_no_interval_not_a_zero_width_one():
    """Null is "we do not know". A zero-width interval would claim exactness
    that nothing measured."""
    assert _result(None)["ci"] is None


def test_a_negative_bound_does_not_collide_with_the_separator():
    """Signed traits produce them routinely - a cow-hock deviation, an udder
    floor above the hock. "-3.0-4.8 degrees" is unreadable."""
    t = _result(4.0, value=1.0)
    assert t["ci"] == "-3.0 to 5.0 degrees", t["ci"]
    assert "--" not in t["ci"]


def test_a_wider_uncertainty_gives_a_wider_interval():
    narrow = _result(1.0)["ci"]
    wide = _result(20.0)["ci"]
    assert narrow != wide
    assert "141.0-143.0" in narrow
    assert "122.0-162.0" in wide


def test_the_interval_carries_the_traits_own_unit():
    assert _result(0.05, value=1.10, unit="ratio")["ci"].endswith("ratio")
