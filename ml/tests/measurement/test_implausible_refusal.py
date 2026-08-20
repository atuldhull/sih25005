"""A measurement that cannot describe a real animal must be refused.

Found during the first end-to-end chain run: two chest-width keypoints
collapsed onto nearly the same pixel, producing a 0.5 cm chest width that
sailed through to scoring. config/traits.py defines no ranges, so nothing
caught it. Showing a farmer a 0.5 cm chest is exactly what score=null and
not_scored_reason exist to prevent.
"""
import pytest

from ml.measurement.traits import IMPOSSIBLE_OUTSIDE, _is_impossible


@pytest.mark.parametrize("value", [0.5, 0.0, 0.9, 301.0, 10000.0])
def test_impossible_cm_values_are_flagged(value):
    assert _is_impossible(value, "cm")


@pytest.mark.parametrize("value", [2.0, 5.2, 47.0, 113.0, 182.0, 300.0])
def test_real_cm_measurements_are_kept(value):
    """Anything a bovine could actually be must survive."""
    assert not _is_impossible(value, "cm")


def test_biological_extremes_are_not_suppressed():
    """The ICAR scale runs 1-9 precisely to describe extremes.

    A very small or very large REAL animal must still be measured; only
    physically impossible values are refused. A guard that quietly discarded
    unusual animals would defeat the scorecard it is meant to protect.
    """
    assert not _is_impossible(95.0, "cm")     # a small dwarf breed
    assert not _is_impossible(5.2, "cm")      # a real teat length
    assert not _is_impossible(2.0, "cm")      # a real teat thickness
    assert not _is_impossible(175.0, "cm")    # a very tall Ongole
    assert not _is_impossible(1.0, "degrees")
    assert not _is_impossible(179.0, "degrees")


def test_none_is_not_impossible():
    """None already means 'not measured'; it must not be double-refused."""
    assert not _is_impossible(None, "cm")


def test_unknown_unit_is_left_alone():
    """Only units with a known physical bound are policed."""
    assert not _is_impossible(1e9, "score")


def test_every_policed_unit_has_a_sane_bound():
    for unit, (lo, hi) in IMPOSSIBLE_OUTSIDE.items():
        assert lo < hi, f"{unit} bounds inverted"
