"""A pipeline result that measured nothing is VALID, and must survive.

This file used to lock in the opposite rule. `validate(..., mode="pipeline")`
carried an "adoption gate" that rejected any result in which all twenty traits
scored null, so that a half-built pipeline could not silently displace the
baseline engine. The intent was sound; the effect was not.

Because scoring_loader treats a rejected result as a reason to call the
BASELINE engine, the gate did not suppress an unscored result - it replaced it
with an invented one. Measured on this build before the change, each of these
came back HTTP 200 with twenty confident scores and a weight near 350 kg:

    a drawing of a chair        (the pipeline itself: 0/20, no_animal_detected)
    pure random RGB noise       (byte-identical output)
    a 44-byte ASCII text file   (byte-identical output)

and so did 12 of 16 real photographs of Indian cattle and buffalo taken
without an ear-tag close-up. One of those fabrications wrote a Lumpy Skin
Disease row into a veterinary officer's alert feed.

So the gate is gone, and these tests now pin the rule that replaced it: an
all-null result validates, and the caller is expected to return it. What the
validator still catches is genuine malformation - a half-null weight, an
impossible weight, a missing trait - which is what a shape validator is for.

contract/ has no __init__.py (not a package) - it is imported the same way
server/scoring_loader.py does it, via a sys.path append.
"""

import copy
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "contract") not in sys.path:
    sys.path.append(str(_REPO / "contract"))
from validate_result import validate  # noqa: E402

_EXAMPLE = json.loads((_REPO / "contract" / "scoring_result.json").read_text())


def _base_result(any_scored: bool = True) -> dict:
    result = copy.deepcopy(_EXAMPLE)
    if not any_scored:
        for t in result["traits"]:
            t["score"] = None
            t["not_scored_reason"] = "no_animal_detected"
    return result


def _refused() -> dict:
    """What the pipeline returns when it looked and could measure nothing."""
    result = _base_result(any_scored=False)
    result["weight_kg"] = {"low": None, "high": None,
                           "method": None, "cross_check": None}
    return result


# --- the rule that replaced the gate -------------------------------------

def test_a_result_that_measured_nothing_is_valid():
    """The whole point. An honest refusal must pass the validator, because a
    caller that rejects it will substitute a fabrication instead."""
    problems = validate(_refused(), mode="pipeline")
    assert problems == [], (
        "a fully unscored result was rejected - the caller will fall back to "
        f"the inventing baseline engine. Problems: {problems}")


def test_no_adoption_gate_remains_in_any_mode():
    for mode in ("pipeline", "full"):
        problems = validate(_refused(), mode=mode)
        assert not any("adoption gate" in p for p in problems), (
            f"the adoption gate is back in mode={mode}; see this file's "
            "docstring for what it does to a photograph of a chair")


def test_the_refusal_still_has_to_say_why():
    """A refusal with no reason is a blank, and a blank is not an answer. The
    validator does not enforce this - it is a property of the pipeline - so it
    is asserted here against the shape the pipeline actually produces."""
    result = _refused()
    assert all(t["not_scored_reason"] for t in result["traits"])


# --- what the validator must still catch ---------------------------------

def test_partially_scored_result_with_honest_null_weight_passes():
    result = _base_result(any_scored=True)  # example already has real scores
    result["weight_kg"] = {"low": None, "high": None,
                           "method": None, "cross_check": None}
    assert validate(result, mode="pipeline") == []


def test_mixed_null_weight_is_still_rejected():
    """One of low/high null and the other a number is malformed, not honest
    refusal - it must not be confused with the valid None/None state."""
    result = _base_result(any_scored=True)
    result["weight_kg"] = {"low": 100, "high": None,
                           "method": "x", "cross_check": None}
    assert any("weight_kg" in p for p in validate(result, mode="pipeline"))


def test_mixed_null_weight_is_rejected_on_a_refused_result_too():
    """Removing the gate must not have opened a hole for the all-null case."""
    result = _refused()
    result["weight_kg"] = {"low": None, "high": 400,
                           "method": "x", "cross_check": None}
    assert any("weight_kg" in p for p in validate(result, mode="pipeline"))


def test_implausible_weight_still_rejected():
    result = _base_result(any_scored=True)
    result["weight_kg"] = {"low": 5, "high": 10,
                           "method": "x", "cross_check": None}
    assert any("implausible" in p for p in validate(result, mode="pipeline"))


def test_a_missing_trait_is_still_rejected_on_a_refused_result():
    result = _refused()
    dropped = result["traits"].pop()["name"]
    problems = validate(result, mode="pipeline")
    assert any(dropped in p for p in problems), (
        "all twenty NDDB traits must be present even when none of them scored")
