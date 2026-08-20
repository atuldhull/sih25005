"""Regression tests for the pipeline adoption safety gate
(implementation item 1) and the honest-null weight_kg it was required to
precede (item 3).

server/scoring_loader.py adopts a real ml.pipeline result over the working
baseline engine iff validate(result, mode="pipeline") returns no problems.
Before this fix, a fully-unscored NOT_SCORED result (all 20 traits score:
None) happened to fail validation only because weight_kg.low/high were None
and the validator required them to be numbers - an accidental protection.
Making weight_kg honestly nullable (item 3) would have removed that
accidental protection, so an explicit "at least one trait scored" check was
added first. This test locks in both halves together.

contract/ has no __init__.py (not a package) - it's imported the same way
server/scoring_loader.py does it, via a sys.path append.
"""

import copy
import json
import sys
from pathlib import Path

import pytest

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
            t["not_scored_reason"] = "pose_estimation_not_implemented"
    return result


def test_all_null_result_is_rejected_by_adoption_gate():
    result = _base_result(any_scored=False)
    result["weight_kg"] = {"low": None, "high": None, "method": None, "cross_check": None}
    problems = validate(result, mode="pipeline")
    assert any("adoption gate" in p for p in problems)


def test_partially_scored_result_with_honest_null_weight_passes():
    result = _base_result(any_scored=True)  # example result already has real scores
    result["weight_kg"] = {"low": None, "high": None, "method": None, "cross_check": None}
    problems = validate(result, mode="pipeline")
    assert problems == []


def test_mixed_null_weight_is_still_rejected():
    """One of low/high null and the other a number is malformed, not honest
    refusal - must not be confused with the valid None/None state."""
    result = _base_result(any_scored=True)
    result["weight_kg"] = {"low": 100, "high": None, "method": "x", "cross_check": None}
    problems = validate(result, mode="pipeline")
    assert any("weight_kg" in p for p in problems)


def test_implausible_weight_still_rejected():
    result = _base_result(any_scored=True)
    result["weight_kg"] = {"low": 5, "high": 10, "method": "x", "cross_check": None}
    problems = validate(result, mode="pipeline")
    assert any("implausible" in p for p in problems)


def test_full_mode_does_not_apply_adoption_gate():
    """The adoption gate is pipeline-mode only - full mode (e.g. validating
    the frozen contract example itself) must not require any trait scored."""
    result = _base_result(any_scored=False)
    result["weight_kg"] = {"low": None, "high": None, "method": None, "cross_check": None}
    problems = validate(result, mode="full")
    assert not any("adoption gate" in p for p in problems)
