"""Chooses the scoring engine at call time.

Integration-day design: when Person 2's real pipeline (ml/pipeline.py)
becomes importable AND its output passes the contract validator, it is
used automatically - the import is retried every RETRY_SECONDS, so
landing ml/ needs NO server restart. Anything less - import error,
crash, or contract violations - falls back to the fake engine, with
the reason recorded in the result's 'engine' field. Nobody flips a
switch; the validator IS the switch.
"""
import importlib
import os
import sys
import threading
import time
from pathlib import Path

from scoring import score_animal as _fake_score

_REPO = Path(__file__).parent.parent
# append (not insert) so site-packages always outranks repo folders
if str(_REPO / "contract") not in sys.path:
    sys.path.append(str(_REPO / "contract"))
from validate_result import validate  # noqa: E402

RETRY_SECONDS = 30.0

_lock = threading.Lock()
_real_score = None
_real_import_error = "not tried yet"
_next_retry = 0.0


def _try_import():
    global _real_score, _real_import_error, _next_retry
    with _lock:
        if _real_score is not None or time.monotonic() < _next_retry:
            return
        _next_retry = time.monotonic() + RETRY_SECONDS
        try:
            if str(_REPO) not in sys.path:
                sys.path.append(str(_REPO))
            importlib.invalidate_caches()
            mod = importlib.import_module("ml.pipeline")
            _real_score = mod.score_animal
            _real_import_error = None
        except Exception as e:
            _real_import_error = str(e)


def engine_status() -> dict:
    _try_import()
    status = {"real_pipeline_importable": _real_score is not None}
    # full error text only for the team (SIH_DEBUG=1) - don't broadcast
    # internals to everyone on the venue network
    if os.environ.get("SIH_DEBUG"):
        status["import_error"] = _real_import_error
        status["retry_every_seconds"] = RETRY_SECONDS
    return status


def score_animal(side_img, rear_img, video_path, animal_record) -> dict:
    _try_import()
    if _real_score is not None:
        try:
            result = _real_score(side_img, rear_img, video_path, animal_record)
            problems = validate(result, mode="pipeline")
            if not problems:
                result["engine"] = "ml-pipeline"
                return result
            reason = f"contract violations: {len(problems)} (e.g. {problems[0]})"
        except Exception as e:
            reason = f"pipeline crashed: {e}"
    else:
        reason = f"not importable: {_real_import_error}"

    result = _fake_score(side_img, rear_img, video_path, animal_record)
    result["engine"] = f"fake (real pipeline unavailable - {reason})"
    return result
