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
_importing = False


def _do_import():
    """Runs in a daemon thread: a real pipeline importing torch can
    take 30+ seconds, and that must never happen on a request."""
    global _real_score, _real_import_error, _importing
    try:
        if str(_REPO) not in sys.path:
            sys.path.append(str(_REPO))
        importlib.invalidate_caches()
        mod = importlib.import_module("ml.pipeline")
        with _lock:
            _real_score = mod.score_animal
            _real_import_error = None
    except Exception as e:
        with _lock:
            _real_import_error = str(e)
    finally:
        with _lock:
            _importing = False


def _try_import():
    global _next_retry, _importing
    with _lock:
        if _real_score is not None or _importing or \
                time.monotonic() < _next_retry:
            return
        _next_retry = time.monotonic() + RETRY_SECONDS
        _importing = True
    threading.Thread(target=_do_import, daemon=True).start()


def engine_status() -> dict:
    _try_import()
    status = {"real_pipeline_importable": _real_score is not None}
    # full error text only for the team (SIH_DEBUG=1) - don't broadcast
    # internals to everyone on the venue network
    if os.environ.get("SIH_DEBUG"):
        status["import_error"] = _real_import_error
        status["retry_every_seconds"] = RETRY_SECONDS
    return status


def score_animal(side_img, rear_img, video_path, animal_record,
                 tag_img=None) -> dict:
    """tag_img is the ear-tag close-up, and is optional.

    Passed only to the real pipeline, and only when present - the baseline
    engine has no use for it, and older pipelines may not accept the argument
    at all, so a TypeError falls back to the four-argument call rather than
    taking the whole run down.
    """
    _try_import()
    if _real_score is not None:
        try:
            try:
                result = _real_score(side_img, rear_img, video_path,
                                     animal_record, tag_img=tag_img)
            except TypeError:
                result = _real_score(side_img, rear_img, video_path,
                                     animal_record)
            problems = validate(result, mode="pipeline")
            if not problems:
                # A CONTRACT-VALID RESULT IS THE ANSWER, EVEN WHEN IT SCORES
                # NOTHING.
                #
                # This used to require scored > 0 and otherwise fall through
                # to the baseline engine, on the reasoning that a pipeline
                # scoring nothing must have models missing. That reasoning is
                # wrong, and it was the single most dangerous line in the
                # server: a pipeline that has looked at a photograph and
                # refused every trait has told us something true, and we were
                # discarding it and inventing twenty scores instead.
                #
                # Measured on this build, every one of these returned HTTP 200
                # with 20/20 traits, a confident weight near 350 kg, and a
                # fabricated skin_nodules finding that wrote a Lumpy Skin
                # Disease row into a veterinary officer's alert feed:
                #     a drawing of a chair          (pipeline: 0/20, no_animal_detected)
                #     pure random RGB noise         (byte-identical output)
                #     a 44-byte ASCII text file     (byte-identical output)
                # and, less obviously but far more often, 12 of 16 real
                # photographs of Indian cattle and buffalo taken without an
                # ear-tag close-up.
                #
                # The invented figures came from random.Random(animal_id), so
                # they are stable per animal and look like a measurement that
                # reproduces. Someone could drive to a farm over one.
                #
                # So: refusing is an answer. It ships as ml-pipeline with each
                # trait's own not_scored_reason intact, and the app renders it
                # as "nothing could be measured" rather than as twenty scores.
                result["engine"] = "ml-pipeline"
                return result
            reason = f"contract violations: {len(problems)} (e.g. {problems[0]})"
        except Exception as e:
            reason = f"pipeline crashed: {e}"
    else:
        reason = f"not importable: {_real_import_error}"

    # Everything reaching here is a case where the pipeline gave us nothing
    # to report at all - it would not import, it crashed, or its output broke
    # the contract. An honest refusal no longer lands here; only a broken
    # pipeline does. The baseline engine exists to keep the demo's shape alive
    # in that situation, and its output is labelled so that every surface -
    # console, app, farmer report, vet report, alert feed - can mark it as
    # invented rather than measured.
    result = _fake_score(side_img, rear_img, video_path, animal_record)
    # 'baseline', not 'fake' - and NEVER persist the internal reason
    # into stored sessions (they're served over unauthenticated GET);
    # the reason goes to the server log only
    result["engine"] = "baseline"
    import sys
    print(f"[loader] baseline engine used - {reason}", file=sys.stderr)
    return result
