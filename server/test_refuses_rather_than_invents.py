"""A photograph of a chair must not come back as a scored animal.

The server carried an "adoption gate": a pipeline result in which all twenty
traits scored null was treated as a contract violation, so that a half-built
pipeline could not displace the working baseline engine. The intent was sound.
The effect was the opposite of the name.

Because scoring_loader answers a rejected result by calling the BASELINE
engine - which invents all twenty scores, a weight, and symptoms from
random.Random(animal_id) - the gate did not suppress an honest refusal. It
replaced it with a fabrication. Measured before the fix, all of these returned
HTTP 200 with twenty confident scores and a weight near 350 kg:

    a drawing of a chair        (the pipeline itself: 0/20, no_animal_detected)
    pure random RGB noise       (byte-identical output)
    a 44-byte ASCII text file   (byte-identical output, and a fabricated
                                 skin_nodules finding at confidence 0.82 that
                                 wrote a Lumpy Skin Disease row into a
                                 veterinary officer's alert feed)

and so did 12 of 16 real photographs of Indian cattle and buffalo taken
without an ear-tag close-up - which is the case that would actually have
happened in a village, over and over, unnoticed.

This test drives the loader the way the /session route does and pins the rule
that replaced the gate: what the pipeline says is what ships, including when
what it says is "I could not measure anything here".

    venv/Scripts/python test_refuses_rather_than_invents.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np  # noqa: E402
import cv2  # noqa: E402

import scoring_loader  # noqa: E402

failures = 0


def check(ok, label):
    global failures
    print(("PASS  " if ok else "FAIL  ") + label)
    if not ok:
        failures += 1


ANIMAL = {"_id": "356279812345", "breed": "Gir", "species": "cattle",
          "village": "Anand", "owner": "R. Patel", "dob": "2021-03-14",
          "lactation_no": 1, "last_calving_date": "2026-06-20"}


def _write(tmp: Path, name: str, img) -> str:
    path = tmp / name
    cv2.imwrite(str(path), img)
    return str(path)


def _chair() -> "np.ndarray":
    img = np.full((900, 700, 3), 245, np.uint8)
    for a, b in (((200, 300), (500, 340)), ((200, 340), (240, 800)),
                 ((460, 340), (500, 800)), ((200, 120), (240, 300)),
                 ((460, 120), (500, 300)), ((200, 150), (500, 200))):
        cv2.rectangle(img, a, b, (90, 60, 40), -1)
    return img


def _noise() -> "np.ndarray":
    return np.random.default_rng(7).integers(0, 255, (900, 700, 3),
                                             dtype=np.uint8)


# The loader only reaches the real pipeline when ml/ imports. Without it there
# is nothing to test and a silent pass would be worse than a skip: this file
# exists precisely to catch the fallback.
status = scoring_loader.engine_status()
if not status.get("real_pipeline_importable"):
    import time
    for _ in range(40):
        time.sleep(2)
        if scoring_loader.engine_status().get("real_pipeline_importable"):
            break
if not scoring_loader.engine_status().get("real_pipeline_importable"):
    print("SKIP  ml.pipeline is not importable, so the fallback cannot be "
          "exercised - this test proves nothing here")
    sys.exit(0)


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    cases = {
        "a drawing of a chair": _write(tmp, "chair.jpg", _chair()),
        "pure random noise": _write(tmp, "noise.jpg", _noise()),
    }
    text = tmp / "notaphoto.txt"
    text.write_text("this is not a photograph of any animal", encoding="utf-8")
    cases["a text file"] = str(text)

    for label, path in cases.items():
        print(f"\n--- {label} ---")
        result = scoring_loader.score_animal(path, path, None, ANIMAL)

        engine = result.get("engine")
        traits = result.get("traits", [])
        scored = sum(1 for t in traits
                     if isinstance(t, dict) and t.get("score") is not None)
        weight = result.get("weight_kg") or {}
        symptoms = result.get("symptom_vector") or []

        check(engine == "ml-pipeline",
              f"the pipeline's own answer is returned, not the inventing "
              f"baseline (engine={engine!r})")
        check(scored == 0,
              f"nothing is scored from {label} (scored={scored})")
        check(weight.get("low") is None and weight.get("high") is None,
              f"no weight is reported (got {weight.get('low')}-"
              f"{weight.get('high')})")
        check(not symptoms,
              f"no symptoms are invented, so nothing reaches a vet's alert "
              f"feed ({len(symptoms)} found)")
        check(len(traits) == 20,
              f"all twenty traits are still present as refusals "
              f"({len(traits)} found)")
        missing = [t.get("name") for t in traits
                   if isinstance(t, dict) and t.get("score") is None
                   and not t.get("not_scored_reason")]
        check(not missing,
              f"every refusal says why - a blank row is not an answer "
              f"({missing[:3]})")


print("\n" + ("ALL CHECKS PASSED" if not failures else f"{failures} FAILED"))
sys.exit(1 if failures else 0)
