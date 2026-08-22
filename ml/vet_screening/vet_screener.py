"""Stage 7: the veterinary pre-screen.

WHAT THIS DELIBERATELY DOES NOT DO

It emits no symptoms. Not because the wiring is missing - it is all here -
but because there is no trained detector behind it, and a symptom here is not
a display value. `symptom_vector` feeds `risk_report`, which produces
"refer to vet". Inventing an entry would send a farmer to a veterinarian, or
worse, fail to when something is wrong.

The temptation is real and worth naming. From the keypoints we already have,
a left-right asymmetry in hock or pastern angles is trivially computable, and
it correlates with lameness in the literature. But "correlates in the
literature" is not the same as "validated on this model's output for this
species at this image quality", and nothing here has been. An asymmetry
number dressed up as `gait_asymmetry` with a confidence attached would look
exactly like the other symptoms in the vector and carry none of the evidence.

So the function returns an empty vector and says why. When a detector is
trained, it fills this in and the pipeline needs no other change.

WHAT IT DOES DO

Reports honestly that a gait video was supplied and not analysed. That is a
data-completeness fact, not a clinical claim, and the farmer should know the
video they took was not used - otherwise a silent empty screen looks like a
clean bill of health.
"""
from typing import Any, Dict, List, Optional, Tuple

# Left/right joint pairs that a future lameness detector would compare. Kept
# here so the anatomy is written down once, in the module that will need it,
# rather than rediscovered later.
SYMMETRY_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("hock_left", "hock_right"),
    ("knee_left", "knee_right"),
    ("pastern_left", "pastern_right"),
    ("hoof_left", "hoof_right"),
    ("hook_left", "hook_right"),
    ("pin_left", "pin_right"),
)


def screen(
    keypoints: Optional[Dict[str, Tuple[float, float, float]]] = None,
    video_path: Optional[str] = None,
    species: str = "cattle",
) -> Dict[str, Any]:
    """Run the pre-screen.

    Returns {"symptom_vector": [...], "notes": [...], "screened": bool}.

    symptom_vector is ALWAYS a list, never None, so the pipeline and the app
    can iterate it without a None check - and an empty list means "nothing
    found or nothing looked for", which `notes` then disambiguates.
    """
    notes: List[str] = []

    if video_path:
        # The farmer recorded a video. It is not being used, and silence here
        # would read as "we checked the gait and it was fine".
        notes.append(
            "gait video was provided but not analysed - no gait model is "
            "trained yet, so lameness was NOT screened for")

    usable = 0
    if keypoints:
        usable = sum(1 for v in keypoints.values()
                     if isinstance(v, (tuple, list)) and len(v) >= 3
                     and v[2] > 0)
        if usable == 0:
            notes.append("no usable keypoints, so no postural screen was "
                         "possible")

    notes.append(
        "no veterinary screening was performed: this build has no trained "
        "symptom detector. An empty symptom list here means NOT SCREENED, "
        "not HEALTHY.")

    return {"symptom_vector": [], "notes": notes, "screened": False}


def symptom_vector_or_empty(result: Optional[Dict[str, Any]]) -> List[dict]:
    """The list the contract expects, defaulting to empty rather than None."""
    if not result:
        return []
    v = result.get("symptom_vector")
    return v if isinstance(v, list) else []


def screening_notes(result: Optional[Dict[str, Any]]) -> List[str]:
    return list((result or {}).get("notes") or [])
