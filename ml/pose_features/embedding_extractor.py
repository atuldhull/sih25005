"""Stage 4b: DINOv2 features, used to check the recorded breed.

WHAT THIS ANSWERS
Not "what breed is this animal" - "is this photo consistent with the breed on
the BPA record". Three outcomes matter: agree, disagree (a misregistered
animal is exactly what the system exists to catch), and abstain.

WHY THE EXACT-BREED ANSWER IS SWITCHED OFF
Measured source-held-out - trained on one photo source, tested on a different
one - exact breed scores 38.1% against a 17.7% background-only control. The
margin is real, but the confidence carries no information: tightening the
threshold from answering 100% of the time to 30% moves accuracy just +5.6
points. And three of eight breeds are broken outright (Surti 0%, Mehsana 3%,
Murrah 24% - it has learnt "any buffalo is a Murrah"). So the checkpoint
disables its own breed head, and breed_verified stays null.

WHAT DOES WORK
The coarser GROUP call - red_zebu / grey_draught / dwarf_cattle /
exotic_dairy / buffalo - reaches 80.2% against a 60.7% background control,
and 85.4% when it answers 80% of the time. Buffalo specifically is 98%. The
errors the breed head makes are almost entirely WITHIN group (Gir into
Sahiwal, Kankrej into Tharparkar, Surti into Murrah), which is precisely why
the group is the claim the images can support.

THE CONTROL THAT MADE ALL THIS HONEST
A first attempt scored 97.9% on a random split and 95.7% with the animal
ERASED from the picture - it had learnt which farm, not which breed. Every
number above is quoted next to a background-only control for that reason, and
validation holds out a whole photo source rather than random images.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from ml.config.models import BREED_ENABLED, BREED_MODEL_PATH

_verifier = None
_load_error: Optional[str] = None


class BreedBackendError(RuntimeError):
    """The breed verifier could not be loaded or run."""


def _get_verifier(model_path: Optional[str] = None):
    global _verifier, _load_error
    if _verifier is not None:
        return _verifier
    path = Path(model_path or BREED_MODEL_PATH)
    if not path.exists():
        _load_error = (
            f"breed verifier not found at {path}. Weights are not in git - "
            f"run scripts/fetch_models.py, see models/README.md")
        raise BreedBackendError(_load_error)
    try:
        from ml.pose_features.breed_verify_infer import BreedVerifier
        _verifier = BreedVerifier(str(path))
    except Exception as exc:
        _load_error = f"breed verifier failed to load: {exc}"
        raise BreedBackendError(_load_error) from exc
    return _verifier


def verify_breed(image_path: str, animal_bbox: Sequence[float],
                 claimed_breed: Optional[str] = None,
                 model_path: Optional[str] = None) -> Dict[str, Any]:
    """Check a photo against the recorded breed.

    Cropping to animal_bbox is not optional. On the full frame the classifier
    reads the scenery, which is the failure the background control exposed.

    Every field it returns can be None, and each None is a deliberate refusal
    rather than a missing implementation. The measured accuracy travels back
    with the answer in 'measured_accuracy' so nothing downstream has to guess
    how much to trust it.
    """
    if not BREED_ENABLED:
        return {"breed_verified": None, "abstained": True,
                "reason": "breed verification disabled in config"}
    v = _get_verifier(model_path)
    try:
        return v.verify(image_path, bbox=animal_bbox,
                        claimed_breed=claimed_breed)
    except Exception as exc:
        raise BreedBackendError(f"breed verification failed: {exc}") from exc


def to_contract_fields(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Map a verify() result onto the contract's fields.

    breed_verified stays None unless the exact-breed head actually answered,
    which - on the current data - it never does. The group fields are
    additive: the app can ignore them safely today and use them once Person 1
    renders them.
    """
    out: Dict[str, Any] = {
        "breed_verified": None,
        "breed_verify_confidence": None,
        "predicted_species": None,
        "species_confidence": None,
        "predicted_group": None,
        "group_confidence": None,
        "group_consistent": None,
        "group_reliable": None,
        "breed_verify_note": None,
    }
    if not result:
        return out
    out["breed_verified"] = result.get("breed_verified")
    out["breed_verify_confidence"] = result.get("breed_verify_confidence")
    out["predicted_species"] = result.get("predicted_species")
    out["species_confidence"] = result.get("species_confidence")
    out["predicted_group"] = result.get("predicted_group")
    out["group_confidence"] = result.get("group_confidence")
    out["group_consistent"] = result.get("group_consistent")
    out["group_reliable"] = result.get("group_reliable")
    out["breed_verify_note"] = result.get("group_note") or result.get("reason")
    return out


def species_or(default: str, fields: Dict[str, Any],
               min_confidence: float = 0.90) -> str:
    """The species to score against.

    Falls back to the BPA record's species rather than a low-confidence
    guess. This choice decides which NDDB trait rubric applies - buffalo
    teat traits are measured on the left REAR teat, and rump landmarks
    differ - so a wrong call here corrupts every trait, not just one.
    """
    got = fields.get("predicted_species")
    conf = fields.get("species_confidence") or 0.0
    if got and conf >= min_confidence:
        return got
    return default


def load_error() -> Optional[str]:
    return _load_error
