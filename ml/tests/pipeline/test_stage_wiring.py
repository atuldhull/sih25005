"""The pipeline's own guarantees, independent of any model being present.

Everything here runs with no weights installed, which is deliberate: these are
the properties that must hold on a machine that has not fetched the models,
because that is the state a teammate or a fresh clone starts in.
"""
from ml.pipeline import _breed_extra, score_animal

ADDITIVE_KEYS = [
    "predicted_species", "species_confidence", "species_consistent",
    "predicted_group", "group_confidence", "group_consistent",
    "group_reliable", "breed_verify_note",
    # Whether a veterinary screening actually ran. This build has no trained
    # symptom detector, so it never does - and an empty symptom_vector
    # therefore means NOT SCREENED, not healthy. Before this key existed the
    # farmer's report said "No health problems were flagged from today's
    # photos and video" to an animal nothing had examined.
    "vet_screened",
]
RECORD = {"animal_id": "356279812345", "_id": "356279812345",
          "species": "cattle", "breed": "Gir"}


def test_breed_extra_always_has_the_same_keys():
    for arg in (None, {}, {"predicted_group": "buffalo"}):
        assert set(_breed_extra(arg)) == set(ADDITIVE_KEYS)


def test_vet_screened_defaults_to_false_on_the_refusal_path():
    """A run that refused before reaching the screener has not screened.

    The refusal path calls _breed_extra without the flag, so the default is
    what a not-scored session reports - and it has to be the cautious one.
    """
    assert _breed_extra(None)["vet_screened"] is False
    assert _breed_extra({}, vet_screened=True)["vet_screened"] is True


def test_missing_models_degrade_instead_of_crashing():
    """No weights, no cv2, no torch - and still a contract-shaped result.

    This is what lets scoring_loader fall back to the baseline cleanly rather
    than the server logging a stack trace and the demo showing nothing.
    """
    r = score_animal("no_such_side.jpg", "no_such_rear.jpg", None, RECORD)
    assert isinstance(r, dict)
    assert "traits" in r
    assert all(k in r for k in ADDITIVE_KEYS)


def test_not_scored_path_carries_the_additive_keys_too():
    """Same key set whether scoring succeeded or not.

    Without this the app would see these fields appear and disappear
    depending on whether the animal happened to be scoreable.
    """
    r = score_animal("no_such_side.jpg", "no_such_rear.jpg", None, RECORD)
    for k in ADDITIVE_KEYS:
        assert k in r, f"{k} missing from the NOT_SCORED result"


def test_breed_verified_is_never_fabricated():
    """null means 'not verified'. It must never be invented as True.

    Exact-breed verification was measured at 38.1% source-held-out with
    uninformative confidence, so the model disables its own breed head.
    """
    r = score_animal("no_such_side.jpg", "no_such_rear.jpg", None, RECORD)
    assert r.get("breed_verified") is None


def test_unscored_result_is_blocked_by_the_adoption_gate():
    """A shape-valid but fully unscored result must NOT replace the baseline."""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "contract"))
    from validate_result import validate
    r = score_animal("no_such_side.jpg", "no_such_rear.jpg", None, RECORD)
    problems = validate(r, mode="pipeline")
    assert any("adoption gate" in p for p in problems), (
        "an all-null result passed validation - it would silently replace a "
        "working baseline engine the moment ml.pipeline became importable")
