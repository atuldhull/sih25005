"""Breed/group verification: the shape of the answer and its refusals.

The measured position these tests encode: exact-breed verification does not
work on data we can legally use (38.1% source-held-out, uninformative
confidence), so breed_verified stays null. The GROUP call does work - 80.2%
against a 60.7% background-only control - and is what gets reported.
"""
from ml.pose_features import embedding_extractor as ee

EXPECTED_KEYS = {
    "breed_verified", "breed_verify_confidence",
    "predicted_species", "species_confidence",
    "predicted_group", "group_confidence", "group_consistent",
    "group_reliable", "breed_verify_note",
}


def test_no_result_still_produces_every_key():
    """The key set must never change between runs.

    A field that sometimes exists and sometimes does not is much harder for
    the app to handle than one that is always present and sometimes null.
    """
    out = ee.to_contract_fields(None)
    assert set(out) == EXPECTED_KEYS
    assert all(v is None for v in out.values())


def test_fields_are_copied_through_verbatim():
    out = ee.to_contract_fields({
        "breed_verified": None,
        "predicted_group": "red_zebu",
        "group_confidence": 0.93,
        "group_consistent": True,
        "group_reliable": True,
        "predicted_species": "cattle",
        "species_confidence": 0.98,
    })
    assert out["predicted_group"] == "red_zebu"
    assert out["group_consistent"] is True
    assert out["breed_verified"] is None, "the breed head stays silent"


def test_species_falls_back_when_the_model_is_unsure():
    """Species decides which NDDB rubric applies.

    Buffalo teat traits are measured on the left REAR teat and rump landmarks
    differ, so a wrong species corrupts every trait at once rather than one.
    Below the threshold we keep the registry's answer.
    """
    unsure = {"predicted_species": "buffalo", "species_confidence": 0.55}
    assert ee.species_or("cattle", unsure) == "cattle"

    confident = {"predicted_species": "buffalo", "species_confidence": 0.97}
    assert ee.species_or("cattle", confident) == "buffalo"

    assert ee.species_or("cattle", {}) == "cattle"


def test_missing_checkpoint_message_is_actionable(tmp_path):
    import pytest
    with pytest.raises(ee.BreedBackendError) as exc:
        ee.verify_breed("x.jpg", (0, 0, 10, 10), "Gir",
                        model_path=str(tmp_path / "absent.pt"))
    assert "fetch_models" in str(exc.value)


def test_disabled_in_config_abstains_rather_than_guessing(monkeypatch):
    monkeypatch.setattr(ee, "BREED_ENABLED", False)
    out = ee.verify_breed("x.jpg", (0, 0, 10, 10), "Gir")
    assert out["breed_verified"] is None
    assert out["abstained"] is True
