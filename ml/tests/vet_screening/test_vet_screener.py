"""The pre-screen must never invent a symptom.

symptom_vector feeds risk_report, which produces "refer to vet". An entry
here is not a display value - it drives a real-world action. Until a detector
is trained and validated, the only honest output is an empty list plus a note
saying screening did not happen.
"""
from ml.vet_screening.vet_screener import (
    SYMMETRY_PAIRS,
    screen,
    screening_notes,
    symptom_vector_or_empty,
)


def test_no_symptoms_are_ever_emitted():
    for kw in ({}, {"video_path": "gait.mp4"},
               {"keypoints": {"hock_left": (1.0, 2.0, 0.9),
                              "hock_right": (9.0, 2.0, 0.9)}}):
        assert screen(**kw)["symptom_vector"] == []


def test_asymmetric_keypoints_do_not_become_a_lameness_claim():
    """A large left-right difference is computable and is NOT a diagnosis.

    Emitting it as gait_asymmetry would look identical to a validated symptom
    while carrying none of the evidence.
    """
    lopsided = {"hock_left": (10.0, 10.0, 0.95),
                "hock_right": (400.0, 300.0, 0.95),
                "pastern_left": (12.0, 90.0, 0.9),
                "pastern_right": (390.0, 95.0, 0.9)}
    assert screen(keypoints=lopsided)["symptom_vector"] == []


def test_an_unused_gait_video_is_reported():
    """Silence would read as 'we checked the gait and it was fine'."""
    notes = screening_notes(screen(video_path="gait.mp4"))
    assert any("not analysed" in n for n in notes)


def test_no_video_means_no_video_note():
    assert not any("not analysed" in n for n in screening_notes(screen()))


def test_empty_is_labelled_not_screened_rather_than_healthy():
    notes = screening_notes(screen())
    assert any("NOT SCREENED" in n for n in notes)
    assert screen()["screened"] is False


def test_symptom_vector_is_always_a_list():
    """The app iterates it; None would be a crash rather than an empty state."""
    assert symptom_vector_or_empty(None) == []
    assert symptom_vector_or_empty({}) == []
    assert symptom_vector_or_empty({"symptom_vector": None}) == []
    assert symptom_vector_or_empty(screen()) == []


def test_symmetry_pairs_are_real_left_right_joints():
    for left, right in SYMMETRY_PAIRS:
        assert left.endswith("_left") and right.endswith("_right")
        assert left[:-5] == right[:-6]
