"""Test score_animal() degrades truthfully on an unrecognized detector label.

A checkpoint whose native class names don't match RAW_LABEL_TO_CLASS_NAME
(e.g. label id 99 instead of 0/1) makes detect_animal raise
DetectionLabelError. pipeline.py must catch it and return a contract-shaped
result where every trait is honestly not_scored, with the real reason
surfaced in not_scored_reason - never propagate a crash, and never silently
return an internal-shape dict the server can't parse.
"""
import cv2
import numpy as np
import pytest

from ml.detection import detector  # noqa: E402
from ml.pipeline import score_animal  # noqa: E402


@pytest.fixture
def quality_images(tmp_path):
    """Two images that satisfy the ingestion quality gate (>=640x480, noisy)."""
    img = np.random.default_rng(0).normal(120, 25, (600, 800, 3)).astype(np.uint8)
    side = tmp_path / "side.png"
    rear = tmp_path / "rear.png"
    cv2.imwrite(str(side), img)
    cv2.imwrite(str(rear), img)
    return str(side), str(rear)


def test_unrecognized_label_degrades_to_not_scored(quality_images, monkeypatch):
    monkeypatch.setattr(detector, "load_rt_detr", lambda device=None: (object(), object()))
    monkeypatch.setattr(
        detector,
        "_run_detector",
        lambda image_bgr, model_and_processor, threshold=0.5: [
            {"label": 99, "score": 0.92, "bbox": (1, 2, 300, 150)}
        ],
    )
    side, rear = quality_images
    result = score_animal(side, rear, None, {"animal_id": None, "species": "cattle"})

    # CHANGED (fixed the "adapter never called" bug flagged by review): the
    # contract has no top-level status/warnings field in pipeline mode, so
    # the only honest place a degraded run can surface "why" is inside every
    # trait's not_scored_reason. Assert on the contract shape, not the old
    # internal ScoringResult shape.
    assert "traits" in result
    assert len(result["traits"]) == 20, "contract requires exactly 20 traits, even when none are scored"

    for trait in result["traits"]:
        assert trait["score"] is None, f"{trait['name']} should be unscored on a detection failure"
        assert trait["not_scored_reason"] is not None
        assert "detection_label_unrecognized" in trait["not_scored_reason"]
        assert "99" in trait["not_scored_reason"]

    # eligible/eligible_reason are server-owned in pipeline mode - this
    # module must not claim to know NDDB eligibility, so they should not be
    # asserted true/false here. captured reflects what was actually provided.
    assert result["captured"]["side_photo"] is True
    assert result["captured"]["rear_photo"] is True
    assert result["captured"]["gait_video"] is False