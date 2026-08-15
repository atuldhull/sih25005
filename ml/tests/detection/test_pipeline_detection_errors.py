"""Test score_animal() degrades truthfully on an unrecognized detector label.

A checkpoint whose native class names don't match RAW_LABEL_TO_CLASS_NAME
(e.g. label id 99 instead of 0/1) makes detect_animal raise
DetectionLabelError. pipeline.py must catch it and return a NOT_SCORED
result with a truthful warning - never propagate a crash.
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
    # CHANGED (REVIEW-ml-dev.md B4 fallout): mock load_rt_detr to return the
    # (model, processor) tuple the real code now expects, and mock
    # detector._run_detector directly to inject a prediction whose label
    # (99) isn't in RAW_LABEL_TO_CLASS_NAME - the same scenario as before,
    # just expressed through the new backend-agnostic mocking boundary.
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
    assert result["status"] == "NOT_SCORED"
    assert any("detection_label_unrecognized" in w for w in result["warnings"])
    assert any("99" in w for w in result["warnings"])
    assert result["eligibility"]["passed"] is False