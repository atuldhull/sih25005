"""Test score_animal() degrades truthfully on an unrecognized detector label.

A checkpoint whose native class names don't match RAW_LABEL_TO_CLASS_NAME
(e.g. "ears" instead of "ear_tag") makes detect_animal raise
DetectionLabelError. pipeline.py must catch it and return a NOT_SCORED
result with a truthful warning — never propagate a crash.
"""

import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, "ml_pipeline")

from detection import detector  # noqa: E402
from pipeline import score_animal  # noqa: E402


class FakeUnknownLabelModel:
    """Minimal model whose predictions carry a label missing from the label map."""

    def predict(self, image, verbose=False):
        return [{"label": "ears", "score": 0.92, "bbox": (1, 2, 300, 150)}]


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
    monkeypatch.setattr(detector, "load_rt_detr", lambda device=None: FakeUnknownLabelModel())
    side, rear = quality_images

    result = score_animal(side, rear, None, {"animal_id": None, "species": "cattle"})

    assert result["status"] == "NOT_SCORED"
    assert any("detection_label_unrecognized" in w for w in result["warnings"])
    assert any("ears" in w for w in result["warnings"])
    assert result["eligibility"]["passed"] is False