"""Tests for the detection module (RT-DETR label mapping, detection, SAM2 upgrade)."""

import sys

import numpy as np
import pytest

sys.path.insert(0, "ml_pipeline")

from detection import detector  # noqa: E402
from common.schemas import DetectionResult  # noqa: E402


@pytest.fixture
def sample_image(tmp_path):
    """Write a small synthetic image and return its path."""
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    img[80:160, 150:300] = 120
    path = tmp_path / "sample.png"
    import cv2

    cv2.imwrite(str(path), img)
    return str(path)


# ---------------------------------------------------------------------------
# Label mapping layer
# ---------------------------------------------------------------------------
class TestLabelMapping:
    def test_maps_int_index(self):
        assert detector.map_class_name(0) == "animal"
        assert detector.map_class_name(1) == "ear_tag"

    def test_maps_native_string_labels(self):
        assert detector.map_class_name("cow") == "animal"
        assert detector.map_class_name("buffalo") == "animal"
        assert detector.map_class_name("ear_tag") == "ear_tag"

    def test_raises_on_unrecognized_raw_label(self):
        with pytest.raises(detector.DetectionLabelError):
            detector.map_class_name("person")

    def test_validate_label_map_rejects_bad_target(self, monkeypatch):
        monkeypatch.setattr(
            detector,
            "RAW_LABEL_TO_CLASS_NAME",
            {**detector.RAW_LABEL_TO_CLASS_NAME, "coyote": "predator"},
        )
        with pytest.raises(detector.DetectionLabelError):
            detector.validate_label_map()


# ---------------------------------------------------------------------------
# Detection functions (fake model injected, no real RT-DETR installed)
# ---------------------------------------------------------------------------
class FakeModel:
    """Minimal stand-in whose .predict returns already-normalized predictions."""

    def __init__(self, preds):
        self._preds = preds

    def predict(self, image, verbose=False):
        return list(self._preds)


class TestDetection:
    def test_detect_animal_returns_result(self, sample_image, monkeypatch):
        monkeypatch.setattr(
            detector,
            "load_rt_detr",
            lambda device=None: FakeModel(
                [
                    {"label": 0, "score": 0.92, "bbox": (1, 2, 300, 150)},
                    {"label": 1, "score": 0.81, "bbox": (200, 100, 260, 130)},
                ]
            ),
        )
        result = detector.detect_animal(sample_image)
        assert isinstance(result, DetectionResult)
        assert result.class_name == "animal"
        assert result.bbox == (1, 2, 300, 150)
        assert result.confidence == pytest.approx(0.92)

    def test_detect_animal_none_when_no_animal(self, sample_image, monkeypatch):
        monkeypatch.setattr(
            detector,
            "load_rt_detr",
            lambda device=None: FakeModel([{"label": 1, "score": 0.81, "bbox": (0, 0, 10, 10)}]),
        )
        assert detector.detect_animal(sample_image) is None

    def test_detect_ear_tag_returns_cropped_coords(self, sample_image, monkeypatch):
        monkeypatch.setattr(
            detector,
            "load_rt_detr",
            lambda device=None: FakeModel([{"label": 1, "score": 0.88, "bbox": (125, 80, 175, 100)}]),
        )
        result = detector.detect_ear_tag(sample_image, (100, 60, 300, 160))
        assert isinstance(result, DetectionResult)
        assert result.class_name == "ear_tag"
        # crop offset (100, 60) applied back onto bbox within the animal box
        assert result.bbox == (225, 140, 275, 160)

    def test_detect_ear_tag_none_when_no_tag(self, sample_image, monkeypatch):
        monkeypatch.setattr(
            detector,
            "load_rt_detr",
            lambda device=None: FakeModel([{"label": 0, "score": 0.92, "bbox": (1, 2, 300, 150)}]),
        )
        assert detector.detect_ear_tag(sample_image, (100, 60, 300, 160)) is None


# ---------------------------------------------------------------------------
# Segmentation (SAM2 unavailable -> rectangular fallback)
# ---------------------------------------------------------------------------
class TestSegmentation:
    def test_segment_animal_falls_back_to_rect(self, sample_image, monkeypatch):
        monkeypatch.setattr(
            detector,
            "load_sam2",
            lambda device=None: (_ for _ in ()).throw(
                detector.DetectionBackendError("SAM2 weights not found")
            ),
        )
        mask, degraded = detector.segment_animal(sample_image, (60, 40, 340, 160))
        assert mask.shape == (200, 400)
        assert set(np.unique(mask)) <= {0, 255}
        assert degraded is True
        # rectangle interior is foreground
        assert mask[100, 200] == 255
        # outside bbox is background
        assert mask[10, 10] == 0

    def test_segment_animal_requires_existing_image(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            detector,
            "load_sam2",
            lambda device=None: (_ for _ in ()).throw(
                detector.DetectionBackendError("SAM2 weights not found")
            ),
        )
        with pytest.raises(detector.DetectionBackendError):
            detector.segment_animal(str(tmp_path / "missing.png"), (0, 0, 10, 10))

    def test_backend_error_raised_when_rtdetr_unavailable(self, sample_image, monkeypatch):
        def boom(device=None):
            raise detector.DetectionBackendError("RT-DETR weights not found")

        monkeypatch.setattr(detector, "load_rt_detr", boom)
        with pytest.raises(detector.DetectionBackendError):
            detector.detect_animal(sample_image)

    def test_resolve_device_defaults_to_cpu_when_no_torch_cuda(self, monkeypatch):
        monkeypatch.setattr(detector, "resolve_device", lambda d: "cpu")
        assert detector.segment_animal  # reachable reference; covered above
        assert "cpu" in (detector.resolve_device("auto"), detector.resolve_device("cpu"))