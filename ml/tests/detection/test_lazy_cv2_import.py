"""Regression test for lazy cv2 imports (Rev 2 audit item 7 / implementation
item 15).

cv2 used to be imported at module scope in both ml/detection/detector.py and
ml/ingestion/quality_validation.py, so a missing cv2 installation raised
ModuleNotFoundError the moment `import ml.pipeline` ran - before the
DetectionBackendError try/except in ml/pipeline.py ever got a chance to
handle it, and before item 1's adoption gate existed to make that safe. Both
modules must have no module-level "cv2" name bound; cv2 is imported lazily
inside each function that needs it via a local _get_cv2() helper that
translates a missing installation into DetectionBackendError.

This is checked structurally (no "cv2" key in the module's own __dict__)
rather than by actually uninstalling cv2, since cv2 IS installed in the test
environment - see this session's manual verification (blocking cv2 via
builtins.__import__) for the "actually missing" case, run once at
implementation time, not repeated here as an automated test since faking a
missing installed package safely across test runs is its own hazard.
"""

import ml.detection.detector as detector_module
import ml.ingestion.quality_validation as quality_module
from ml.detection.detector import DetectionBackendError


def test_detector_module_has_no_module_level_cv2():
    assert "cv2" not in vars(detector_module)


def test_quality_validation_module_has_no_module_level_cv2():
    assert "cv2" not in vars(quality_module)


def test_detector_defines_lazy_cv2_helper():
    assert hasattr(detector_module, "_get_cv2")


def test_quality_validation_defines_lazy_cv2_helper():
    assert hasattr(quality_module, "_get_cv2")


def test_get_cv2_returns_real_cv2_when_installed():
    """In this environment cv2 IS installed, so the helper should just
    return it - confirming the happy path still works, not just the
    missing-dependency path."""
    cv2 = detector_module._get_cv2()
    assert cv2.__name__ == "cv2"


def test_quality_validation_imports_detection_backend_error():
    """Both modules must raise the SAME exception type for a missing cv2,
    not two different error types for the same underlying failure mode."""
    assert quality_module.DetectionBackendError is DetectionBackendError
