"""Validate image/video quality (blur, exposure, resolution, framing) before downstream processing."""

import os
from typing import List

from ml.detection.detector import DetectionBackendError

BLUR_VARIANCE_THRESHOLD = 100.0
MIN_WIDTH = 640
MIN_HEIGHT = 480
BRIGHTNESS_MIN = 30
BRIGHTNESS_MAX = 220
MIN_FRAMES = 30
FRAME_SAMPLE_INTERVAL = 10


def _get_cv2():
    """Lazily import cv2 (see ml/detection/detector.py's _get_cv2 for why:
    this module previously imported cv2 at module scope too, which
    independently blocked `import ml.pipeline` from succeeding without cv2
    installed, even after detector.py's own fix). Reuses
    DetectionBackendError so both modules signal the same failure mode."""
    try:
        import cv2
        return cv2
    except ImportError as exc:
        raise DetectionBackendError(
            "'opencv-python' (cv2) is not installed. Install it with "
            "`pip install opencv-python` to use image/video quality validation."
        ) from exc


def _load_image(image_path: str):
    """Read an image file, returning (image, None) or (None, reason) on failure."""
    if not os.path.exists(image_path):
        return None, "file_not_found"
    cv2 = _get_cv2()
    image = cv2.imread(image_path)
    if image is None:
        return None, "unreadable_image"
    return image, None


# Blur is measured at a FIXED working size. Laplacian variance counts
# per-pixel high-frequency content, so the same photograph scores wildly
# differently depending only on its resolution - measured on one real image:
#
#     5184x3456 (original)   62.9   <- would FAIL a threshold of 100
#     2000x1333             271.6
#     1200x800              490.0
#      640x426              755.4   <- passes easily
#
# A 12x swing on identical pixels. With a fixed threshold that rejects sharp
# photographs from a modern phone while accepting compressed thumbnails -
# exactly backwards, and it would refuse the very images this system is built
# for. Normalising the long side first makes the number comparable across
# devices, which is the only way a single threshold can mean anything.
BLUR_WORK_LONG_SIDE = 1024


def _blur_score(image) -> float:
    """Laplacian variance at a normalised size; lower means more blur."""
    cv2 = _get_cv2()
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest > BLUR_WORK_LONG_SIDE:
        scale = BLUR_WORK_LONG_SIDE / float(longest)
        image = cv2.resize(image, (max(1, int(w * scale)),
                                   max(1, int(h * scale))),
                           interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _brightness_mean(image) -> float:
    """Return mean pixel brightness in the standard 0-255 gamma space."""
    cv2 = _get_cv2()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def _frame_metrics(image) -> dict:
    """Compute the raw quality metrics dict for a single frame."""
    height, width = image.shape[:2]
    return {
        "blur_score": _blur_score(image),
        "brightness_mean": _brightness_mean(image),
        "width": int(width),
        "height": int(height),
    }


def _screen_metrics(metrics: dict) -> List[str]:
    """Return the list of failure reasons implied by a metrics dict."""
    reasons = []
    if metrics["blur_score"] < BLUR_VARIANCE_THRESHOLD:
        reasons.append("blur_too_high")
    if metrics["brightness_mean"] < BRIGHTNESS_MIN:
        reasons.append("underexposed")
    elif metrics["brightness_mean"] > BRIGHTNESS_MAX:
        reasons.append("overexposed")
    if metrics["width"] < MIN_WIDTH or metrics["height"] < MIN_HEIGHT:
        reasons.append("resolution_too_low")
    aspect = metrics["width"] / float(metrics["height"]) if metrics["height"] else 0.0
    if aspect < 0.4 or aspect > 2.5:
        reasons.append("bad_aspect_ratio")
    return reasons


def validate_image(image_path: str, image_type: str = "side") -> dict:
    """Validate a single image for blur, exposure, and minimum resolution.

    Returns {"passed", "reasons", "metrics"}; passed is False (with a clear
    reason) if the file is missing, unreadable, or any metric check fails.
    """
    image, load_reason = _load_image(image_path)
    if image is None:
        return {"passed": False, "reasons": [load_reason], "metrics": {}}

    metrics = _frame_metrics(image)
    reasons = _screen_metrics(metrics)
    return {"passed": not reasons, "reasons": reasons, "metrics": metrics}


def validate_video(video_path: str) -> dict:
    """Validate a video for accessibility, duration, and sampled-frame quality.

    Samples every FRAME_SAMPLE_INTERVAL-th frame, aggregates blur/exposure
    checks across samples, and requires at least MIN_FRAMES total frames.
    """
    if not os.path.exists(video_path):
        return {"passed": False, "reasons": ["file_not_found"], "metrics": {}}

    cv2 = _get_cv2()
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return {"passed": False, "reasons": ["unreadable_video"], "metrics": {}}

    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = _count_frames(capture)

        reasons = []
        if total_frames < MIN_FRAMES:
            reasons.append("video_too_short")

        sample_metrics = []
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % FRAME_SAMPLE_INTERVAL == 0:
                sample_metrics.append(_frame_metrics(frame))
            frame_index += 1
    finally:
        capture.release()

    if not sample_metrics:
        return {"passed": False, "reasons": ["no_decodable_frames"], "metrics": {}}

    for metrics in sample_metrics:
        reasons += _screen_metrics(metrics)
    reasons = list(dict.fromkeys(reasons))

    worst_blur = min(m["blur_score"] for m in sample_metrics)
    overall = {
        "total_frames": int(total_frames),
        "samples_checked": len(sample_metrics),
        "blur_score": worst_blur,
        "brightness_mean": sum(m["brightness_mean"] for m in sample_metrics) / len(sample_metrics),
        "width": min(m["width"] for m in sample_metrics),
        "height": min(m["height"] for m in sample_metrics),
    }

    return {"passed": not reasons, "reasons": reasons, "metrics": overall}


def _count_frames(capture) -> int:
    """Count frames by reading through the capture when the metadata is missing."""
    cv2 = _get_cv2()
    count = 0
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        count += 1
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return count