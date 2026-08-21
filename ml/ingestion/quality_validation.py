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
    metrics = {
        "blur_score": _blur_score(image),
        "brightness_mean": _brightness_mean(image),
        "width": int(width),
        "height": int(height),
    }
    # A soft photograph and an empty one score the same on Laplacian variance
    # and need opposite advice: one is worth keeping, the other means move
    # closer or clean the lens. Telling them apart is a resolution question,
    # not a contrast one - see ml/ingestion/deblur.py.
    try:
        from ml.ingestion.deblur import assess
        report = assess(image)
        metrics["sharpness_verdict"] = report.verdict
        metrics["detail_rmse"] = round(report.detail_rmse, 2)
    except Exception:
        pass          # the extra detail is a bonus, never a blocker
    return metrics


# Which defects are worth REFUSING over, and which are worth recording.
#
# Blur was fatal until it was measured against what it actually costs. Blurring
# one photograph by a known amount and re-running detection and pose:
#
#   sigma  blur score   usable joints   keypoint drift
#     0       371.3           8              -
#     1        52.3           8           0.18%
#     2        10.8           9           0.49%
#     3         4.7           9           0.94%
#     4.5       2.8           9           1.09%
#     6         2.56          9           1.52%
#     8         2.14          7           2.85%
#    12         1.89          6           3.81%
#    16         1.76          2           2.67%
#    24         1.64          0            -
#
# Two things follow. First, the pose model degrades gently: out to sigma 6 the
# drift stays inside its own measured median error of 1.56% of the box side, so
# images scoring 2.5 still produce usable joints. A threshold of 100 was
# rejecting them. This repo's own demo side.jpg scores 8.9 and yields 8 usable
# joints and a measured trait - it was being refused at stage 1 before anything
# ever looked at it.
#
# Second, and worse for a gate: the score SATURATES. From sigma 4.5 to sigma 24
# it moves 2.8 -> 1.64 while the pipeline goes from fully usable to producing
# no joints at all. Almost all of its range is spent on the difference between
# sharp and slightly soft, and almost none on the difference between usable and
# hopeless. No threshold placed on it can separate the cases that matter.
#
# So blur is recorded, not refused. The honest gate is the outcome one the
# pipeline already has - usable_joint_count == 0 - which measures whether the
# image worked instead of predicting it from a proxy that cannot tell.
# quality_passed still goes False, which flows into the not-scored reasons and
# the eligibility calculation, so a soft photograph is reported as a soft
# photograph rather than silently accepted.
# Neither blur_too_high nor no_fine_detail appears here: both are advisory.
FATAL_REASONS = frozenset({
    "file_not_found",
    "unreadable_image",
    "resolution_too_low",     # below 640x480 there is genuinely too little
    "bad_aspect_ratio",       # a panorama or a sliver is not a photo of an animal
})


def is_fatal(reasons: List[str]) -> bool:
    """Do these defects mean the pipeline cannot proceed at all?

    Anything not listed is advisory: it degrades confidence and is reported,
    but the image still gets its chance. Refusing to try is only correct when
    trying cannot possibly work.
    """
    return any(r in FATAL_REASONS for r in reasons)


def _screen_metrics(metrics: dict) -> List[str]:
    """Return the list of failure reasons implied by a metrics dict."""
    reasons = []
    if metrics["blur_score"] < BLUR_VARIANCE_THRESHOLD:
        # Both are advisory - neither stops the pipeline - but they mean
        # different things to whoever took the photograph. "Soft" is worth
        # keeping and usually still measures; "no detail" is a photograph
        # taken too far away, or a thumbnail, and retaking it is the only fix.
        if metrics.get("sharpness_verdict") == "no_detail":
            reasons.append("no_fine_detail")
        else:
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
        return {"passed": False, "reasons": [load_reason], "metrics": {},
                "fatal": True}

    metrics = _frame_metrics(image)
    reasons = _screen_metrics(metrics)
    return {"passed": not reasons, "reasons": reasons, "metrics": metrics,
            "fatal": is_fatal(reasons)}


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
        if total_frames <= 1:
            # OpenCV happily opens a still image as a one-frame "video", so a
            # JPEG or PNG renamed .mp4 arrives here looking like a very short
            # clip. It is not one, and "too short" sends whoever reads it off
            # to record a longer video, which cannot help. The app's demo mode
            # produces exactly this file when its walking-video asset is
            # missing: it writes the placeholder image out as demo_walking.mp4
            # so the capture flow still completes.
            reasons.append("not_a_video")
        elif total_frames < MIN_FRAMES:
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