"""Detect animals and ear tags using RT-DETRv2 (HuggingFace transformers), with SAM2
foreground segmentation.

Inference wiring only - no training code. Loads a fine-tuned checkpoint (exported in
HuggingFace format) from config/detection.py. Both backends degrade gracefully:
RT-DETRv2 unavailability raises DetectionBackendError, while SAM2 unavailability
falls back to a rectangular mask (segmentation_degraded=True) instead of crashing.

CHANGED (REVIEW-ml-dev.md B4): the Ultralytics/YOLO backend has been removed
entirely. Ultralytics is AGPL-licensed, which conflicts with this project's
licensing stance (see architecture doc Section 16: "RT-DETRv2 over YOLO/Ultralytics
- matches the project's license constraints"). This module now loads RT-DETRv2
exclusively via HuggingFace `transformers` (Apache-2.0), which is license-safe.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from common.schemas import DetectionResult
from config.detection import (
    CANONICAL_CLASS_NAMES,
    DEFAULT_DEVICE,
    DEBUG_DIR,
    RAW_LABEL_TO_CLASS_NAME,
    RTDETR_CONFIDENCE_THRESHOLD,
    RTDETR_MODEL_PATH,
    SAM2_CONFIG_PATH,
    SAM2_MODEL_PATH,
)


class DetectionBackendError(RuntimeError):
    """Raised when a required inference model/weights cannot be loaded or run."""


class DetectionLabelError(ValueError):
    """Raised when a raw model label cannot be mapped to a canonical class name."""


_rt_detr_model: Any = None
_rt_detr_processor: Any = None
_sam2_predictor: Any = None


def validate_label_map() -> None:
    """Fail loudly if the configured label map references anything non-canonical.

    This is the guard against a checkpoint whose native labels don't match
    "animal"/"ear_tag" - a mismatched config must error, not mislabel.
    """
    bad_values = {
        raw: cls for raw, cls in RAW_LABEL_TO_CLASS_NAME.items()
        if cls not in CANONICAL_CLASS_NAMES
    }
    if bad_values:
        raise DetectionLabelError(
            "Label map maps raw labels to non-canonical class names: "
            f"{bad_values}. Valid targets are {CANONICAL_CLASS_NAMES}."
        )


def map_class_name(raw_label: Any) -> str:
    """Map a checkpoint's raw label (int index) to a canonical class name.

    Raises DetectionLabelError if the label is not in the configured map - a
    loud failure intended to surface checkpoint/training mismatches early.
    """
    validate_label_map()
    if raw_label not in RAW_LABEL_TO_CLASS_NAME:
        raise DetectionLabelError(
            f"Unrecognized raw detector label {raw_label!r}. Expected one of "
            f"{sorted(RAW_LABEL_TO_CLASS_NAME, key=str)}. The checkpoint's native "
            "labels likely don't match this deployment's label map."
        )
    return RAW_LABEL_TO_CLASS_NAME[raw_label]


def resolve_device(device: str) -> str:
    """Resolve a device request to 'cuda' or 'cpu', with an 'auto' CPU fallback."""
    if device and device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ---------------------------------------------------------------------------
# RT-DETRv2 (HuggingFace transformers)
# ---------------------------------------------------------------------------
def load_rt_detr(device: str = DEFAULT_DEVICE) -> Tuple[Any, Any]:
    """Lazily load and cache the RT-DETRv2 model + image processor.

    Returns (model, image_processor). RTDETR_MODEL_PATH must be a directory in
    HuggingFace format (config.json + weights + preprocessor_config.json) -
    see config/detection.py for the expected export procedure.
    """
    global _rt_detr_model, _rt_detr_processor
    if _rt_detr_model is not None:
        return _rt_detr_model, _rt_detr_processor

    resolved = resolve_device(device)

    try:
        from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection
    except ImportError as exc:
        raise DetectionBackendError(
            "'transformers' is not installed. Install it with `pip install "
            "transformers` to use the RT-DETRv2 detection backend."
        ) from exc

    if not os.path.isdir(RTDETR_MODEL_PATH):
        raise DetectionBackendError(
            f"RT-DETRv2 model directory not found at {RTDETR_MODEL_PATH}. Expected "
            "a HuggingFace-format export (config.json + weights + "
            "preprocessor_config.json), not a single checkpoint file."
        )

    try:
        model = RTDetrV2ForObjectDetection.from_pretrained(RTDETR_MODEL_PATH)
        processor = AutoImageProcessor.from_pretrained(RTDETR_MODEL_PATH)
        model.to(resolved)
        model.eval()
    except Exception as exc:
        raise DetectionBackendError(
            f"Failed to load RT-DETRv2 via transformers: {exc}"
        ) from exc

    _rt_detr_model = model
    _rt_detr_processor = processor
    return _rt_detr_model, _rt_detr_processor


def _run_detector(
    image_bgr: np.ndarray,
    model_and_processor: Tuple[Any, Any],
    threshold: float = RTDETR_CONFIDENCE_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Run RT-DETRv2 over a BGR image (as loaded by cv2.imread) and return
    normalized detections: [{label: int, score: float, bbox: (x1,y1,x2,y2)}].
    """
    import torch

    model, processor = model_and_processor

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    try:
        inputs = processor(images=pil_image, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # target_sizes expects (height, width) per image.
        target_sizes = torch.tensor([pil_image.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold
        )[0]
    except Exception as exc:
        raise DetectionBackendError(f"RT-DETRv2 inference failed: {exc}") from exc

    detections: List[Dict[str, Any]] = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detections.append(
            {
                "label": int(label.item()),
                "score": float(score.item()),
                "bbox": _pad_bbox(box.detach().cpu().numpy()),
            }
        )
    return detections


def _pad_bbox(box) -> Tuple[float, float, float, float]:
    """Coerce a raw box row to a 4-float (x1, y1, x2, y2) tuple."""
    vals = np.asarray(box).flatten()
    if vals.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]))


def detect_animal(image_path: str, device: str = DEFAULT_DEVICE) -> Optional[DetectionResult]:
    """Detect the best-animal box in an image, or return None if none is found."""
    validate_label_map()
    image = cv2.imread(image_path)
    if image is None:
        return None

    model_and_processor = load_rt_detr(device)
    detections = _run_detector(image, model_and_processor)
    candidates = [
        d for d in detections
        if map_class_name(d["label"]) == "animal" and d["score"] >= RTDETR_CONFIDENCE_THRESHOLD
    ]
    if not candidates:
        return None
    best = max(candidates, key=lambda d: d["score"])
    return DetectionResult(bbox=best["bbox"], confidence=best["score"], class_name="animal")


def detect_ear_tag(
    image_path: str,
    animal_bbox: Optional[Tuple[float, float, float, float]],
    device: str = DEFAULT_DEVICE,
) -> Optional[DetectionResult]:
    """Detect an ear tag within the animal box, or return None if none is found.

    Searching within the animal bbox constrains the tag model to the relevant
    image region (tags are small relative to the full frame).
    """
    validate_label_map()
    image = cv2.imread(image_path)
    if image is None or animal_bbox is None:
        return None

    x1, y1, x2, y2 = [int(round(v)) for v in animal_bbox]
    x1, x2 = max(0, x1), min(image.shape[1], x2)
    y1, y2 = max(0, y1), min(image.shape[0], y2)
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    crop = image[y1:y2, x1:x2]

    model_and_processor = load_rt_detr(device)
    detections = _run_detector(crop, model_and_processor)
    candidates = [
        d for d in detections
        if map_class_name(d["label"]) == "ear_tag" and d["score"] >= RTDETR_CONFIDENCE_THRESHOLD
    ]
    if not candidates:
        return None
    best = max(candidates, key=lambda d: d["score"])
    bx1, by1, bx2, by2 = best["bbox"]
    return DetectionResult(
        bbox=(bx1 + x1, by1 + y1, bx2 + x1, by2 + y1),
        confidence=best["score"],
        class_name="ear_tag",
    )


# ---------------------------------------------------------------------------
# SAM2 segmentation (unchanged from original - not part of B4)
# ---------------------------------------------------------------------------
def load_sam2(device: str = DEFAULT_DEVICE) -> Any:
    """Lazily load and cache the SAM2 image predictor, raising if unavailable."""
    global _sam2_predictor
    if _sam2_predictor is not None:
        return _sam2_predictor

    if not os.path.exists(SAM2_MODEL_PATH) or not os.path.exists(SAM2_CONFIG_PATH):
        raise DetectionBackendError(
            f"SAM2 weights/config not found: {SAM2_MODEL_PATH} / {SAM2_CONFIG_PATH}. "
            "Segmentation will fall back to the RT-DETR box rect."
        )
    try:
        from sam2.build_sam import build_sam2  # type: ignore
    except ImportError as exc:
        raise DetectionBackendError(
            "SAM2 package ('sam2') not installed. Segmentation will fall back "
            "to the RT-DETR box rect."
        ) from exc

    try:
        predictor = build_sam2(SAM2_CONFIG_PATH, SAM2_MODEL_PATH, device=resolve_device(device))
    except Exception as exc:
        raise DetectionBackendError(f"Failed to load SAM2: {exc}") from exc
    _sam2_predictor = predictor
    return _sam2_predictor


def _bbox_mask(shape: Tuple[int, int], bbox: Tuple[float, float, float, float]) -> np.ndarray:
    """Build a rectangular binary mask (0/255 uint8) from an xyxy bbox."""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    mask[y1:y2, x1:x2] = 255
    return mask


def segment_animal(
    image_path: str,
    bbox: Tuple[float, float, float, float],
    device: str = DEFAULT_DEVICE,
    debug_dir: str = DEBUG_DIR,
) -> Tuple[np.ndarray, bool]:
    """Segment the foreground animal from its background.

    Returns (binary mask 0/255 uint8, segmentation_degraded). When SAM2 weights
    are unavailable (DetectionBackendError) the mask falls back to the RT-DETR
    box as a rectangle and segmentation_degraded is True. A debug masked-blur
    image is written under debug_dir.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise DetectionBackendError(f"Cannot read image for segmentation: {image_path}")
    h, w = image.shape[:2]

    try:
        model = load_sam2(device)
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

        predictor = SAM2ImagePredictor(model)
        predictor.set_image(image)
        box_arr = np.array([list(bbox)], dtype=np.float32)
        masks, _, _ = predictor.predict(box=box_arr, multimask_output=False)
        mask = (np.asarray(masks[0]) > 0.5).astype(np.uint8) * 255
        degraded = False
    except DetectionBackendError:
        mask = _bbox_mask((h, w), bbox)
        degraded = True
    except Exception:
        mask = _bbox_mask((h, w), bbox)
        degraded = True

    _write_debug_images(image, mask, bbox, image_path, debug_dir)
    return mask, degraded


def _write_debug_images(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: Tuple[float, float, float, float],
    image_path: str,
    debug_dir: str,
) -> str:
    """Blur everything outside the mask and save it, plus the raw mask, for debugging."""
    os.makedirs(debug_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]

    mask3 = cv2.merge([mask, mask, mask])
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=31.0)
    masked_img = np.where(mask3 > 0, image, blurred)
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    cv2.rectangle(masked_img, (x1, y1), (x2, y2), (0, 0, 255), 3)

    out_path = os.path.join(debug_dir, f"{stem}_masked.jpg")
    mask_path = os.path.join(debug_dir, f"{stem}_mask.png")
    cv2.imwrite(out_path, masked_img)
    cv2.imwrite(mask_path, mask)
    return out_path


__all__ = [
    "DetectionBackendError",
    "DetectionLabelError",
    "detect_animal",
    "detect_ear_tag",
    "load_rt_detr",
    "load_sam2",
    "map_class_name",
    "segment_animal",
    "validate_label_map",
]