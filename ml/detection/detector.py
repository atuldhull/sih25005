"""Detect animals and ear tags using RT-DETR, with SAM2 foreground segmentation.

Inference wiring only — no training code. Loads pretrained/fine-tuned checkpoints
from config/detection.py. Both backends degrade gracefully: RT-DETR unavailability
raises DetectionBackendError, while SAM2 unavailability falls back to a rectangular
mask (segmentation_degraded=True) instead of crashing.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

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
_sam2_predictor: Any = None


def validate_label_map() -> None:
    """Fail loudly if the configured label map references anything non-canonical.

    This is the guard against a checkpoint whose native labels don't match
    "animal"/"ear_tag" — a mismatched config must error, not mislabel.
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
    """Map a checkpoint's raw label (int index or string) to a canonical class name.

    Raises DetectionLabelError if the label is not in the configured map — a
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
# RT-DETR
# ---------------------------------------------------------------------------
def load_rt_detr(device: str = DEFAULT_DEVICE) -> Any:
    """Lazily load and cache the RT-DETR model (ultralytics or rtdetr-pytorch)."""
    global _rt_detr_model
    if _rt_detr_model is not None:
        return _rt_detr_model

    resolved = resolve_device(device)

    # Preferred backend: ultralytics YOLO wrapper (loads RT-DETRv2 checkpoints).
    try:
        from ultralytics import YOLO
    except ImportError:
        YOLO = None

    if YOLO is not None:
        if not os.path.exists(RTDETR_MODEL_PATH):
            raise DetectionBackendError(
                f"RT-DETR weights not found at {RTDETR_MODEL_PATH}."
            )
        try:
            model = YOLO(RTDETR_MODEL_PATH)
            model.to(device=resolved) if resolved == "cuda" else model.cpu()
            _rt_detr_model = model
            return _rt_detr_model
        except Exception as exc:
            raise DetectionBackendError(f"Failed to load RT-DETR via ultralytics: {exc}") from exc

    # Fallback backend: lyuwenyu/RT-DETR (rtdetr-pytorch). Model variant and
    # post-processor are configured from the same weights path when available.
    try:
        from rtdetr.models import build_model  # type: ignore
        from rtdetr.config import get_cfg  # type: ignore
    except ImportError as exc:
        raise DetectionBackendError(
            "Neither 'ultralytics' nor 'rtdetr-pytorch' is installed, and "
            f"RT-DETR weights ({RTDETR_MODEL_PATH}) were not found."
        ) from exc

    raise DetectionBackendError(
        "RT-DETRv2 backend configured but requires explicit model/config wiring; "
        "install 'ultralytics' and point RTDETR_MODEL_PATH at a fine-tuned checkpoint."
    )


def _parse_predictions(preds: Any, names: Optional[Dict[int, str]] = None) -> List[Dict[str, Any]]:
    """Normalize raw inference output into [{label, score, bbox(0-indexed xyxy)}]."""
    detections: List[Dict[str, Any]] = []
    if not isinstance(preds, (list, tuple)):
        preds = [preds]
    for result in preds:
        if result is None:
            continue
        if isinstance(result, dict):
            # Already-normalized prediction dicts are passed through directly.
            if "bbox" in result and "label" in result and "score" in result:
                detections.append({
                    "label": result["label"],
                    "score": float(result["score"]),
                    "bbox": _pad_bbox(result["bbox"]),
                })
                continue
            boxes = result.get("boxes") or result.get("box")
            scores = result.get("scores") or result.get("score")
            labels = result.get("labels") or result.get("label")
            if boxes is None:
                continue
            boxes_np = _to_numpy(boxes).reshape(-1, 4)
            labels_arr = _to_numpy(labels) if labels is not None else np.zeros(len(boxes_np))
            scores_arr = _to_numpy(scores) if scores is not None else np.ones(len(boxes_np))
            for box, label, score in zip(boxes_np, labels_arr, scores_arr):
                detections.append({
                    "label": label.item() if hasattr(label, "item") else label,
                    "score": float(score),
                    "bbox": _pad_bbox(box),
                })
        elif hasattr(result, "boxes") and hasattr(result, "names"):
            result_names = getattr(result, "names", {}) or {}
            boxes = result.boxes
            xyxy = getattr(boxes, "xyxy", None)
            conf = getattr(boxes, "conf", None)
            cls = getattr(boxes, "cls", None)
            if xyxy is None:
                continue
            for row, conf_row, cls_row in zip(
                _to_numpy(xyxy), _to_numpy(conf), _to_numpy(cls)
            ):
                cls_id = int(cls_row)
                detections.append({
                    "label": result_names.get(cls_id, cls_id),
                    "score": float(conf_row),
                    "bbox": _pad_bbox(row),
                })
        elif isinstance(result, (list, tuple)) and len(result) >= 3:
            # rtdetr style: (scores, labels, boxes)
            scores, labels, boxes = result[0], result[1], result[2]
            boxes_np = _to_numpy(boxes).reshape(-1, 4)
            labels_np = _to_numpy(labels)
            scores_np = _to_numpy(scores)
            for i, box in enumerate(boxes_np):
                detections.append({
                    "label": int(labels_np[i]),
                    "score": float(scores_np[i]),
                    "bbox": _pad_bbox(box),
                })
    if names is not None:
        for d in detections:
            d["label"] = names.get(d["label"], d["label"])
    return detections


def _pad_bbox(box) -> Tuple[float, float, float, float]:
    """Coerce a raw box row to a 4-float (x1, y1, x2, y2) tuple."""
    vals = np.asarray(box).flatten()
    if vals.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]))


def _to_numpy(obj) -> np.ndarray:
    """Coerce tensors/lists/arrays to a numpy array, detaching to CPU first."""
    if hasattr(obj, "detach"):
        obj = obj.detach()
    if hasattr(obj, "cpu"):
        obj = obj.cpu()
    return np.asarray(obj)


def _run_detector(image_bgr: np.ndarray, model: Any) -> List[Dict[str, Any]]:
    """Run a loaded detector over an image and normalize its output."""
    try:
        if hasattr(model, "predict"):
            raw = model.predict(image_bgr, verbose=False)
        else:
            raw = model(image_bgr)
    except Exception as exc:
        raise DetectionBackendError(f"RT-DETR inference failed: {exc}") from exc
    return _parse_predictions(raw)


def detect_animal(image_path: str, device: str = DEFAULT_DEVICE) -> Optional[DetectionResult]:
    """Detect the best-animal box in an image, or return None if none is found."""
    validate_label_map()
    image = cv2.imread(image_path)
    if image is None:
        return None

    model = load_rt_detr(device)
    detections = _run_detector(image, model)
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

    model = load_rt_detr(device)
    detections = _parse_predictions(_predict_crop(model, crop))
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


def _predict_crop(model: Any, crop: np.ndarray) -> Any:
    """Run the detector over a cropped region, normalizing the call surface."""
    try:
        if hasattr(model, "predict"):
            return model.predict(crop, verbose=False)
        return model(crop)
    except Exception as exc:
        raise DetectionBackendError(f"RT-DETR cropped inference failed: {exc}") from exc


# ---------------------------------------------------------------------------
# SAM2 segmentation
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