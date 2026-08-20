"""Stage 4a: bovine keypoints.

Backs onto bovine_pose_infer.BovinePoseModel - an HRNet-W32 top-down heatmap
model at 384px, NOT RTMPose. mmcv/mmpose do not install on Python 3.13
(pkgutil.ImpImporter was removed in 3.12), so the model is plain torch+timm.
Same top-down heatmap method, different library; the output contract below is
unchanged, which is the part that matters to everything downstream.

WHAT THIS RETURNS
    {joint_name: (x, y, confidence)}  in FULL-IMAGE pixels

which is exactly the shape measure_all_traits() consumes. Joint names are the
canonical KEYPOINT_SCHEMA names, so nothing downstream needs remapping.

WHAT IT REFUSES
Only 22 of the 41 canonical joints were ever annotated - the udder and teat
landmarks were not. Those return confidence 0.0 rather than a guessed
position, so measurement treats them as unmeasurable and the affected traits
carry not_scored_reason instead of a fabricated number. That refusal is the
reason confidence is in the tuple at all.

Measured on a held-out split: PCK@0.05 89.9%, PCK@0.02 61.7%, median error
1.56% of the bounding-box side (roughly 2.4 cm on an adult animal). Per-joint
figures live in bovine_pose_infer.PCK02 and set each landmark's confidence, so
a historically weak joint cannot report high confidence on one lucky frame.
"""
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from ml.config.models import POSE_MIN_KEYPOINT_CONFIDENCE, POSE_MODEL_PATH

_model = None
_load_error: Optional[str] = None


class PoseBackendError(RuntimeError):
    """The pose model could not be loaded or run."""


def _get_model(model_path: Optional[str] = None):
    """Load once and cache. Kept lazy so importing ml.pipeline stays cheap -
    the server imports it on a background thread and must not pay for torch."""
    global _model, _load_error
    if _model is not None:
        return _model
    path = Path(model_path or POSE_MODEL_PATH)
    if not path.exists():
        _load_error = (
            f"pose checkpoint not found at {path}. Weights are not in git - "
            f"run scripts/fetch_models.py, see models/README.md")
        raise PoseBackendError(_load_error)
    try:
        from ml.pose_features.bovine_pose_infer import BovinePoseModel
        _model = BovinePoseModel(str(path))
    except Exception as exc:
        _load_error = f"pose model failed to load: {exc}"
        raise PoseBackendError(_load_error) from exc
    return _model


def extract_keypoints(
    image_path: str,
    animal_bbox: Sequence[float],
    model_path: Optional[str] = None,
    min_confidence: float = POSE_MIN_KEYPOINT_CONFIDENCE,
) -> Dict[str, Tuple[float, float, float]]:
    """Locate the canonical joints for the animal in `animal_bbox`.

    animal_bbox is (x1, y1, x2, y2) in FULL-IMAGE pixels - the same frame
    detector.detect_animal returns, so nothing is converted in between. A
    coordinate-frame mismatch here would silently shift every measurement,
    which is why ml/chain_test.py asserts the frames agree rather than
    trusting that they do.
    """
    model = _get_model(model_path)
    try:
        kps = model.extract(image_path, animal_bbox)
    except Exception as exc:
        raise PoseBackendError(f"pose inference failed: {exc}") from exc

    out: Dict[str, Tuple[float, float, float]] = {}
    for name, kp in kps.items():
        conf = float(kp.confidence)
        if conf < min_confidence:
            # Report the joint as unusable rather than dropping the key.
            # Measurement looks joints up by name, so a missing key would be
            # a KeyError; a zero confidence is a refusal it already handles.
            out[name] = (float(kp.x), float(kp.y), 0.0)
        else:
            out[name] = (float(kp.x), float(kp.y), conf)
    return out


def usable_joint_count(
        keypoints: Dict[str, Tuple[float, float, float]]) -> int:
    """How many joints are actually usable. Zero means measurement cannot run
    at all, and the pipeline should say so rather than emit empty traits."""
    return sum(1 for v in keypoints.values() if v[2] > 0.0)


def load_error() -> Optional[str]:
    """The last load failure, so the pipeline can give a truthful reason."""
    return _load_error
