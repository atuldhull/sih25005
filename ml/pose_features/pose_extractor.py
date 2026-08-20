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

# Two DIFFERENT joints predicted at the same pixel are mutually
# invalidating - at most one can be right, and there is no way to tell
# which. Measured on a real photo: knee_left (2158, 2149) and pin_left
# (2160, 2154) landed 5 px apart on a 3066 px animal, as did chest_front
# and hook_left. The angle built on that knee then came out at 74.8 deg
# against a rule table expecting 130-160, so every leg trait refused.
#
# The geometry was right the whole time; the joints were wrong. Nothing
# downstream could have detected that, because a collapsed pair looks
# exactly like a confident measurement.
# Calibrated against the measured separations rather than guessed. On a sharp
# 4623 px animal the pairwise distances between distinct joints fall into two
# clearly separated populations:
#
#     pin_left    <-> knee_left        4.9 px   0.11%   collapsed
#     chest_front <-> hook_left        9.1 px   0.20%   collapsed
#     pin_right   <-> knee_right      12.3 px   0.27%   collapsed
#     ----------------------------------------------------------
#     back_mid    <-> hip_bone_left   34.8 px   0.75%   plausible anatomy
#     back_mid    <-> hip_bone_right  75.9 px   1.64%   plausible anatomy
#     pin_left    <-> knee_right     148.2 px   3.21%   clearly distinct
#
# The gap between 0.27% and 0.75% is where the threshold belongs. An earlier
# value of 2% sat above BOTH populations and discarded nine of nineteen usable
# joints, including the hip bones and the topline - over-pruning until almost
# nothing could be measured, which fails just as badly as measuring garbage,
# only more quietly.
#
# 0.5% is also well inside the model's own error: median 1.56% of the box side,
# PCK@0.02 61.7%. Two joints closer together than a third of the typical error
# are not distinguishable, which is exactly the claim being made here.
COLLAPSE_MIN_SEPARATION_FRAC = 0.005   # of the animal box's longer side

# When two joints do collapse, which one is wrong? Refusing both was the first
# answer, on the grounds that picking the survivor would be a coin flip. It is
# not a coin flip when one of them is far more confident than the other.
#
# Confidence here is sigmoid(heatmap peak) * PCK@0.02 / 0.75, so it carries
# both how sharp THIS prediction was and how reliable the joint is
# historically. A joint at 0.62 sitting on top of one at 0.11 is not a
# symmetric situation: the weak one has almost certainly drifted onto the
# strong one, and discarding the strong one as well destroys a good landmark
# to punish a bad one.
#
# Measured: refusing both meant that loosening the confidence gate made things
# WORSE, not better - usable joints rose 6.3 -> 14.3 per image while measured
# traits FELL 41 -> 34, because every newly admitted weak joint took a good
# one down with it. Keeping the clearly-stronger member removes that
# perversity, and comparable pairs are still both refused, which is the case
# the coin-flip argument was actually about.
COLLAPSE_KEEP_RATIO = 1.5

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
    return _drop_collapsed(out, animal_bbox)


def _same_landmark_other_side(a: str, b: str) -> bool:
    """Is this the SAME landmark on the left and right side?

    On a side-on photograph the near and far legs overlap, so pastern_left
    and pastern_right genuinely land within a few pixels of each other - as do
    the hocks and the hip bones. That is correct geometry, not a model
    failure, and an earlier version of the collapse filter threw all of them
    away: 19 usable joints down to 6.

    Only pairs that are anatomically distinct can invalidate each other.
    """
    for suffix in ("_left", "_right"):
        other = "_right" if suffix == "_left" else "_left"
        if a.endswith(suffix) and b.endswith(other):
            return a[: -len(suffix)] == b[: -len(other)]
    return False


def _drop_collapsed(
    kps: Dict[str, Tuple[float, float, float]],
    animal_bbox: Sequence[float],
) -> Dict[str, Tuple[float, float, float]]:
    """Refuse joints that landed on top of another joint.

    When two distinct landmarks share a pixel the model has not separated
    them, and any measurement spanning either is meaningless while still
    looking confident. A clearly stronger member survives (see
    COLLAPSE_KEEP_RATIO); a pair of comparable confidence is refused entirely,
    because there is then genuinely no way to tell which one moved.
    """
    x1, y1, x2, y2 = (float(v) for v in animal_bbox)
    reach = COLLAPSE_MIN_SEPARATION_FRAC * max(1.0, max(x2 - x1, y2 - y1))
    live = [(n, v) for n, v in kps.items() if v[2] > 0]
    bad = set()
    for i, (na, va) in enumerate(live):
        for nb, vb in live[i + 1:]:
            if _same_landmark_other_side(na, nb):
                continue
            if abs(va[0] - vb[0]) > reach or abs(va[1] - vb[1]) > reach:
                continue
            hi, lo = (va[2], vb[2]) if va[2] >= vb[2] else (vb[2], va[2])
            if lo > 0 and hi >= COLLAPSE_KEEP_RATIO * lo:
                # One is clearly better founded than the other; the weaker has
                # drifted onto it. Keep the stronger.
                bad.add(nb if va[2] >= vb[2] else na)
            else:
                # Comparably confident and in the same place. There is no
                # basis for preferring either, so neither is trusted.
                bad.add(na)
                bad.add(nb)
    if not bad:
        return kps
    return {n: ((v[0], v[1], 0.0) if n in bad else v) for n, v in kps.items()}


def usable_joint_count(
        keypoints: Dict[str, Tuple[float, float, float]]) -> int:
    """How many joints are actually usable. Zero means measurement cannot run
    at all, and the pipeline should say so rather than emit empty traits."""
    return sum(1 for v in keypoints.values() if v[2] > 0.0)


def load_error() -> Optional[str]:
    """The last load failure, so the pipeline can give a truthful reason."""
    return _load_error
