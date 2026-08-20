"""Canonical 41-point keypoint schema and the pose-model normalization seam.

The measurement engine (measurement/traits.py + config/traits.py) indexes animal
keypoints by fixed anatomical names (e.g. "withers", "hook_left", "pastern_left").
No off-the-shelf RTMPose checkpoint emits these exact names — pretrained skeletons
use indexed or differently-named joints (COCO-animal, AP-10K, bovine sets). This
module is the single seam that maps a pose model's raw output onto the canonical
skeleton before measurement runs.

KEYPOINT_SCHEMA is the project's canonical skeleton order (41 joints, indices
0-40: the original 24 body/leg joints plus 17 udder/teat joints required by the
9 contract udder traits). Whatever checkpoint is chosen later, normalize_keypoints()
must translate its native output into this schema, and any joint the source cannot
resolve maps to KeypointResult(0, 0, 0.0) so measurement's 0.3-confidence filter
marks the dependent traits not_measurable (fail-local, never a crash).
"""

from typing import Dict, List, NamedTuple, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# Canonical keypoint result + registry
# ---------------------------------------------------------------------------

Numeric = Union[float, int]


class KeypointResult(NamedTuple):
    """A single (x, y, confidence) joint in the canonical skeleton.

    A 3-field NamedTuple so it unpacks exactly like the (x, y, confidence)
    tuples measurement/traits.py already consumes.
    """

    x: float
    y: float
    confidence: float


MISSING_JOINT = KeypointResult(0.0, 0.0, 0.0)

# Stable (name -> index) contract. The index is deliberately fixed: it is the
# order the training dataset must be annotated in for the next phase.
KEYPOINT_SCHEMA: List[Dict[str, object]] = [
    {"keypoint_id": "withers", "index": 0},
    {"keypoint_id": "back_mid", "index": 1},
    {"keypoint_id": "chest_front", "index": 2},
    {"keypoint_id": "chest_bottom", "index": 3},
    {"keypoint_id": "chest_width_left", "index": 4},
    {"keypoint_id": "chest_width_right", "index": 5},
    {"keypoint_id": "shoulder_left", "index": 6},
    {"keypoint_id": "shoulder_right", "index": 7},
    {"keypoint_id": "tail_head", "index": 8},
    {"keypoint_id": "hip_bone_left", "index": 9},
    {"keypoint_id": "hip_bone_right", "index": 10},
    {"keypoint_id": "hook_left", "index": 11},
    {"keypoint_id": "hook_right", "index": 12},
    {"keypoint_id": "pin_left", "index": 13},
    {"keypoint_id": "pin_right", "index": 14},
    {"keypoint_id": "knee_left", "index": 15},
    {"keypoint_id": "knee_right", "index": 16},
    {"keypoint_id": "hock_left", "index": 17},
    {"keypoint_id": "hock_right", "index": 18},
    {"keypoint_id": "pastern_left", "index": 19},
    {"keypoint_id": "pastern_right", "index": 20},
    {"keypoint_id": "hoof_left", "index": 21},
    {"keypoint_id": "hoof_right", "index": 22},
    {"keypoint_id": "rear_udder", "index": 23},
    {"keypoint_id": "fore_udder_top", "index": 24},
    {"keypoint_id": "fore_udder_body_junction", "index": 25},
    {"keypoint_id": "vulva_base", "index": 26},
    {"keypoint_id": "rear_udder_top", "index": 27},
    {"keypoint_id": "udder_cleft_top", "index": 28},
    {"keypoint_id": "udder_cleft_bottom", "index": 29},
    {"keypoint_id": "udder_floor", "index": 30},
    {"keypoint_id": "teat_front_left", "index": 31},
    {"keypoint_id": "teat_front_right", "index": 32},
    {"keypoint_id": "teat_front_left_top", "index": 33},
    {"keypoint_id": "teat_front_left_bottom", "index": 34},
    {"keypoint_id": "teat_rear_left", "index": 35},
    {"keypoint_id": "teat_rear_right", "index": 36},
    {"keypoint_id": "rear_udder_left", "index": 37},
    {"keypoint_id": "rear_udder_right", "index": 38},
    {"keypoint_id": "teat_width_left_1", "index": 39},
    {"keypoint_id": "teat_width_left_2", "index": 40},
]

KEYPOINT_NAMES: List[str] = [str(e["keypoint_id"]) for e in KEYPOINT_SCHEMA]
KEYPOINT_NAME_BY_INDEX: Dict[int, str] = {int(e["index"]): str(e["keypoint_id"]) for e in KEYPOINT_SCHEMA}
KEYPOINT_INDEX_BY_NAME: Dict[str, int] = {str(e["keypoint_id"]): int(e["index"]) for e in KEYPOINT_SCHEMA}

# source_format values that normalize_keypoints() understands. Only "identity"
# is wired today; the RTMPose formats are acknowledged seams to implement once
# a checkpoint is chosen.
VALID_SOURCE_FORMATS = ("identity", "rtmpose_coco_animal", "rtmpose_custom_bovine")


def get_keypoint_index(keypoint_id: str) -> int:
    """Return the canonical index for an anatomical keypoint name."""
    if keypoint_id not in KEYPOINT_INDEX_BY_NAME:
        raise KeyError(f"Unknown keypoint {keypoint_id!r}. Canonical names: {KEYPOINT_NAMES}")
    return KEYPOINT_INDEX_BY_NAME[keypoint_id]


def get_keypoint_name(index: int) -> str:
    """Return the canonical anatomical name for an index (0-40)."""
    if index not in KEYPOINT_NAME_BY_INDEX:
        raise KeyError(f"Unknown keypoint index {index!r}. Valid indices: 0-{len(KEYPOINT_SCHEMA) - 1}")
    return KEYPOINT_NAME_BY_INDEX[index]


# ---------------------------------------------------------------------------
# Normalization seam
# ---------------------------------------------------------------------------

def normalize_keypoints(
    raw_keypoints: object,
    source_format: str = "identity",
) -> Dict[str, KeypointResult]:
    """Map raw pose-model output onto the canonical 41-point keypoint dict.

    Returns a dict keyed by the canonical anatomical names ready to hand to
    measurement/traits.py exactly as the simulated fixture data does today.
    Any joint the source model does not resolve is set to KeypointResult(0, 0, 0.0)
    so measurement's 0.3-confidence filter marks dependent traits not_measurable
    (fail-local, never a crash).

    source_format="identity" is a passthrough for input already keyed by the
    canonical names (e.g. the simulated-keypoint test fixtures). The RTMPose
    formats are intentionally unimplemented and raise NotImplementedError.
    """
    if source_format not in VALID_SOURCE_FORMATS:
        raise ValueError(
            f"Unknown source_format {source_format!r}. Valid formats: {sorted(VALID_SOURCE_FORMATS)}"
        )
    if source_format != "identity":
        raise NotImplementedError(
            f"source_format={source_format!r} is not implemented yet. Once an RTMPose "
            "checkpoint is chosen, teach normalize_keypoints() to map its native joint "
            "order/names onto KEYPOINT_SCHEMA (indices 0-40); unresolved joints should "
            "become confidence=0.0 joints, not be omitted."
        )

    if not isinstance(raw_keypoints, dict):
        raise ValueError(
            f"identity passthrough expects a dict keyed by canonical keypoint names, "
            f"got {type(raw_keypoints).__name__}."
        )

    result: Dict[str, KeypointResult] = {name: MISSING_JOINT for name in KEYPOINT_NAMES}
    for name, value in raw_keypoints.items():
        if name not in KEYPOINT_INDEX_BY_NAME:
            raise ValueError(
                f"Unknown keypoint name {name!r} in identity input. Canonical names: {KEYPOINT_NAMES}"
            )
        result[name] = _coerce_joint(value, name)
    return result


def _coerce_joint(value: object, name: str) -> KeypointResult:
    """Coerce a raw joint value to KeypointResult, validating its (x, y, confidence) shape."""
    try:
        x, y, confidence = value[0], value[1], value[2]
        return KeypointResult(float(x), float(y), float(confidence))
    except (TypeError, IndexError, ValueError, AttributeError) as exc:
        raise ValueError(
            f"Keypoint {name!r} value must be a 3-element numeric (x, y, confidence), "
            f"got {value!r}."
        ) from exc


__all__ = [
    "KEYPOINT_SCHEMA",
    "KEYPOINT_NAMES",
    "KEYPOINT_NAME_BY_INDEX",
    "KEYPOINT_INDEX_BY_NAME",
    "VALID_SOURCE_FORMATS",
    "KeypointResult",
    "MISSING_JOINT",
    "get_keypoint_index",
    "get_keypoint_name",
    "normalize_keypoints",
]