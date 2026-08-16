"""Measurement engine: computes trait measurements from keypoints using pure geometry."""

import math
from typing import Dict, List, Optional, Tuple

from ml.common.schemas import MeasurementResult
from ml.config.traits import get_trait, TRAIT_REGISTRY

KEYPOINT_CONFIDENCE_THRESHOLD = 0.3

Keypoint = Tuple[float, float]


def _keypoint(keypoints: Dict[str, Tuple[float, float, float]], name: str) -> Optional[Keypoint]:
    """Return the (x, y) of a keypoint if present and confidently detected, else None."""
    if name not in keypoints:
        return None
    x, y, confidence = keypoints[name]
    if confidence < KEYPOINT_CONFIDENCE_THRESHOLD:
        return None
    return (x, y)


def _reduce_points(points: List[Keypoint]) -> List[Keypoint]:
    """Average paired keypoints (first half vs second half, e.g. left/right anatomy).

    Repeatedly merges symmetric pairs into midpoints until at most 3 points remain,
    so 4-point angle traits (left+right pairs) collapse to a 2-point line.
    """
    pts = list(points)
    while len(pts) > 3 and len(pts) % 2 == 0:
        half = len(pts) // 2
        pts = [
            ((pts[i][0] + pts[i + half][0]) / 2.0, (pts[i][1] + pts[i + half][1]) / 2.0)
            for i in range(half)
        ]
    return pts


def _pixel_distance(a: Keypoint, b: Keypoint) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _compute_angle(points: List[Keypoint]) -> Optional[float]:
    """Angle in degrees from reduced points.

    With 3 points the middle one is the vertex and the angle between the two
    vectors to the endpoints is computed via arccos(dot / |a||b|). With 2 points
    (after pairing) the line's angle from the horizontal is returned.
    """
    pts = _reduce_points(points)
    if len(pts) == 3:
        vx, vy = pts[1][0], pts[1][1]
        a = (pts[0][0] - vx, pts[0][1] - vy)
        b = (pts[2][0] - vx, pts[2][1] - vy)
        mag_a = math.hypot(*a)
        mag_b = math.hypot(*b)
        if mag_a == 0 or mag_b == 0:
            return None
        cos_theta = (a[0] * b[0] + a[1] * b[1]) / (mag_a * mag_b)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        return math.degrees(math.acos(cos_theta))
    if len(pts) == 2:
        raw_angle = math.degrees(math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0]))
        # Fold into (-90, 90]: a line's orientation is direction-agnostic, so
        # atan2 can return an angle 180 degrees away for the same physical
        # slope depending on point/facing order (e.g. 5.7 vs 174.3 degrees).
        if raw_angle > 90.0:
            raw_angle -= 180.0
        elif raw_angle <= -90.0:
            raw_angle += 180.0
        return raw_angle
def _compute_leg_set_angle(points: List[Keypoint]) -> Optional[float]:
    """Per-leg deviation from vertical, averaged left/right, for leg-set traits.

    Expects points ordered [hip_left, hip_right, hock_left, hock_right], matching
    rear_legs_set's required_keypoints order. Each leg's hip->hock vector angle
    from vertical (0 = straight down) is computed independently. The right leg's
    deviation is mirrored before averaging: the right side is anatomically
    mirrored relative to the left, so without mirroring, a symmetric inward lean
    on both hocks (cow-hocked) cancels to 0 instead of registering as a
    deviation. This differs from the generic _reduce_points() pairing, which
    merges hip_bone_left+hip_bone_right and hock_left+hock_right into two
    midpoints and measures the angle between those (also degenerate: masks
    cow-hocked animals instead of detecting them).
    """
    if len(points) != 4:
        return None
    hip_left, hip_right, hock_left, hock_right = points
    left_dev = math.degrees(math.atan2(hock_left[0] - hip_left[0], hock_left[1] - hip_left[1]))
    right_dev = math.degrees(math.atan2(hock_right[0] - hip_right[0], hock_right[1] - hip_right[1]))
    return (left_dev - right_dev) / 2.0
def _compute_ratio(points: List[Keypoint]) -> Optional[float]:
    """Ratio of the distance between the first point-pair to the second point-pair."""
    if len(points) < 4:
        return None
    d1 = _pixel_distance(points[0], points[1])
    d2 = _pixel_distance(points[2], points[3])
    if d2 == 0:
        return None
    return d1 / d2


def _compute_distance(points: List[Keypoint], scale_factor: float) -> Optional[float]:
    """Pixel distance between the first two points scaled into cm."""
    if len(points) < 2:
        return None
    return _pixel_distance(points[0], points[1]) * scale_factor


def measure_trait(
    trait_id: str,
    keypoints: Dict[str, Tuple[float, float, float]],
    scale_factor: Optional[float] = None,
    scale_confidence: float = 1.0,
) -> MeasurementResult:
    """Measure a single trait from keypoints per its registry definition.

    Class A traits compute an angle, Class B a ratio of distances, and Class C a
    scaled cm distance (requiring scale_factor). Returns a MeasurementResult with
    flags=["not_measurable"] (plus "no_scale" for unscaled Class C) when a required
    keypoint is missing/low-confidence or a degenerate geometry is produced.
    """
    trait = get_trait(trait_id)
    trait_class = trait["trait_class"]
    unit = trait["unit"]

    points: List[Keypoint] = []
    confidences: List[float] = []
    for name in trait["required_keypoints"]:
        pt = _keypoint(keypoints, name)
        if pt is None:
            return MeasurementResult(
                trait_id=trait_id,
                trait_class=trait_class,
                value=None,
                unit=unit,
                confidence=0.0,
                flags=["not_measurable"],
            )
        points.append(pt)
        confidences.append(keypoints[name][2])

    if trait_class == "C" and scale_factor is None:
        return MeasurementResult(
            trait_id=trait_id,
            trait_class=trait_class,
            value=None,
            unit=unit,
            confidence=0.0,
            flags=["not_measurable", "no_scale"],
        )

    if trait_class == "A":
        if trait_id == "rear_legs_set":
            value = _compute_leg_set_angle(points)
        else:
            value = _compute_angle(points)
    elif trait_class == "B":
        value = _compute_ratio(points)
    else:  # Class C
        value = _compute_distance(points, scale_factor)

    if value is None:
        return MeasurementResult(
            trait_id=trait_id,
            trait_class=trait_class,
            value=None,
            unit=unit,
            confidence=0.0,
            flags=["not_measurable"],
        )

    confidence = sum(confidences) / len(confidences)
    if trait_class == "C":
        confidence *= scale_confidence

    return MeasurementResult(
        trait_id=trait_id,
        trait_class=trait_class,
        value=value,
        unit=unit,
        confidence=confidence,
        flags=[],
    )


def measure_all_traits(
    keypoints: Dict[str, Tuple[float, float, float]],
    scale_factor: Optional[float] = None,
    species: str = "cattle",
    scale_confidence: float = 1.0,
) -> List[MeasurementResult]:
    """Measure every trait registered for the given species and return all results."""
    matching = [t["trait_id"] for t in TRAIT_REGISTRY if species in t["species_variants"]]
    return [
        measure_trait(trait_id, keypoints, scale_factor, scale_confidence)
        for trait_id in matching
    ]