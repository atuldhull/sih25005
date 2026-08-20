"""Measurement engine: computes trait measurements from keypoints using pure geometry."""

import math
from typing import Dict, List, Optional, Tuple

from ml.common.schemas import MeasurementResult
from ml.config.traits import get_trait, TRAIT_REGISTRY

# Measurement re-gates independently of the pose stage, so the EFFECTIVE
# threshold is the stricter of the two and lowering only one of them changes
# nothing. Kept equal to POSE_MIN_KEYPOINT_CONFIDENCE deliberately; see the
# measured sweep in ml/config/models.py for why it is 0.10.
KEYPOINT_CONFIDENCE_THRESHOLD = 0.10

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


# The pose model's median localisation error, as a fraction of the animal's
# longer side. Mirrors bovine_pose_infer.MEDIAN_ERR_FRAC; duplicated rather
# than imported so measurement does not depend on the inference module.
KEYPOINT_ERR_FRAC = 0.0156


def _animal_scale(keypoints: Dict[str, Tuple[float, float, float]]) -> float:
    """Size of the animal in pixels, from the spread of its own keypoints.

    The widest gap between any two usable joints. Not the bounding box, which
    measurement is not given - and this is the quantity that matters anyway,
    since keypoint error is quoted relative to the animal, not to the frame.
    """
    live = [(v[0], v[1]) for v in keypoints.values() if v[2] > 0]
    if len(live) < 2:
        return 0.0
    xs = [p[0] for p in live]
    ys = [p[1] for p in live]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def _angle_uncertainty(points: List[Keypoint], scale_px: float,
                       geometry: str) -> Optional[float]:
    """Degrees the angle could be out by, from keypoint error alone.

    An angle is only as good as the segment it is measured across. Two joints
    a long way apart pin a direction tightly; two joints close together barely
    pin it at all, and the error does not shrink with them - it is a fixed
    fraction of the ANIMAL's size, not of the segment's.

    Which segment that is depends on the geometry, and getting it wrong makes
    this useless. The leg-set traits take four points and compare the LEFT and
    RIGHT legs; the direction each contributes comes from its own
    upper-to-lower span, not from the gap between the two sides - which on a
    side-on photograph is a few pixels and would imply an enormous error where
    there is none.

    Measured over 55 photographs, median segment as a fraction of the animal:

        rump_angle      33.1%   ->  +/-4 deg    (its band is 15 deg wide)
        shoulder_angle   4.8%   ->  +/-27 deg   (its band is 20 deg wide)
        foot_angle       3.9%   ->  +/-25 deg   (its band is 25 deg wide)

    The last two are scored on a quantity whose uncertainty covers the whole
    range the score is drawn from, which is not a measurement.
    """
    if scale_px <= 0:
        return None
    err_px = KEYPOINT_ERR_FRAC * scale_px * math.sqrt(2.0)   # two endpoints

    def seg(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    if geometry == "leg_set" and len(points) == 4:
        # [upper_left, upper_right, lower_left, lower_right]: each side's own
        # span is what fixes that leg's direction.
        spans = [seg(points[0], points[2]), seg(points[1], points[3])]
    else:
        pts = _reduce_points(points)
        if len(pts) < 2:
            return None
        # a 3-point angle has two arms; the shorter one limits the result
        spans = [seg(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]

    shortest = min(spans)
    if shortest <= 0:
        return None
    return math.degrees(math.atan2(err_px, shortest))


def _pixel_distance(a: Keypoint, b: Keypoint) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _pixel_vertical_distance(a: Keypoint, b: Keypoint) -> float:
    """Vertical-only (y-axis) pixel distance, for traits whose definition is a
    height rather than a diagonal/straight-line distance (e.g. Stature, Body
    Depth). Using full Euclidean distance for these overstates the true
    height whenever the two keypoints aren't perfectly vertically aligned in
    the image (any camera roll, or the hoof/chest point not sitting directly
    under the withers point).

    NOTE: this does not correct for camera roll/tilt - it only drops the
    horizontal component. True roll correction (rotating both points into a
    level frame before taking the vertical delta) would need a reference
    signal for "level" (e.g. device orientation at capture time, or two
    ground-contact keypoints to establish the image's horizon) that does not
    exist anywhere in this pipeline yet - not implemented, flagged as a
    follow-up rather than guessed at here.
    """
    return abs(a[1] - b[1])


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
        # dx flips sign under a horizontal image mirror; dy (the physical
        # height relationship between the two points) does not. Using
        # abs(dx) keeps the result mirror-invariant instead of flipping
        # from +theta to (180-theta) when the same geometry is mirrored.
        dx = pts[1][0] - pts[0][0]
        dy = pts[1][1] - pts[0][1]
        return math.degrees(math.atan2(dy, abs(dx)))
    return None


def _compute_leg_set_angle(points: List[Keypoint]) -> Optional[float]:
    """Per-leg deviation from vertical, averaged left/right, for leg-set traits.

    Expects four points ordered [upper_left, upper_right, lower_left,
    lower_right] - hip/hock for rear_legs_rear_view, shoulder/hoof for
    fore_leg_set. Each leg's hip->hock vector angle
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


def _compute_vertical_distance(points: List[Keypoint], scale_factor: float) -> Optional[float]:
    """Vertical-only pixel distance between the first two points, scaled into
    cm. See _pixel_vertical_distance for why this differs from
    _compute_distance for height-type traits (Stature, Body Depth)."""
    if len(points) < 2:
        return None
    return _pixel_vertical_distance(points[0], points[1]) * scale_factor



# Physically impossible measurements, by unit. These are NOT "unusual" bounds -
# the ICAR 1-9 scale exists precisely to describe biological extremes, and
# suppressing a real extreme would defeat the whole scorecard. These are the
# limits outside which a value cannot describe a bovine at all.
#
# Why this is needed: a degenerate keypoint pair (two joints predicted at
# almost the same pixel) produces a tiny distance that scales to something
# like a 0.5 cm chest width. Nothing downstream caught that - config/traits.py
# defines no ranges - so it reached scoring as if it were a real measurement.
# Showing a farmer a 0.5 cm chest is exactly what score=null and
# not_scored_reason exist to prevent.
IMPOSSIBLE_OUTSIDE = {
    # 1 cm, not 5: teat length is about 5 cm and teat thickness 2-3 cm, so a
    # 5 cm floor would have refused real udder traits. Caught by
    # test_keypoint_schema's interop test, which is exactly what it is for.
    "cm": (1.0, 300.0),
    "degrees": (-180.0, 180.0),
    "ratio": (0.0, 20.0),
}


def _is_impossible(value, unit):
    """True when a value cannot describe a real animal, so must be refused."""
    if value is None:
        return False
    bounds = IMPOSSIBLE_OUTSIDE.get(unit)
    if bounds is None:
        return False
    lo, hi = bounds
    return not (lo <= float(value) <= hi)


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

    uncertainty = None
    if trait_class == "A":
        geometry = ("leg_set"
                    if trait_id in ("rear_legs_rear_view", "fore_leg_set")
                    else "angle")
        uncertainty = _angle_uncertainty(
            points, _animal_scale(keypoints), geometry)

    if trait_class == "A":
        # fore_leg_set is the FORE-leg analogue of rear_legs_rear_view: the
        # same four-point layout (two upper joints, two lower joints) and the
        # same signed deviation-from-vertical rule band (-8 to +8, against
        # rear_legs_rear_view's -10 to +10). It was falling through to the
        # generic absolute-angle path instead, which returned 84.1 degrees on
        # a real animal - a quantity the rule table cannot score at all, so
        # the trait refused every time. rear_legs_rear_view was the only leg
        # trait that ever scored, and this is why.
        if trait_id in ("rear_legs_rear_view", "fore_leg_set"):
            # Rear Legs Rear View: rear-view cow-hock deviation-from-vertical,
            # per-leg (see _compute_leg_set_angle's docstring). This geometry
            # was previously mis-attached to rear_legs_set; rear_legs_set now
            # uses the generic 3-point side-view hock angle below instead.
            value = _compute_leg_set_angle(points)
        else:
            value = _compute_angle(points)
    elif trait_class == "B":
        value = _compute_ratio(points)
    elif trait_class == "SMAL":
        # SMAL traits (Heart Girth, Body Condition Score) require a 3D mesh
        # fit and are not derivable from a single 2D distance/angle. Refuse
        # honestly instead of falling through to the Class C distance branch,
        # which previously either crashed (TypeError on `distance * None`
        # when unscaled) or silently returned a wrong-shape value (a flat 2D
        # chord standing in for a circumference, or a length standing in for
        # a 1-9 index).
        return MeasurementResult(
            trait_id=trait_id,
            trait_class=trait_class,
            value=None,
            unit=unit,
            confidence=0.0,
            flags=["not_measurable", "requires_3d_model"],
        )
    else:  # Class C
        if trait_id in ("stature", "body_depth"):
            # Both are height measurements per the trait definition, not
            # diagonal/straight-line distances (unlike e.g. Body Length,
            # which is genuinely front-to-back and correctly uses the full
            # 2D distance). See _pixel_vertical_distance for why Euclidean
            # distance overstates height here.
            value = _compute_vertical_distance(points, scale_factor)
        else:
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

    # Refuse rather than report a value that cannot describe an animal. This
    # happens when two keypoints collapse onto nearly the same pixel, or when
    # the scale is wrong by an order of magnitude - both of which produce a
    # confident-looking number rather than an obvious failure.
    if _is_impossible(value, unit):
        return MeasurementResult(
            trait_id=trait_id,
            trait_class=trait_class,
            value=None,
            unit=unit,
            confidence=0.0,
            flags=["not_measurable", "implausible_value"],
        )

    return MeasurementResult(
        trait_id=trait_id,
        trait_class=trait_class,
        value=value,
        unit=unit,
        confidence=confidence,
        flags=[],
        uncertainty=uncertainty,
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