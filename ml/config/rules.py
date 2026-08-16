"""Species rule tables mapping trait measurements to 1-9 scores, plus scoring helpers."""

from typing import List, Optional, Tuple

def _bins(min_v: float, max_v: float, reverse: bool = False) -> List[Tuple[float, float, int]]:
    """Generate 9 contiguous bins spanning [min_v, max_v] mapped to scores 1-9.

    Bin 1 always covers the low end of the range and bin 9 the high end. For
    traits where the low end of the range is the biological extreme for score 9
    (e.g. a flat rump is score 9 in some classification schemes), pass reverse=True
    to flip the mapping. Values are placeholders pending expert calibration.
    """
    width = (max_v - min_v) / 9.0
    bins = []
    for i in range(9):
        lo = min_v + i * width
        hi = max_v if i == 8 else min_v + (i + 1) * width
        score = (9 - i) if reverse else (i + 1)
        bins.append((lo, hi, score))
    return bins


SPECIES_RULES = {
    "cattle": {
        # ---------------- Class A: angles (degrees) ----------------
        # NOTE (REVIEW-ml-dev.md, minor fix, resolved): confirmed against ICAR/
        # Nordic conformation-recording reference - low pins (high angle) score
        # higher, high pins (low/negative angle) score 1. reverse=True was
        # inverting this. Removed (default ascending mapping is correct).
        "rump_angle": {"min": 0.0, "max": 15.0, "bins": _bins(0.0, 15.0)},
        "hock_angle": {"min": 130.0, "max": 160.0, "bins": _bins(130.0, 160.0)},
        # RENAMED from pastern_angle -> foot_angle to match contract trait name.
        "foot_angle": {"min": 40.0, "max": 65.0, "bins": _bins(40.0, 65.0)},
        # RENAMED from rear_leg_set -> rear_legs_set to match contract trait name.
        # NOTE (REVIEW-ml-dev.md, Fix #2): the underlying geometry for this trait
        # is degenerate (wrong joint pairing) - fine to keep this rule table as-is,
        # the bug is in measurement/traits.py, not here.
        "rear_legs_set": {"min": -10.0, "max": 10.0, "bins": _bins(-10.0, 10.0)},
        "fore_leg_set": {"min": -8.0, "max": 8.0, "bins": _bins(-8.0, 8.0)},
        "shoulder_angle": {"min": 45.0, "max": 65.0, "bins": _bins(45.0, 65.0)},
        # NEW - no real reference range exists yet. Placeholder only; do not treat
        # any score produced from this as trustworthy until expert-calibrated.
        "angularity": {"min": 10.0, "max": 35.0, "bins": _bins(10.0, 35.0)},
        # ---------------- Class B: ratios (dimensionless) ----------------
        "body_length_to_height_ratio": {"min": 0.90, "max": 1.40, "bins": _bins(0.90, 1.40)},
        "chest_width_to_depth_ratio": {"min": 0.50, "max": 0.80, "bins": _bins(0.50, 0.80)},
        "topline_symmetry": {"min": 0.90, "max": 1.10, "bins": _bins(0.90, 1.10)},
        "pin_width_to_hook_distance_ratio": {"min": 0.40, "max": 0.70, "bins": _bins(0.40, 0.70)},
        "udder_depth_ratio": {"min": 0.0, "max": 0.60, "bins": _bins(0.0, 0.60, reverse=True)},
        "body_width_to_length_ratio": {"min": 0.30, "max": 0.45, "bins": _bins(0.30, 0.45)},
        # NEW placeholders - no real reference ranges yet (require udder/teat
        # keypoints that don't exist in keypoint_schema.py yet).
        "fore_udder_attachment": {"min": 0.0, "max": 1.0, "bins": _bins(0.0, 1.0)},
        "central_ligament": {"min": 0.0, "max": 1.0, "bins": _bins(0.0, 1.0)},
        "front_teat_placement": {"min": 0.0, "max": 1.0, "bins": _bins(0.0, 1.0)},
        "rear_teat_placement": {"min": 0.0, "max": 1.0, "bins": _bins(0.0, 1.0)},
        "rear_legs_rear_view": {"min": 0.0, "max": 1.0, "bins": _bins(0.0, 1.0)},
        # ---------------- Class C: measurements (cm) ----------------
        "heart_girth": {"min": 130.0, "max": 220.0, "bins": _bins(130.0, 220.0)},
        "body_length": {"min": 100.0, "max": 180.0, "bins": _bins(100.0, 180.0)},
        # RENAMED from height_at_withers -> stature to match contract trait name.
        "stature": {"min": 100.0, "max": 170.0, "bins": _bins(100.0, 170.0)},
        "rump_length": {"min": 35.0, "max": 65.0, "bins": _bins(35.0, 65.0)},
        "chest_width": {"min": 30.0, "max": 60.0, "bins": _bins(30.0, 60.0)},
        # chest_depth kept (internal-only trait, same keypoints as body_depth).
        "chest_depth": {"min": 55.0, "max": 95.0, "bins": _bins(55.0, 95.0)},
        # NEW: body_depth is the contract-facing trait with the same underlying
        # measurement as chest_depth above (same keypoints: withers, chest_bottom).
        # NOTE (REVIEW-ml-dev.md, Fix #3): this 2D distance is NOT true heart girth
        # circumference - do not feed body_depth's value into Schaeffer's formula
        # directly for heart_girth. See weight/estimator.py.
        "body_depth": {"min": 55.0, "max": 95.0, "bins": _bins(55.0, 95.0)},
        "rump_width": {"min": 35.0, "max": 65.0, "bins": _bins(35.0, 65.0)},
        "udder_depth": {"min": -10.0, "max": 25.0, "bins": _bins(-10.0, 25.0, reverse=True)},
        # NEW placeholders - no real reference ranges yet.
        "rear_udder_height": {"min": 10.0, "max": 35.0, "bins": _bins(10.0, 35.0)},
        "teat_length": {"min": 3.0, "max": 9.0, "bins": _bins(3.0, 9.0)},
        "rear_udder_width": {"min": 8.0, "max": 22.0, "bins": _bins(8.0, 22.0)},
        "teat_thickness": {"min": 1.5, "max": 4.5, "bins": _bins(1.5, 4.5)},
        # measure_class SMAL in the contract - placeholder range on a 1-9-like scale.
        "body_condition_score": {"min": 1.0, "max": 9.0, "bins": _bins(1.0, 9.0)},
    },
    "buffalo": {
        # ---------------- Class A: angles (degrees) ----------------
       # NOTE (REVIEW-ml-dev.md, minor fix, resolved): same ICAR-direction fix
        # as the cattle rump_angle entry above - reverse=True was inverting the
        # correct ascending mapping (low pins/high angle = higher score).
        "rump_angle": {"min": 0.0, "max": 16.0, "bins": _bins(0.0, 16.0)},
        "hock_angle": {"min": 132.0, "max": 162.0, "bins": _bins(132.0, 162.0)},
        "foot_angle": {"min": 42.0, "max": 67.0, "bins": _bins(42.0, 67.0)},
        "rear_legs_set": {"min": -10.0, "max": 10.0, "bins": _bins(-10.0, 10.0)},
        "fore_leg_set": {"min": -8.0, "max": 8.0, "bins": _bins(-8.0, 8.0)},
        "shoulder_angle": {"min": 45.0, "max": 65.0, "bins": _bins(45.0, 65.0)},
        "angularity": {"min": 10.0, "max": 35.0, "bins": _bins(10.0, 35.0)},
        # ---------------- Class B: ratios (dimensionless) ----------------
        "body_length_to_height_ratio": {"min": 0.90, "max": 1.40, "bins": _bins(0.90, 1.40)},
        "chest_width_to_depth_ratio": {"min": 0.52, "max": 0.82, "bins": _bins(0.52, 0.82)},
        "topline_symmetry": {"min": 0.90, "max": 1.10, "bins": _bins(0.90, 1.10)},
        "pin_width_to_hook_distance_ratio": {"min": 0.40, "max": 0.70, "bins": _bins(0.40, 0.70)},
        "udder_depth_ratio": {"min": 0.0, "max": 0.60, "bins": _bins(0.0, 0.60, reverse=True)},
        "body_width_to_length_ratio": {"min": 0.30, "max": 0.45, "bins": _bins(0.30, 0.45)},
        "fore_udder_attachment": {"min": 0.0, "max": 1.0, "bins": _bins(0.0, 1.0)},
        "central_ligament": {"min": 0.0, "max": 1.0, "bins": _bins(0.0, 1.0)},
        # NOTE: buffalo teat traits are measured on the left REAR teat, not front
        # (REVIEW-ml-dev.md B3). This rule table entry does not encode which
        # physical teat is used - that resolution happens in measurement/traits.py.
        "front_teat_placement": {"min": 0.0, "max": 1.0, "bins": _bins(0.0, 1.0)},
        "rear_teat_placement": {"min": 0.0, "max": 1.0, "bins": _bins(0.0, 1.0)},
        "rear_legs_rear_view": {"min": 0.0, "max": 1.0, "bins": _bins(0.0, 1.0)},
        # ---------------- Class C: measurements (cm; buffalo larger) ----------------
        "heart_girth": {"min": 150.0, "max": 240.0, "bins": _bins(150.0, 240.0)},
        "body_length": {"min": 115.0, "max": 195.0, "bins": _bins(115.0, 195.0)},
        "stature": {"min": 110.0, "max": 180.0, "bins": _bins(110.0, 180.0)},
        "rump_length": {"min": 40.0, "max": 70.0, "bins": _bins(40.0, 70.0)},
        "chest_width": {"min": 32.0, "max": 64.0, "bins": _bins(32.0, 64.0)},
        "chest_depth": {"min": 60.0, "max": 105.0, "bins": _bins(60.0, 105.0)},
        "body_depth": {"min": 60.0, "max": 105.0, "bins": _bins(60.0, 105.0)},
        "rump_width": {"min": 40.0, "max": 70.0, "bins": _bins(40.0, 70.0)},
        "udder_depth": {"min": -10.0, "max": 25.0, "bins": _bins(-10.0, 25.0, reverse=True)},
        "rear_udder_height": {"min": 12.0, "max": 38.0, "bins": _bins(12.0, 38.0)},
        "teat_length": {"min": 3.5, "max": 10.0, "bins": _bins(3.5, 10.0)},
        "rear_udder_width": {"min": 9.0, "max": 24.0, "bins": _bins(9.0, 24.0)},
        "teat_thickness": {"min": 1.8, "max": 5.0, "bins": _bins(1.8, 5.0)},
        "body_condition_score": {"min": 1.0, "max": 9.0, "bins": _bins(1.0, 9.0)},
    },
}


def score_from_value(trait_id: str, species: str, value: float) -> Tuple[Optional[int], float]:
    """Map a trait measurement to a (score_1_9, confidence) for the given species.

    Confidence is highest (~0.95) when the value sits in the middle of a bin and
    lowest (~0.45) at bin boundaries.

    NOTE (REVIEW-ml-dev.md, Important Fix #4, resolved): out-of-range values used
    to CLAMP to the nearest extreme score (1 or 9) at confidence 0.3, producing a
    confident-looking score from a garbage measurement. Now returns (None, 0.0)
    instead, so the caller (score_trait) can refuse to score the trait - matching
    the "refusing to score is a feature" pitch point and the contract's
    not_scored_reason mechanism.
    """
    if species not in SPECIES_RULES:
        raise KeyError(f"Unknown species: {species!r}")
    species_rules = SPECIES_RULES[species]
    if trait_id not in species_rules:
        raise KeyError(f"Unknown trait_id {trait_id!r} for species {species!r}")
    rule = species_rules[trait_id]
    bins = rule["bins"]

    if value < rule["min"] or value > rule["max"]:
        return (None, 0.0)

    for lo, hi, score in bins:
        if lo <= value <= hi:
            width = hi - lo
            center = (lo + hi) / 2.0
            offset = abs(value - center) / (width / 2.0) if width > 0 else 0.0
            confidence = round(0.95 - 0.50 * offset, 3)
            confidence = max(confidence, 0.45)
            return (score, confidence)

    raise ValueError(f"No bin matches value {value} for trait {trait_id!r} ({species})")