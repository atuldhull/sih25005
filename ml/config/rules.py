"""Species rule tables mapping trait measurements to 1-9 scores, plus scoring helpers."""

from typing import List, Tuple


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
        # Very sloped = 1, very flat = 9; breed-typical near 4-6.
        "rump_angle": {"min": 0.0, "max": 15.0, "bins": _bins(0.0, 15.0, reverse=True)},
        # Very sickle = 1, very straight = 9; typical ~144-151.
        "hock_angle": {"min": 130.0, "max": 160.0, "bins": _bins(130.0, 160.0)},
        # Weak/curved pastern = 1, steep = 9; typical ~45-55.
        "pastern_angle": {"min": 40.0, "max": 65.0, "bins": _bins(40.0, 65.0)},
        # Cow-hocked (inward) = 1, bandy (outward) = 9; straight (0) = 5.
        "rear_leg_set": {"min": -10.0, "max": 10.0, "bins": _bins(-10.0, 10.0)},
        # Knock-kneed = 1, bow-legged = 9; straight (0) = 5.
        "fore_leg_set": {"min": -8.0, "max": 8.0, "bins": _bins(-8.0, 8.0)},
        # Upright/sharp shoulder = 1, sloped = 9; typical ~50-58.
        "shoulder_angle": {"min": 45.0, "max": 65.0, "bins": _bins(45.0, 65.0)},
        # ---------------- Class B: ratios (dimensionless) ----------------
        # Very compact = 1, very long = 9; typical ~1.05-1.20.
        "body_length_to_height_ratio": {"min": 0.90, "max": 1.40, "bins": _bins(0.90, 1.40)},
        # Very narrow chest = 1, very wide = 9; typical ~0.60-0.70.
        "chest_width_to_depth_ratio": {"min": 0.50, "max": 0.80, "bins": _bins(0.50, 0.80)},
        # Very weak loin = 1, very roached = 9; level topline = 5.
        "topline_symmetry": {"min": 0.90, "max": 1.10, "bins": _bins(0.90, 1.10)},
        # Very V-shaped rump = 1, very parallel = 9; typical ~0.55-0.65.
        "pin_width_to_hook_distance_ratio": {"min": 0.40, "max": 0.70, "bins": _bins(0.40, 0.70)},
        # Very deep udder = 1, very shallow = 9; typical ~0.20-0.40.
        "udder_depth_ratio": {"min": 0.0, "max": 0.60, "bins": _bins(0.0, 0.60, reverse=True)},
        # Very narrow frame = 1, very broad = 9; typical ~0.35-0.40.
        "body_width_to_length_ratio": {"min": 0.30, "max": 0.45, "bins": _bins(0.30, 0.45)},
        # ---------------- Class C: measurements (cm) ----------------
        # Very small = 1, very large = 9; typical ~160-190.
        "heart_girth": {"min": 130.0, "max": 220.0, "bins": _bins(130.0, 220.0)},
        # Very short = 1, very long = 9; typical ~120-150.
        "body_length": {"min": 100.0, "max": 180.0, "bins": _bins(100.0, 180.0)},
        # Very short = 1, very tall = 9; typical ~125-145.
        "height_at_withers": {"min": 100.0, "max": 170.0, "bins": _bins(100.0, 170.0)},
        # Very short rump = 1, very long = 9; typical ~45-55.
        "rump_length": {"min": 35.0, "max": 65.0, "bins": _bins(35.0, 65.0)},
        # Very narrow = 1, very wide = 9; typical ~40-50.
        "chest_width": {"min": 30.0, "max": 60.0, "bins": _bins(30.0, 60.0)},
        # Very shallow = 1, very deep = 9; typical ~70-85.
        "chest_depth": {"min": 55.0, "max": 95.0, "bins": _bins(55.0, 95.0)},
        # Very narrow rump = 1, very wide = 9; typical ~45-55.
        "rump_width": {"min": 35.0, "max": 65.0, "bins": _bins(35.0, 65.0)},
        # Very deep udder = 1, very high/short = 9; typical ~-2 to +10 cm vs hock.
        "udder_depth": {"min": -10.0, "max": 25.0, "bins": _bins(-10.0, 25.0, reverse=True)},
    },
    "buffalo": {
        # ---------------- Class A: angles (degrees) ----------------
        "rump_angle": {"min": 0.0, "max": 16.0, "bins": _bins(0.0, 16.0, reverse=True)},
        "hock_angle": {"min": 132.0, "max": 162.0, "bins": _bins(132.0, 162.0)},
        "pastern_angle": {"min": 42.0, "max": 67.0, "bins": _bins(42.0, 67.0)},
        "rear_leg_set": {"min": -10.0, "max": 10.0, "bins": _bins(-10.0, 10.0)},
        "fore_leg_set": {"min": -8.0, "max": 8.0, "bins": _bins(-8.0, 8.0)},
        "shoulder_angle": {"min": 45.0, "max": 65.0, "bins": _bins(45.0, 65.0)},
        # ---------------- Class B: ratios (dimensionless) ----------------
        "body_length_to_height_ratio": {"min": 0.90, "max": 1.40, "bins": _bins(0.90, 1.40)},
        "chest_width_to_depth_ratio": {"min": 0.52, "max": 0.82, "bins": _bins(0.52, 0.82)},
        "topline_symmetry": {"min": 0.90, "max": 1.10, "bins": _bins(0.90, 1.10)},
        "pin_width_to_hook_distance_ratio": {"min": 0.40, "max": 0.70, "bins": _bins(0.40, 0.70)},
        "udder_depth_ratio": {"min": 0.0, "max": 0.60, "bins": _bins(0.0, 0.60, reverse=True)},
        "body_width_to_length_ratio": {"min": 0.30, "max": 0.45, "bins": _bins(0.30, 0.45)},
        # ---------------- Class C: measurements (cm; buffalo larger) ----------------
        "heart_girth": {"min": 150.0, "max": 240.0, "bins": _bins(150.0, 240.0)},
        "body_length": {"min": 115.0, "max": 195.0, "bins": _bins(115.0, 195.0)},
        "height_at_withers": {"min": 110.0, "max": 180.0, "bins": _bins(110.0, 180.0)},
        "rump_length": {"min": 40.0, "max": 70.0, "bins": _bins(40.0, 70.0)},
        "chest_width": {"min": 32.0, "max": 64.0, "bins": _bins(32.0, 64.0)},
        "chest_depth": {"min": 60.0, "max": 105.0, "bins": _bins(60.0, 105.0)},
        "rump_width": {"min": 40.0, "max": 70.0, "bins": _bins(40.0, 70.0)},
        "udder_depth": {"min": -10.0, "max": 25.0, "bins": _bins(-10.0, 25.0, reverse=True)},
    },
}


def score_from_value(trait_id: str, species: str, value: float) -> Tuple[int, float]:
    """Map a trait measurement to a (score_1_9, confidence) for the given species.

    Confidence is highest (~0.95) when the value sits in the middle of a bin and
    lowest (~0.45) at bin boundaries, dropping to 0.3 when the value falls
    entirely outside the calibrated range.
    """
    if species not in SPECIES_RULES:
        raise KeyError(f"Unknown species: {species!r}")
    species_rules = SPECIES_RULES[species]
    if trait_id not in species_rules:
        raise KeyError(f"Unknown trait_id {trait_id!r} for species {species!r}")
    rule = species_rules[trait_id]
    bins = rule["bins"]

    if value < rule["min"] or value > rule["max"]:
        score = bins[0][2] if value < rule["min"] else bins[-1][2]
        return (score, 0.3)

    for lo, hi, score in bins:
        if lo <= value <= hi:
            width = hi - lo
            center = (lo + hi) / 2.0
            offset = abs(value - center) / (width / 2.0) if width > 0 else 0.0
            confidence = round(0.95 - 0.50 * offset, 3)
            confidence = max(confidence, 0.45)
            return (score, confidence)

    raise ValueError(f"No bin matches value {value} for trait {trait_id!r} ({species})")