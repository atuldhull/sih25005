"""Explainability assembler: human-readable explanations and overlay geometry per trait."""

from typing import Dict, List

from ml.common.schemas import MeasurementResult, ScoreResult
from ml.config.traits import get_trait


def _confidence_band(confidence: float) -> str:
    """Map a confidence value to its human-readable band label."""
    if confidence >= 0.75:
        return "high confidence"
    if confidence >= 0.40:
        return "moderate confidence"
    return "low confidence"


def _band_for(trait_id: str, species: str, value: float):
    """The (lo, hi, score) bin a value fell into, and the trait's full range."""
    try:
        from ml.config.rules import SPECIES_RULES
        rule = SPECIES_RULES.get(species, {}).get(trait_id)
        if not rule:
            return None, None, None
        for lo, hi, sc in rule["bins"]:
            if lo <= value <= hi:
                return (lo, hi, sc), rule.get("min"), rule.get("max")
        return None, rule.get("min"), rule.get("max")
    except Exception:
        return None, None, None


def generate_trait_explanation(measurement: MeasurementResult, score: ScoreResult,
                               species: str = "cattle") -> str:
    """Explain a trait: what was measured, and WHY that is the score it is.

    This used to restate the number and its confidence - "Heart Girth measured
    at 164.5 cm, moderate confidence (0.50)" - which tells a farmer nothing
    they cannot already see on the row above it. The one thing a scorecard has
    to answer is why a 5 is a 5, and the answer is that the value fell in a
    particular band of the 1-9 scale. The scorer knows that band; nobody was
    passing it on.

    Deliberately no "good" or "bad". These scores are biological measurements,
    not quality judgements - a 9 is not a better animal than a 3, it is a
    differently shaped one - and the app renders every score in one neutral
    treatment for the same reason.
    """
    name = get_trait(measurement.trait_id)["name"]
    if measurement.value is None or score is None or score.score_1_9 is None:
        return f"{name} not measurable - required keypoints unreliable."

    unit = measurement.unit or ""
    band, rmin, rmax = _band_for(measurement.trait_id, species, measurement.value)

    parts = [f"Measured {measurement.value:.1f} {unit}".rstrip() + "."]
    if band:
        lo, hi, sc = band
        parts.append(
            f"That falls in the {lo:.1f}-{hi:.1f} {unit}".rstrip()
            + f" band, which is score {sc} of 9 for {species}.")
    if rmin is not None and rmax is not None:
        parts.append(f"The full calibrated range for this trait is "
                     f"{rmin:g}-{rmax:g} {unit}".rstrip() + ".")
    if measurement.uncertainty is not None:
        parts.append(f"Uncertainty +/-{measurement.uncertainty:.1f} {unit}".rstrip() + ".")
    parts.append(f"Landmark confidence {_confidence_band(measurement.confidence)} "
                 f"({measurement.confidence:.2f}).")
    return " ".join(parts)


from ml.measurement.traits import (  # noqa: E402
    KEYPOINT_CONFIDENCE_THRESHOLD)


def generate_overlay_data(trait_id: str, keypoints: dict) -> dict:
    """Return the coordinates needed to draw this trait's measurement on an image.

    Pulls the required keypoints from TRAIT_REGISTRY and their (x, y) positions
    from the keypoints dict, skipping any that are missing or low-confidence.
    Returns {"trait_id", "points", "line_segments"}.
    """
    required = get_trait(trait_id)["required_keypoints"]
    points: List[tuple] = []
    for name in required:
        if name not in keypoints:
            continue
        x, y, confidence = keypoints[name]
        # The SAME threshold measurement used, imported rather than repeated.
        # This was hardcoded at 0.3 while measurement moved to 0.10, so a
        # trait computed from a joint at 0.15 had that joint missing from its
        # own overlay. The overlay is shown to a judge and a vet officer as
        # the proof of the measurement - it has to show the points the
        # measurement actually used, or it is proof of something else.
        if confidence < KEYPOINT_CONFIDENCE_THRESHOLD:
            continue
        points.append((x, y))

    line_segments = [(points[i], points[i + 1]) for i in range(len(points) - 1)]
    return {
        "trait_id": trait_id,
        "points": points,
        "line_segments": line_segments,
    }


def assemble_explainability(
    measurements: List[MeasurementResult],
    scores: List[ScoreResult],
    keypoints: dict,
    species: str = "cattle",
) -> dict:
    """Assemble per-trait text explanations and overlay data for all traits."""
    score_by_trait = {s.trait_id: s for s in scores}
    text_summary: List[str] = []
    overlays: List[dict] = []
    for measurement in measurements:
        score = score_by_trait.get(measurement.trait_id)
        text_summary.append(generate_trait_explanation(measurement, score, species))
        overlays.append(generate_overlay_data(measurement.trait_id, keypoints))
    return {"text_summary": text_summary, "overlays": overlays}