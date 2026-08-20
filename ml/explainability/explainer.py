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


def generate_trait_explanation(measurement: MeasurementResult, score: ScoreResult) -> str:
    """Return a short human-readable sentence explaining one trait's measurement."""
    name = get_trait(measurement.trait_id)["name"]
    if measurement.value is None or score is None or score.score_1_9 is None:
        return f"{name} not measurable - required keypoints unreliable."
    return (
        f"{name} measured at {measurement.value:.1f} {measurement.unit}, "
        f"{_confidence_band(measurement.confidence)} ({measurement.confidence:.2f})."
    )


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
        if confidence < 0.3:
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
) -> dict:
    """Assemble per-trait text explanations and overlay data for all traits."""
    score_by_trait = {s.trait_id: s for s in scores}
    text_summary: List[str] = []
    overlays: List[dict] = []
    for measurement in measurements:
        score = score_by_trait.get(measurement.trait_id)
        text_summary.append(generate_trait_explanation(measurement, score))
        overlays.append(generate_overlay_data(measurement.trait_id, keypoints))
    return {"text_summary": text_summary, "overlays": overlays}