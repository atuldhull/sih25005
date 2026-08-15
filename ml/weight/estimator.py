"""PLACEHOLDER weight estimator using Schaeffer's livestock formula.

This is a formula-based heuristic, NOT a trained model. The real implementation
will be a fine-tuned girth/length -> weight regression per architecture Section 5
once training data linking visual measurements to actual weight is collected.
"""

from typing import List

from common.schemas import MeasurementResult, WeightResult

SCHAFFER_DIVISOR = 10838.4
MIN_WEIGHT_FACTOR = 0.85
MAX_WEIGHT_FACTOR = 1.15


def estimate_weight(measurements: List[MeasurementResult]) -> WeightResult:
    """Estimate weight from heart girth and body length measurements (cm).

    Schaeffer's formula expressed directly in metric units:
        weight_kg = (heart_girth_cm^2 * body_length_cm) / 10838.4
    where 10838.4 = 300 * (2.54^3) * 2.20462, the divisor that converts the
    imperial girth_in^2 * length_in / 300 form into cm^2 * kg units. The +/-
    range and confidence are rough placeholders (real prediction intervals will
    come from the trained regression model later).
    """
    girth = next((m for m in measurements if m.trait_id == "heart_girth"), None)
    length = next((m for m in measurements if m.trait_id == "body_length"), None)

    if girth is None or length is None or girth.value is None or length.value is None:
        return WeightResult(estimate_kg=None, range_kg=(None, None), confidence=0.0)

    weight_kg = (girth.value ** 2 * length.value) / SCHAFFER_DIVISOR
    confidence = (girth.confidence + length.confidence) / 2.0

    return WeightResult(
        estimate_kg=weight_kg,
        range_kg=(weight_kg * MIN_WEIGHT_FACTOR, weight_kg * MAX_WEIGHT_FACTOR),
        confidence=confidence,
    )