"""Scoring engine: maps trait measurements to 1-9 scores and eligibility/status gates."""

from typing import List

from ml.common.schemas import EligibilityResult, MeasurementResult, ScoreResult
from ml.config.rules import score_from_value, SPECIES_RULES

VALID_SPECIES = tuple(SPECIES_RULES)


def score_trait(measurement: MeasurementResult, species: str) -> ScoreResult:
    """Score a single trait measurement into a 1-9 ScoreResult.

    Unmeasurable traits (value is None) return score_1_9=None with zero confidence.
    Otherwise confidence combines the measurement's quality with the rule lookup's
    bin confidence.
    """
    if measurement.value is None:
        return ScoreResult(trait_id=measurement.trait_id, score_1_9=None, confidence=0.0)

    score, rule_confidence = score_from_value(measurement.trait_id, species, measurement.value)
    return ScoreResult(
        trait_id=measurement.trait_id,
        score_1_9=score,
        confidence=measurement.confidence * rule_confidence,
    )


def score_all_traits(measurements: List[MeasurementResult], species: str) -> List[ScoreResult]:
    """Score every measurement in the list via score_trait()."""
    return [score_trait(m, species) for m in measurements]


def check_eligibility(
    measurements: List[MeasurementResult],
    species: str,
    quality_passed: bool,
    min_traits_required: int = 10,
) -> EligibilityResult:
    """Determine whether the animal is eligible for scoring.

    Fails on poor image quality, fewer measurable traits than required, or an
    unsupported species. Passed only when no reasons were recorded.
    """
    reasons: List[str] = []
    if not quality_passed:
        reasons.append("image_quality_failed")
    measurable_count = sum(1 for m in measurements if m.value is not None)
    if measurable_count < min_traits_required:
        reasons.append("insufficient_measurable_traits")
    if species not in VALID_SPECIES:
        reasons.append("invalid_species")
    return EligibilityResult(passed=len(reasons) == 0, reasons=reasons)


def determine_status(eligibility: EligibilityResult, measurements: List[MeasurementResult]) -> str:
    """Return the top-level status: SCORED, PARTIAL, or NOT_SCORED."""
    if not eligibility.passed:
        return "NOT_SCORED"
    measured = sum(1 for m in measurements if m.value is not None)
    if measured == len(measurements):
        return "SCORED"
    return "PARTIAL"