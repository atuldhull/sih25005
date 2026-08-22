"""Scoring engine: maps trait measurements to 1-9 scores and eligibility/status gates."""

from typing import List, Optional

from ml.common.schemas import EligibilityResult, MeasurementResult, ScoreResult
from ml.config.rules import score_from_value, SPECIES_RULES

VALID_SPECIES = tuple(SPECIES_RULES)

# How much of a trait's scoring band the uncertainty may cover before a score
# stops meaning anything. At 1.0 the error bar spans the entire band; two
# thirds already leaves most of the nine bins indistinguishable, and that is
# where the line is drawn.
UNCERTAINTY_BAND_FRACTION = 0.67


def _band_width(trait_id: str, species: str) -> Optional[float]:
    """Width of the calibrated range a 1-9 score is drawn from."""
    band = SPECIES_RULES.get(species, {}).get(trait_id)
    if not band:
        return None
    try:
        return float(band["max"]) - float(band["min"])
    except (KeyError, TypeError, ValueError):
        return None


def score_trait(measurement: MeasurementResult, species: str) -> ScoreResult:
    """Score a single trait measurement into a 1-9 ScoreResult.

    Unmeasurable traits (value is None) return score_1_9=None with zero confidence.
    A measured value outside the trait's calibrated range also returns
    score_1_9=None (REVIEW-ml-dev.md, Important Fix #4: refuse instead of
    clamping to a confident-looking extreme score). Otherwise confidence
    combines the measurement's quality with the rule lookup's bin confidence.
    """
    if measurement.value is None:
        return ScoreResult(trait_id=measurement.trait_id, score_1_9=None, confidence=0.0)

    # A measurement whose uncertainty covers the whole range the score is drawn
    # from is not a measurement. Two traits are built on segments a few percent
    # of the animal long - foot_angle on pastern-to-hoof, shoulder_angle on
    # withers-to-chest - and keypoint error there works out at +/-25 and +/-27
    # degrees against bands 25 and 20 degrees wide. Every 1-9 bin falls inside
    # the error bar, so which bin comes out is decided by noise.
    #
    # The rules module owns the bands, so this check belongs here rather than
    # in measurement, which computes the uncertainty but cannot know what it
    # is being compared against.
    band = _band_width(measurement.trait_id, species)
    if measurement.uncertainty is not None and band:
        if measurement.uncertainty >= UNCERTAINTY_BAND_FRACTION * band:
            return ScoreResult(
                trait_id=measurement.trait_id, score_1_9=None, confidence=0.0,
                not_scored_reason=(
                    f"too imprecise to score: the landmarks this trait uses "
                    f"put the value within +/-{measurement.uncertainty:.0f} "
                    f"{measurement.unit}, against a scoring range only "
                    f"{band:g} wide - every 1-9 bin falls inside that"))

    score, rule_confidence = score_from_value(measurement.trait_id, species, measurement.value)
    if score is None:
        # Out of range. WHY matters, and the uncertainty distinguishes the two
        # cases. A value that is far outside its band while being precisely
        # measured is not an unusual animal - it is landmarks in the wrong
        # place, and saying "outside the calibrated range" implies the opposite.
        precise = (measurement.uncertainty is not None and band
                   and measurement.uncertainty < 0.25 * band)
        reason = (
            f"measured {measurement.value:.2f} {measurement.unit}, outside the "
            f"calibrated range. The measurement itself is tight "
            f"(+/-{measurement.uncertainty:.2f}), so the landmarks it was "
            f"built from are more likely wrong than the animal unusual"
            if precise else
            "measured value outside calibrated range for this trait")
        return ScoreResult(trait_id=measurement.trait_id, score_1_9=None,
                           confidence=0.0, not_scored_reason=reason)

    return ScoreResult(
        trait_id=measurement.trait_id,
        score_1_9=score,
        confidence=measurement.confidence * rule_confidence,
    )


def score_all_traits(measurements: List[MeasurementResult], species: str) -> List[ScoreResult]:
    """Score every measurement in the list via score_trait()."""
    return [score_trait(m, species) for m in measurements]


def scoreability(
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