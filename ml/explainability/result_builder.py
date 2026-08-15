"""Result contract builder: final assembly point producing the top-level ScoringResult."""

from dataclasses import is_dataclass
from typing import Dict, List, Optional

from common.schemas import (
    EligibilityResult,
    MeasurementResult,
    ScoreResult,
    ScoringResult,
    SymptomVector,
    TagResult,
    WeightResult,
)


def build_scoring_result(
    animal_id: Optional[str],
    species: str,
    measurements: List[MeasurementResult],
    scores: List[ScoreResult],
    eligibility: EligibilityResult,
    status: str,
    tag_result: Optional[TagResult] = None,
    weight_result: Optional[WeightResult] = None,
    symptom_vectors: Optional[List[SymptomVector]] = None,
    model_versions: Optional[Dict[str, str]] = None,
    extra_warnings: Optional[List[str]] = None,
) -> ScoringResult:
    """Assemble every pipeline stage's output into the final ScoringResult.

    This is the final assembly point per architecture Section 11.6, matching the
    schema exactly. Note the schema has no measurements field; the measurements
    argument is accepted to derive ``warnings`` marking any trait that could not
    be measured (merged with any passed via ``extra_warnings``, e.g. image
    quality failures or unimplemented stages). All optional inputs
    (tag/weight/vet data) degrade gracefully to empty lists or None so partial
    pipeline runs produce valid results.
    """
    warnings = [
        f"{m.trait_id}: not_measurable" for m in measurements if m.value is None
    ]
    warnings.extend(extra_warnings or [])

    return ScoringResult(
        animal_id=animal_id,
        species=species,
        status=status,
        tag=tag_result,
        traits=scores,
        weight=weight_result,
        symptom_vector=symptom_vectors or [],
        eligibility=eligibility,
        warnings=warnings,
        model_versions=model_versions or {},
    )


def scoring_result_to_dict(result: ScoringResult) -> dict:
    """Convert a ScoringResult (and nested dataclasses) into a JSON-serializable dict."""
    return _to_jsonable(result)


def _to_jsonable(obj):
    """Recursively convert dataclasses, tuples, lists, and dicts to plain JSON types."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if is_dataclass(obj) and not isinstance(obj, type):
        return {key: _to_jsonable(value) for key, value in vars(obj).items()}
    if isinstance(obj, tuple):
        return [_to_jsonable(item) for item in obj]
    if isinstance(obj, list):
        return [_to_jsonable(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _to_jsonable(value) for key, value in obj.items()}
    return obj