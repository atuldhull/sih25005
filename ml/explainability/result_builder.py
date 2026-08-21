"""Result contract builder: final assembly point producing the top-level ScoringResult."""

from dataclasses import is_dataclass
from typing import Dict, List, Optional

from ml.common.schemas import (
    EligibilityResult,
    MeasurementResult,
    ScoreResult,
    ScoringResult,
    SymptomVector,
    TagResult,
    WeightResult,
)
from ml.config.traits import get_contract_traits
from ml.explainability.explainer import generate_trait_explanation, generate_overlay_data


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









"""
ADD THIS to ml/explainability/result_builder.py (do not replace the whole file -
keep build_scoring_result() and scoring_result_to_dict() exactly as they are;
those still produce your internal shape and your 26 tests still exercise them).

This adds ONE new function, to_contract_dict(), which is the adapter the review
asked for: it takes your internal ScoringResult + the explainer's per-trait
data + the raw measurements, and emits the exact contract/scoring_result.json
shape. Internal shape never leaks past this function.

Per REVIEW-ml-dev.md "Division of labor": animal_id, species, breed_*, captured,
traits[20], weight_kg, symptom_vector, health_flags are YOURS to emit.
session_id, eligible/eligible_reason, risk_report, herd_alerts, reports,
escalated, captured_at, synced are SERVER-injected - this function intentionally
does NOT set them (or sets safe empty/None placeholders the server overwrites).
"""

from typing import Dict, List, Optional

from ml.common.schemas import MeasurementResult, ScoreResult, ScoringResult, WeightResult
from ml.config.traits import get_contract_traits
from ml.explainability.explainer import generate_trait_explanation, generate_overlay_data


def to_contract_dict(
    result: ScoringResult,
    measurements: List[MeasurementResult],
    keypoints: dict,
    breed_registered: Optional[str] = None,
    breed_verified: Optional[bool] = None,
    breed_verify_confidence: Optional[float] = None,
    captured: Optional[Dict[str, bool]] = None,
    global_not_scored_reason: Optional[str] = None,
) -> dict:
    """Map internal ScoringResult -> the frozen contract shape (contract/scoring_result.json).

    Args:
        result: your existing internal ScoringResult (unchanged).
        measurements: the same MeasurementResult list used to build `result`,
            needed here because measured_value/unit/trait_class live there,
            not on ScoreResult.
        keypoints: raw keypoints dict, needed to (re)compute overlay points
            per trait via the explainer.
        breed_registered/breed_verified/breed_verify_confidence: from your
            Tag Intelligence layer once implemented; pass None until then -
            the contract fields will be null, which is honest and correct
            for the current NOT_SCORED/PARTIAL states.
        captured: dict like {"side_photo": True, "rear_photo": True,
            "gait_video": False}; pass explicitly from pipeline.py based on
            which inputs were actually provided/passed QC.
        global_not_scored_reason: when the WHOLE pipeline failed early (no
            animal detected, unrecognized detection label, quality gate
            failed, pose estimation not yet implemented, etc.), pass the
            real reason string here. The contract has no top-level
            status/warnings field for pipeline mode - the only honest place
            to surface "why nothing was scored" is inside every trait's
            not_scored_reason, so this overrides the generic per-trait
            message with the real, specific cause. Leave None for a normal
            run where individual traits may still legitimately be
            not-measurable for their own separate reasons.

    Returns:
        A JSON-serializable dict matching contract/scoring_result.json exactly
        for every field this module owns. Fields owned by the server
        (session_id, eligible, eligible_reason, risk_report, herd_alerts,
        reports, escalated, captured_at, synced) are intentionally omitted
        or set to safe defaults - the server overwrites/injects them.
    """
    measurement_by_trait = {m.trait_id: m for m in measurements}
    score_by_trait = {s.trait_id: s for s in result.traits}

    traits_out = []
    for trait_def in get_contract_traits():
        trait_id = trait_def["trait_id"]
        measurement = measurement_by_trait.get(trait_id)
        score = score_by_trait.get(trait_id)

        # Trait wasn't measured at all this run (e.g. upstream stage not implemented
        # yet) - emit a fully honest not_scored entry rather than skipping it, since
        # the contract expects all 20 trait names present every time.
        if measurement is None or score is None or score.score_1_9 is None:
            traits_out.append(
                {
                    "name": trait_def["name"],
                    "category": trait_def["category"],
                    "score": None,
                    "confidence": measurement.confidence if measurement else 0.0,
                    "measured_value": None,
                    "ci": None,
                    "measure_class": trait_def["trait_class"],
                    "view": trait_def["view"],
                    "overlay_points": [],
                    "explanation": None,
                    "not_scored_reason": (
                        global_not_scored_reason
                        if global_not_scored_reason
                        else (score.not_scored_reason
                              if score is not None
                              and getattr(score, "not_scored_reason", None)
                              else (
                            "; ".join(measurement.flags)
                            if measurement and measurement.flags
                            else (
                                # REVIEW-ml-dev.md, Important Fix #4: a measurement
                                # was successfully computed (has a value, no flags)
                                # but fell outside the trait's calibrated range, so
                                # score_trait() refused to score it - distinct from
                                # the missing-keypoints case below.
                                "measured value outside calibrated range for this trait"
                                if measurement and measurement.value is not None
                                else "required keypoints not available"
                            )
                        ))
                    ),
                }
            )
            continue

        overlay = generate_overlay_data(trait_id, keypoints)
        # species matters: a buffalo scored against the cattle band table
        # would be told it fell in a band that does not apply to it.
        explanation = generate_trait_explanation(
            measurement, score, getattr(result, "species", "cattle") or "cattle")

        measured_value_str = (
            f"{measurement.value:.1f} {measurement.unit}"
            if measurement.value is not None
            else None
        )

        # The interval, when the measurement carries one. Angle and ratio
        # traits now compute an uncertainty from their own geometry - an angle
        # taken across a segment a few percent of the animal long inherits a
        # much wider one than the same angle across its whole body - so there
        # is a real interval to report rather than a fabricated one.
        #
        # Still null where nothing computes it, which is what the field meant
        # before: null is "we do not know", never "the measurement is exact".
        ci_str = None
        if (measurement.value is not None
                and getattr(measurement, "uncertainty", None) is not None):
            u = abs(measurement.uncertainty)
            lo, hi = measurement.value - u, measurement.value + u
            # "135.0-141.0 cm" reads fine; "-3.0-4.8 degrees" does not - the
            # hyphen becomes ambiguous the moment a bound is negative, and
            # signed traits (a cow-hock deviation, an udder above the hock)
            # produce those routinely.
            sep = " to " if lo < 0 else "-"
            ci_str = f"{lo:.1f}{sep}{hi:.1f} {measurement.unit}"

        traits_out.append(
            {
                "name": trait_def["name"],
                "category": trait_def["category"],
                "score": score.score_1_9,
                "confidence": round(score.confidence, 2),
                "measured_value": measured_value_str,
                "ci": ci_str,
                "measure_class": trait_def["trait_class"],
                "view": trait_def["view"],
                "overlay_points": [[int(round(p[0])), int(round(p[1]))] for p in overlay["points"]],
                "explanation": explanation,
            }
        )

    weight_out = _weight_to_contract(result.weight)
    symptom_vector_out = _symptom_vector_to_contract(result.symptom_vector)

    return {
        # --- server-owned fields: safe placeholders, server overwrites these ---
        "session_id": None,
        "eligible": None,
        "eligible_reason": None,
        # --- fields this module owns ---
        "animal_id": result.animal_id,
        "species": result.species,
        "breed_registered": breed_registered,
        "breed_verified": breed_verified,
        "breed_verify_confidence": breed_verify_confidence,
        "captured": captured or {"side_photo": False, "rear_photo": False, "gait_video": False},
        "traits": traits_out,
        "weight_kg": weight_out,
        "symptom_vector": symptom_vector_out,
        # --- server-owned, safe placeholders ---
        "risk_report": [],
        "herd_alerts": [],
        "reports": None,
        "escalated": False,
        "health_flags": _health_flags_from_symptoms(result.symptom_vector),
        "captured_at": None,
        "synced": False,
    }


def _weight_to_contract(weight: Optional[WeightResult]) -> dict:
    """Map WeightResult{estimate_kg, range_kg, confidence} -> weight_kg{low, high, method, cross_check}."""
    if weight is None or weight.estimate_kg is None:
        return {"low": None, "high": None, "method": None, "cross_check": None}
    low, high = weight.range_kg
    return {
        "low": round(low, 1) if low is not None else None,
        "high": round(high, 1) if high is not None else None,
        "method": weight.method or "girth-length-regression",
        "cross_check": weight.cross_check,
    }


def _symptom_vector_to_contract(symptom_vectors: list) -> List[dict]:
    """Map internal SymptomVector{category, present, confidence} -> contract
    {symptom, confidence, region, source}.

    IMPORTANT: `symptom` values MUST come from the vocabulary defined in
    server/vkg.json (per REVIEW-ml-dev.md B1). This function currently passes
    `category` straight through as `symptom` and leaves `region`/`source` as
    None placeholders - before this is production-correct, cross-check every
    category name your vet_screening module produces against vkg.json's
    allowed symptom vocabulary, and fill in region/source from whichever
    module actually generated the signal (e.g. source="video" for gait
    symptoms from the Gait Analyzer, source="image" for posture/BCS symptoms).
    Only include symptoms that are actually present (present=True) - the
    contract's symptom_vector is a list of detected signals, not every
    possible category.
    """
    out = []
    for sv in symptom_vectors:
        if not sv.present:
            continue
        out.append(
            {
                "symptom": sv.category,  # TODO: validate against server/vkg.json vocabulary
                "confidence": round(sv.confidence, 2),
                "region": None,  # TODO: fill in per symptom once vet_screening supplies it
                "source": None,  # TODO: "image" | "video" depending on originating module
            }
        )
    return out


def _health_flags_from_symptoms(symptom_vectors: list) -> List[str]:
    """Derive the top-level health_flags[] list from present symptoms.

    Placeholder pass-through of category names for now - once vet_screening
    and the vkg.json vocabulary are finalized, this should map categories to
    the specific flag strings the app expects (e.g. "locomotion_abnormal").
    """
    return [sv.category for sv in symptom_vectors if sv.present]