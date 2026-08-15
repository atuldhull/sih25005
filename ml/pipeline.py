"""Top-level pipeline orchestrator chaining all implemented stages into score_animal()."""
from typing import Dict, List, Optional

from ml.common.schemas import MeasurementResult
from ml.detection.detector import (
    DetectionBackendError,
    DetectionLabelError,
    detect_animal,
    detect_ear_tag,
)
from ml.explainability.explainer import assemble_explainability
from ml.explainability.result_builder import build_scoring_result, to_contract_dict
from ml.ingestion.quality_validation import validate_image, validate_video
from ml.measurement.traits import measure_all_traits
from ml.scoring.scorer import check_eligibility, determine_status, score_all_traits
from ml.weight.estimator import estimate_weight

KEYPOINT_CONFIDENCE_MIN = 0.3


def score_animal(
    side_img: str,
    rear_img: str,
    video_path: Optional[str],
    animal_record: Dict[str, object],
) -> dict:
    """Top-level entrypoint per architecture Section 6/15.

    animal_record must contain at least {"animal_id": str|None,
    "species": "cattle"|"buffalo"}. Returns a JSON-serializable dict matching
    the frozen contract/scoring_result.json shape (pipeline mode) via
    to_contract_dict() - NOT the internal ScoringResult shape.
    """
    animal_id = animal_record.get("animal_id")
    species = animal_record.get("species", "cattle")

    # `captured` reflects whether an input was PROVIDED to this call, not
    # whether it passed quality checks - side_img/rear_img are required
    # params so they're always "captured" in that sense; gait_video depends
    # on whether a video_path was given at all.
    captured = {
        "side_photo": True,
        "rear_photo": True,
        "gait_video": video_path is not None,
    }

    # ---- Stage 1: Ingestion / quality gate ---------------------------------
    side_q = validate_image(side_img, "side")
    rear_q = validate_image(rear_img, "rear")
    video_q = validate_video(video_path) if video_path else None

    if not side_q["passed"]:
        reason = "side_image_quality_failed: " + ", ".join(side_q["reasons"])
        return _build_not_scored(animal_id, species, reason, quality_passed=False, captured=captured)

    quality_passed = side_q["passed"] and rear_q["passed"] and (video_q is None or video_q["passed"])

    # ---- Stage 2: Detection (RT-DETRv2) ------------------------------------
    # Runs the RT-DETR detector on the side image. If the backend or weights
    # aren't available (DetectionBackendError), the checkpoint returns a class
    # label outside RAW_LABEL_TO_CLASS_NAME (DetectionLabelError), or no animal
    # is found at all, scoring cannot proceed -> NOT_SCORED with a truthful
    # reason (never a crash).
    try:
        animal_detection = detect_animal(side_img)
    except DetectionBackendError as exc:
        reason = f"detection_backend_unavailable: {exc}"
        return _build_not_scored(animal_id, species, reason, quality_passed, captured)
    except DetectionLabelError as exc:
        reason = f"detection_label_unrecognized: {exc}"
        return _build_not_scored(animal_id, species, reason, quality_passed, captured)

    if animal_detection is None:
        return _build_not_scored(animal_id, species, "no_animal_detected", quality_passed, captured)

    animal_bbox = animal_detection.bbox
    try:
        tag_detection = detect_ear_tag(side_img, animal_bbox)
        tag_bbox = tag_detection.bbox if tag_detection is not None else None
    except (DetectionBackendError, DetectionLabelError):
        tag_bbox = None

    # ---- TODO Stage 3: Tag Intelligence (solvePnP, OCR) --------------------
    # NOT YET IMPLEMENTED. Will fill tag_result: identity, breed, scale.
    tag_result = None
    scale_factor = None
    scale_confidence = 0.0

    # ---- TODO Stage 4: Pose/Features (RTMPose, DINOv2) ---------------------
    # NOT YET IMPLEMENTED. Once built, this stage produces the real keypoints
    # dict; until then we keep keypoints=None so measurement cannot run.
    keypoints = None

    # ---- TODO Stage 7: Vet Pre-Screen --------------------------------------
    # NOT YET IMPLEMENTED. Passed as None so the result degrades gracefully.
    symptom_vectors = None

    if keypoints is None:
        reasons = []
        if not quality_passed:
            for label, q in (("rear", rear_q), ("video", video_q)):
                if q is not None and not q["passed"]:
                    reasons.append(f"{label}_quality_failed: " + ", ".join(q["reasons"]))
        reasons.append("pose_estimation_not_implemented")
        return _build_not_scored(animal_id, species, "; ".join(reasons), quality_passed, captured)

    # ---- Stages 5-6: Measurement -> Scoring -> Status -----------------------
    measurements: List[MeasurementResult] = measure_all_traits(
        keypoints,
        scale_factor,
        species,
        scale_confidence,
    )
    scores = score_all_traits(measurements, species)
    eligibility = check_eligibility(measurements, species, quality_passed=quality_passed)
    status = determine_status(eligibility, measurements)
    explainability = assemble_explainability(measurements, scores, keypoints)
    weight_result = estimate_weight(measurements)

    result = build_scoring_result(
        animal_id=animal_id,
        species=species,
        measurements=measurements,
        scores=scores,
        eligibility=eligibility,
        status=status,
        tag_result=tag_result,
        weight_result=weight_result,
        symptom_vectors=symptom_vectors,
        model_versions={},  # TODO: populate from each model when implemented
        extra_warnings=explainability.get("text_summary", []),
    )
    # THE ACTUAL FIX: return the contract-shaped dict, not the internal shape.
    # Breed fields stay None until Tag Intelligence (Stage 3) is implemented -
    # that's honest, not a bug: we haven't verified breed, so we don't claim to.
    return to_contract_dict(
        result,
        measurements,
        keypoints,
        breed_registered=None,
        breed_verified=None,
        breed_verify_confidence=None,
        captured=captured,
    )


def _build_not_scored(
    animal_id: Optional[str],
    species: str,
    reason: str,
    quality_passed: bool,
    captured: Dict[str, bool],
) -> dict:
    """Build a contract-shaped NOT_SCORED result for degraded/unimplemented runs.

    `reason` is threaded into every trait's not_scored_reason via
    to_contract_dict()'s global_not_scored_reason - the contract has no
    top-level status/warnings field for pipeline mode, so this is the only
    honest place to surface why nothing could be measured.
    """
    eligibility = check_eligibility(
        measurements=[],
        species=species,
        quality_passed=quality_passed,
    )
    result = build_scoring_result(
        animal_id=animal_id,
        species=species,
        measurements=[],
        scores=[],
        eligibility=eligibility,
        status="NOT_SCORED",
        tag_result=None,
        extra_warnings=[reason],
    )
    return to_contract_dict(
        result,
        measurements=[],
        keypoints={},
        captured=captured,
        global_not_scored_reason=reason,
    )