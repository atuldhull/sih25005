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
from ml.explainability.result_builder import build_scoring_result, scoring_result_to_dict
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
    the ScoringResult contract.
    """
    animal_id = animal_record.get("animal_id")
    species = animal_record.get("species", "cattle")

    # ---- Stage 1: Ingestion / quality gate ---------------------------------
    side_q = validate_image(side_img, "side")
    rear_q = validate_image(rear_img, "rear")
    video_q = validate_video(video_path) if video_path else None

    if not side_q["passed"]:
        warnings = ["side_image_quality_failed: " + ", ".join(side_q["reasons"])]
        return _build_not_scored(animal_id, species, warnings, quality_passed=False)

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
        return _build_not_scored(animal_id, species, [f"detection_backend_unavailable: {exc}"], quality_passed)
    except DetectionLabelError as exc:
        return _build_not_scored(animal_id, species, [f"detection_label_unrecognized: {exc}"], quality_passed)

    if animal_detection is None:
        return _build_not_scored(animal_id, species, ["no_animal_detected"], quality_passed)

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
        warnings = []
        if not quality_passed:
            for label, q in (("rear", rear_q), ("video", video_q)):
                if q is not None and not q["passed"]:
                    warnings.append(f"{label}_quality_failed: " + ", ".join(q["reasons"]))
        warnings.append("pose_estimation_not_implemented")
        return _build_not_scored(animal_id, species, warnings, quality_passed)

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
    return scoring_result_to_dict(result)


def _build_not_scored(
    animal_id: Optional[str],
    species: str,
    warnings: List[str],
    quality_passed: bool,
) -> dict:
    """Build a NOT_SCORED result dict for degraded/unimplemented runs.

    Eligibility reasons are populated truthfully via check_eligibility() against
    the empty measurements list, so the gate always explains why scoring failed.
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
        extra_warnings=warnings,
    )
    return scoring_result_to_dict(result)
