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
from ml.pose_features.embedding_extractor import (
    BreedBackendError,
    to_contract_fields,
    verify_breed,
)
from ml.pose_features.silhouette_landmarks import (
    add_derived_landmarks,
    add_rear_view_landmarks,
)
from ml.pose_features.pose_extractor import (
    PoseBackendError,
    extract_keypoints,
    usable_joint_count,
)
from ml.tag_intelligence.tag_reader import read_tag, scale_factor_from
from ml.vet_screening.vet_screener import (
    screen,
    screening_notes,
    symptom_vector_or_empty,
)
from ml.scoring.scorer import scoreability, determine_status, score_all_traits
from ml.weight.estimator import estimate_weight

KEYPOINT_CONFIDENCE_MIN = 0.3


def score_animal(
    side_img: str,
    rear_img: str,
    video_path: Optional[str],
    animal_record: Dict[str, object],
    tag_img: Optional[str] = None,
) -> dict:
    """Top-level entrypoint per architecture Section 6/15.

    animal_record must contain at least {"animal_id": str|None,
    "species": "cattle"|"buffalo"}. Returns a JSON-serializable dict matching
    the frozen contract/scoring_result.json shape (pipeline mode) via
    to_contract_dict() - NOT the internal ScoringResult shape.
    """
    # Real BPA records from server/seed.py key the animal by "_id", not
    # "animal_id" - fall back so real server records resolve correctly
    # instead of always returning None (verified: server/main.py passes
    # the raw MongoDB document straight through as animal_record).
    animal_id = animal_record.get("animal_id") or animal_record.get("_id")
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

    # Only a defect that makes the work impossible stops it. A soft or badly
    # exposed photograph is recorded (quality_passed goes False below, which
    # reaches the eligibility calculation and the not-scored reasons) and then
    # given its chance - blur is a poor predictor of whether pose will work,
    # and the outcome gate further down measures that directly instead.
    if side_q.get("fatal"):
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

    # The ear tag is found on the CLOSE-UP tag photo, not the side photo.
    # Measured on this checkpoint: a close-up of a tag scores 0.84, while the
    # best ear_tag proposal across 29 whole-animal photos was 0.02 against a
    # 0.5 threshold. A tag on a standing animal is a handful of pixels; the
    # detector was never going to see it there, and asking it to is why the
    # centimetre scale never appeared.
    #
    # The app already captures this separately (ScanTagScreen), so the photo
    # exists - it just has to reach here. tag_img is optional so the existing
    # four-argument call in scoring_loader keeps working; without it we still
    # try the side photo, which costs one forward pass and almost always
    # returns None.
    tag_source = tag_img or side_img
    tag_bbox = None
    tag_from_closeup = tag_img is not None
    try:
        tag_animal = animal_bbox
        if tag_from_closeup:
            # On a close-up the "animal" box is the ear filling the frame, so
            # containment against the SIDE photo's animal box would reject
            # every tag. Constrain to the tag photo's own frame instead.
            from PIL import Image as _Image
            with _Image.open(tag_source) as _im:
                tag_animal = (0.0, 0.0, float(_im.width), float(_im.height))
        tag_detection = detect_ear_tag(tag_source, tag_animal)
        tag_bbox = tag_detection.bbox if tag_detection is not None else None
    except (DetectionBackendError, DetectionLabelError):
        tag_bbox = None
    except Exception:
        tag_bbox = None

    # ---- Stage 3: Tag Intelligence -> a centimetre scale -------------------
    # Scale comes from the tag's 27 mm round BUTTON, never the outer panel:
    # that panel is 55-69 mm depending on the supplier, so using it as a ruler
    # would bake an unknown ~20% error into every class-C trait, invisibly.
    #
    # A refusal here is a designed outcome, not a failure. Without a scale the
    # class-A angle traits still measure fine - angles need no scale - so the
    # run degrades to those instead of inventing centimetres.
    tag_result = None
    scale_factor = None
    scale_confidence = 0.0
    if tag_bbox is not None:
        try:
            from ml.detection.detector import _get_cv2
            # read the image the tag was actually found in, or the button
            # ellipse would be measured against the wrong pixels
            image_bgr = _get_cv2().imread(tag_source)
            if image_bgr is not None:
                tag_result = read_tag(image_bgr, tag_bbox)
                scale_factor, scale_confidence = scale_factor_from(tag_result)
        except Exception as exc:  # never let the ruler kill the run
            tag_result = {"identity": None, "scale": None,
                          "refused_reason": f"tag_scale_failed: {exc}"}

    # ---- Stage 4a: Pose ----------------------------------------------------
    # 22 of the 41 canonical joints are trained; the udder and teat landmarks
    # were never annotated and come back at confidence 0.0, so measurement
    # refuses those traits rather than guessing them.
    keypoints = None
    pose_reason = None
    derived_joints = {}
    try:
        keypoints = extract_keypoints(side_img, animal_bbox)
        if usable_joint_count(keypoints) == 0:
            keypoints = None
            pose_reason = "pose_no_usable_joints"
        else:
            # Some canonical joints were never annotated, so the model cannot
            # detect them. chest_bottom - the brisket - is the costly one: it
            # gates body_depth, chest_depth and chest_width_to_depth_ratio.
            # It is derivable from the silhouette, because on a side view it
            # is simply where the body ends and only legs continue below.
            # Derivation refuses on a weak pose or an implausible result
            # rather than guessing, and every derived point is marked.
            try:
                from ml.detection.detector import segment_animal
                mask, degraded = segment_animal(side_img, animal_bbox)
                if not degraded:
                    keypoints, derived_joints = add_derived_landmarks(
                        keypoints, mask, animal_bbox)
            except Exception:
                pass          # segmentation is a bonus, never a blocker

            # ---- the REAR photo -------------------------------------------
            # Ten of the eleven still-blocked traits are udder or teat traits
            # and every one of them is view "rear". The app captures a rear
            # photo; the pipeline was never running pose on it, so those
            # joints could not appear even once they are annotated.
            #
            # Rear coordinates live in the REAR image's frame. Every udder
            # trait is measured entirely within that view, so that is fine -
            # but they must never be mixed into a side-view distance, which
            # is why only REAR_VIEW_JOINTS are merged.
            try:
                rear_det = detect_animal(rear_img)
                if rear_det is not None:
                    rear_kps = extract_keypoints(rear_img, rear_det.bbox)
                    rear_mask = None
                    try:
                        from ml.detection.detector import segment_animal
                        rm, rdeg = segment_animal(rear_img, rear_det.bbox)
                        rear_mask = None if rdeg else rm
                    except Exception:
                        pass
                    keypoints, rear_prov = add_rear_view_landmarks(
                        keypoints, rear_kps, rear_mask, rear_det.bbox)
                    derived_joints.update(rear_prov)
            except Exception:
                pass          # the rear view is additive, never a blocker
    except PoseBackendError as exc:
        pose_reason = f"pose_unavailable: {exc}"

    # ---- Stage 4b: breed / group verification ------------------------------
    # Additive and never blocking: if this fails the animal is still scored.
    # The exact-breed head is disabled by the checkpoint itself (38.1%
    # source-held-out with uninformative confidence), so breed_verified stays
    # null and the group call - 80.2% against a 60.7% background control -
    # is what actually gets reported.
    breed_fields = to_contract_fields(None)
    try:
        breed_fields = to_contract_fields(verify_breed(
            side_img, animal_bbox,
            claimed_breed=animal_record.get("breed")))
    except BreedBackendError:
        pass
    # The BPA record is authoritative for species - it is a registry entry,
    # and the model is 88.5%. So the model's call is a CHECK, never an
    # override: scoring a buffalo on the cattle rubric because a classifier
    # disagreed with the registry would corrupt every trait at once.
    predicted_species = breed_fields.get("predicted_species")
    breed_fields["species_consistent"] = (
        None if predicted_species is None else predicted_species == species)

    # ---- Stage 7: Vet Pre-Screen -------------------------------------------
    # Wired, but emits no symptoms - there is no trained detector, and a
    # symptom here is not a display value: it feeds risk_report, which
    # produces "refer to vet". What it DOES report is that a gait video was
    # supplied and not analysed, because a silent empty screen would read as
    # a clean bill of health.
    screen_result = screen(keypoints=keypoints, video_path=video_path,
                           species=species)
    symptom_vectors = symptom_vector_or_empty(screen_result) or None

    if keypoints is None:
        reasons = []
        if not quality_passed:
            for label, q in (("rear", rear_q), ("video", video_q)):
                if q is not None and not q["passed"]:
                    reasons.append(f"{label}_quality_failed: " + ", ".join(q["reasons"]))
        reasons.append(pose_reason or "pose_estimation_unavailable")
        # Carry the screening notes here too. This is the path where they
        # matter most: nothing was scored AND the farmer's gait video went
        # unused, and they should learn both at once rather than neither.
        reasons.extend(screening_notes(screen_result))
        return _build_not_scored(animal_id, species, "; ".join(reasons),
                                 quality_passed, captured,
                                 breed_fields=breed_fields)

    # ---- Stages 5-6: Measurement -> Scoring -> Status -----------------------
    measurements: List[MeasurementResult] = measure_all_traits(
        keypoints,
        scale_factor,
        species,
        scale_confidence,
    )
    scores = score_all_traits(measurements, species)
    eligibility = scoreability(measurements, species, quality_passed=quality_passed)
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
        extra_warnings=(explainability.get("text_summary", [])
                        + screening_notes(screen_result)),
    )
    # THE ACTUAL FIX: return the contract-shaped dict, not the internal shape.
    # Breed fields stay None until Tag Intelligence (Stage 3) is implemented -
    # that's honest, not a bug: we haven't verified breed, so we don't claim to.
    contract = to_contract_dict(
        result,
        measurements,
        keypoints,
        breed_registered=animal_record.get("breed"),
        breed_verified=breed_fields.get("breed_verified"),
        breed_verify_confidence=breed_fields.get("breed_verify_confidence"),
        captured=captured,
    )
    # Group/species verification rides alongside the frozen fields rather
    # than changing them. These keys are ADDITIVE: the app ignores unknown
    # keys safely today, and can render them once Person 1 is ready. That is
    # what lets us ship an honest breed signal without a contract change.
    contract.update(_breed_extra(breed_fields))
    return contract


def _breed_extra(breed_fields: Optional[Dict[str, object]]) -> dict:
    """The additive verification keys, with None for anything unmeasured.

    Always the same key set, whether or not the model ran - a field that
    sometimes exists and sometimes does not is far harder for the app to
    handle than one that is reliably present and sometimes null.
    """
    f = breed_fields or {}
    return {
        "predicted_species": f.get("predicted_species"),
        "species_confidence": f.get("species_confidence"),
        "species_consistent": f.get("species_consistent"),
        "predicted_group": f.get("predicted_group"),
        "group_confidence": f.get("group_confidence"),
        "group_consistent": f.get("group_consistent"),
        "group_reliable": f.get("group_reliable"),
        "breed_verify_note": f.get("breed_verify_note"),
    }


def _build_not_scored(
    animal_id: Optional[str],
    species: str,
    reason: str,
    quality_passed: bool,
    captured: Dict[str, bool],
    breed_fields: Optional[Dict[str, object]] = None,
) -> dict:
    """Build a contract-shaped NOT_SCORED result for degraded/unimplemented runs.

    `reason` is threaded into every trait's not_scored_reason via
    to_contract_dict()'s global_not_scored_reason - the contract has no
    top-level status/warnings field for pipeline mode, so this is the only
    honest place to surface why nothing could be measured.
    """
    eligibility = scoreability(
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
    contract = to_contract_dict(
        result,
        measurements=[],
        keypoints={},
        captured=captured,
        global_not_scored_reason=reason,
    )
    contract.update(_breed_extra(breed_fields))
    return contract