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
    facing_sign,
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
from ml.common.schemas import MeasurementResult, WeightResult
from ml.weight.estimator import estimate_weight

KEYPOINT_CONFIDENCE_MIN = 0.3



# Heart girth is the circumference of the chest behind the fore leg. It was
# classed SMAL and refused because a flat 2D chord is NOT a circumference -
# feeding chest depth straight into Schaeffer's formula was a real bug once,
# and the trait was correctly disabled rather than allowed to lie.
#
# The ellipse model answers it without a SMAL mesh, by the route that note was
# really asking for: the chest station is a CLOSED cross-section built from
# both photographs - depth from the side, width from the rear - so its
# perimeter is a circumference rather than a chord. It is an approximation of
# a real chest, which is flatter over the back than an ellipse, and that is
# what the uncertainty below is for.
HEART_GIRTH_MODEL_ERROR = 0.10      # the elliptical cross-section assumption
HEART_GIRTH_SCALE_ERROR = 0.04      # tag scale, per TAG_SCALE_MIN_ERROR_FRAC


def _with_heart_girth(measurements, vol):
    """Replace the SMAL refusal for heart_girth with the ellipse measurement."""
    import math
    girth = float(vol.heart_girth_cm)
    rel = math.sqrt(HEART_GIRTH_MODEL_ERROR ** 2 + HEART_GIRTH_SCALE_ERROR ** 2)
    replacement = MeasurementResult(
        trait_id="heart_girth",
        trait_class="C",          # a scaled distance now, no longer SMAL
        value=girth,
        unit="cm",
        confidence=0.5,
        flags=[],
        uncertainty=girth * rel,
    )
    return [replacement if m.trait_id == "heart_girth" else m
            for m in measurements]


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

    if tag_bbox is None and tag_from_closeup:
        # The detector is a single point of failure for the entire scale path,
        # and it misses tags that are plainly there: on a 2560x1700 photograph
        # with a legible yellow tag it returns nothing, and still nothing with
        # the threshold dropped to 0.02 - the class does not fire at all.
        # Everything measured in centimetres depends on this one call.
        #
        # On a CLOSE-UP that call is not needed. The photograph is of the tag;
        # the frame IS the region of interest. Handing the ruler the whole
        # frame is safe because the ruler is the real validator - it gates on
        # the panel's height against the published 55-69 mm, on the round
        # button, and on the 10/10/18 mm printed rows, and refuses anything
        # that is not a conformant NDDB tag. A wrong box produces a refusal,
        # not a wrong scale.
        #
        # Deliberately NOT done for the side photograph, where the tag is a
        # small part of a large frame and the whole frame would be meaningless.
        tag_bbox = tag_animal

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
    closeup_scale = None
    closeup_err = 0.056
    # The error fraction of whichever scale is finally used. Every centimetre
    # trait's interval is proportional to it, so reporting the default when a
    # looser scale was actually used would understate all of them.
    from ml.measurement.traits import DEFAULT_SCALE_ERROR_FRAC
    scale_err = DEFAULT_SCALE_ERROR_FRAC
    if tag_bbox is not None:
        try:
            from ml.detection.detector import _get_cv2
            # read the image the tag was actually found in, or the button
            # ellipse would be measured against the wrong pixels
            image_bgr = _get_cv2().imread(tag_source)
            if image_bgr is not None:
                tag_result = read_tag(image_bgr, tag_bbox)
                scale_factor, scale_confidence = scale_factor_from(tag_result)
                if scale_factor is not None:
                    scale_err = float((tag_result.get("scale") or {}).get(
                        "error_frac", DEFAULT_SCALE_ERROR_FRAC))

                if tag_from_closeup and scale_factor is not None:
                    # A SCALE BELONGS TO THE PHOTOGRAPH IT WAS MEASURED IN.
                    #
                    # The close-up and the side photograph are different shots
                    # from different distances, so their centimetres-per-pixel
                    # differ - typically by a factor of tens, because the tag
                    # fills one frame and is a thumbnail in the other. Every
                    # keypoint measured below is in SIDE-photo pixels. Handing
                    # them the close-up's scale would multiply every class-C
                    # trait by that factor, silently, and the numbers would
                    # still look like plausible centimetres.
                    #
                    # Nothing downstream could catch it: the plausibility
                    # guards would reject the absurd ones and the survivors
                    # would be quietly wrong. So the close-up's scale is kept
                    # for what it legitimately describes - that this is a
                    # conformant tag, and its own measurements - and is NOT
                    # applied to the body.
                    #
                    # It CAN be carried across, but not from here: the
                    # transfer needs to know which way the animal is facing so
                    # it looks for the tag at the HEAD end, and pose has not
                    # run yet. Searching the whole animal would let any yellow
                    # object near it become the ruler. The close-up's scale is
                    # held and used after Stage 4a.
                    closeup_scale = scale_factor
                    closeup_err = (tag_result.get("scale") or {}).get(
                        "error_frac", 0.056)
                    scale_factor, scale_confidence = None, 0.0
                    tag_result.setdefault(
                        "scale_note",
                        "measured in the close-up photo, which has its own "
                        "pixel scale - not applied to body measurements")
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
    # Kept beyond the pose block: the two silhouettes are what the 2D -> 3D
    # weight estimator reconstructs a torso volume from, and re-segmenting for
    # it would run SAM2 a second time on the same photographs.
    side_mask = None
    rear_mask = None
    belly_y = None
    # Segmentation failing is not a crash - it falls back to the detection box
    # as a rectangle - so it stayed invisible for a long time while silently
    # disabling chest_bottom, the udder floor and the volume weight estimate.
    # Recorded so the result says so instead of just quietly measuring less.
    seg_note = None
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
                if degraded:
                    seg_note = ("segmentation_degraded: no silhouette, so "
                                "brisket-derived traits and the volume weight "
                                "estimate are unavailable")
                if not degraded:
                    side_mask = mask
                    from ml.pose_features.silhouette_landmarks import (
                        belly_line_y)
                    belly_y = belly_line_y(mask)
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

    # ---- Carrying the close-up's scale to the side photograph -------------
    # One physical object appears in both shots. The close-up gives this tag's
    # own height in centimetres - measured against the 18 mm digit row, not
    # assumed, which is what makes it usable at all when a panel is 55-69 mm
    # depending on the supplier. Finding that same tag at the head end of the
    # side photograph then gives the side photograph's own scale.
    #
    # Runs here rather than in Stage 3 because it needs the facing direction,
    # which comes from the pose.
    if closeup_scale is not None and keypoints:
        try:
            from ml.detection.detector import _get_cv2 as _cv
            from ml.tag_intelligence.panel_transfer import (TransferredScale,
                                                            transfer)
            moved = transfer(_cv().imread(side_img), animal_bbox,
                             facing_sign(keypoints, animal_bbox),
                             _cv().imread(tag_source), tag_bbox,
                             closeup_scale, closeup_error_frac=closeup_err)
            if isinstance(moved, TransferredScale):
                scale_factor = moved.cm_per_px
                # never more certain than the close-up it came from, and it
                # carries the ear's parallax on top of that
                scale_confidence = max(0.0,
                                       min(1.0, 1.0 - moved.error_frac * 4.0))
                # a transferred scale is looser than the close-up it came
                # from: it carries the close-up's error, the edge of a small
                # panel in the side photo, and the ear's parallax
                scale_err = float(moved.error_frac)
                if tag_result is not None:
                    tag_result["scale_note"] = moved.note
                    tag_result["scale_method"] = moved.method
            elif tag_result is not None:
                tag_result["scale_note"] = getattr(
                    moved, "reason", "scale could not be transferred")
        except Exception as exc:
            if tag_result is not None:
                tag_result["scale_note"] = f"scale_transfer_failed: {exc}"

    # ---- Stages 5-6: Measurement -> Scoring -> Status -----------------------
    measurements: List[MeasurementResult] = measure_all_traits(
        keypoints,
        scale_factor,
        species,
        scale_confidence,
        scale_err,
    )
    scores = score_all_traits(measurements, species)
    eligibility = scoreability(measurements, species, quality_passed=quality_passed)
    status = determine_status(eligibility, measurements)
    explainability = assemble_explainability(measurements, scores, keypoints)
    # ---- Weight: 2D -> 3D torso volume, falling back to girth-length ------
    # Two orthogonal silhouettes are two projections of one solid, so stacking
    # elliptical cross-sections along the body recovers a volume. The rear
    # photograph contributes only a width-to-depth RATIO, which is
    # dimensionless and therefore survives having been taken from a different
    # distance than the side photograph that carries the tag scale.
    #
    # The older girth-length route stays as the fallback, and is recomputed
    # inside the volume estimator as an independent cross-check - two methods
    # agreeing is evidence, one alone is a guess.
    weight_result = estimate_weight(measurements)
    if keypoints and side_mask is not None:
        try:
            from ml.weight.volume_3d import estimate as estimate_volume
            vol = estimate_volume(
                side_mask, rear_mask, keypoints,
                cm_per_px=scale_factor, belly_y=belly_y)
            if vol.heart_girth_cm is not None:
                measurements = _with_heart_girth(measurements, vol)
                scores = score_all_traits(measurements, species)
            if vol.measured:
                weight_result = WeightResult(
                    estimate_kg=0.5 * (vol.low_kg + vol.high_kg),
                    range_kg=(vol.low_kg, vol.high_kg),
                    # The interval is dominated by an assumed density range and
                    # by the CUBE of the tag-scale error, neither calibrated
                    # against weighed animals. Held well below the measured
                    # stages so nothing downstream treats it as solid.
                    confidence=0.45,
                    method=vol.method,
                    cross_check=vol.cross_check,
                )
        except Exception:
            pass          # the volume route is additive, never a blocker

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
                        + screening_notes(screen_result)
                        + ([seg_note] if seg_note else [])),
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
    contract.update(_breed_extra(
        breed_fields, vet_screened=bool(screen_result.get("screened", False))))
    return contract


def _breed_extra(breed_fields: Optional[Dict[str, object]],
                 vet_screened: bool = False) -> dict:
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
        # Whether a veterinary screening actually ran. This build has no
        # trained symptom detector, so it never does - and an empty
        # symptom_vector therefore means NOT SCREENED, not healthy. Without
        # this flag the reports had no way to tell those apart, and said
        # "No health problems were flagged" to a farmer who had not been
        # screened at all.
        "vet_screened": bool(vet_screened),
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