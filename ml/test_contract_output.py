"""Temporary scratch test: measurement -> scoring -> to_contract_dict() -> validate().

Run from inside the ml/ folder (same convention as test_full_pipeline.py):
    cd ml
    py test_contract_output.py
"""
import os
import sys

# contract/validate_result.py lives one level up from ml/, at the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "contract"))
from validate_result import validate  # noqa: E402

from ml.measurement.traits import measure_all_traits
from ml.scoring.scorer import scoreability, determine_status, score_all_traits
from ml.explainability.result_builder import build_scoring_result, to_contract_dict
from ml.weight.estimator import estimate_weight
from ml.config.traits import TRAIT_REGISTRY

SPECIES = "cattle"
SCALE_FACTOR = 0.05
SCALE_CONFIDENCE = 0.9

keypoints = {
    "withers": (300, 120, 0.93),
    "chest_bottom": (300, 380, 0.90),
    "chest_front": (160, 340, 0.91),
    "back_mid": (430, 140, 0.88),
    "tail_head": (600, 170, 0.85),  # registry's BCS trait now correctly requires
                                     # "tail_head" (underscore) matching this schema key.
    "shoulder_left": (220, 210, 0.90),
    "shoulder_right": (260, 210, 0.89),
    "hip_bone_left": (520, 180, 0.92),
    "hip_bone_right": (560, 180, 0.91),
    "hook_left": (540, 200, 0.93),
    "hook_right": (580, 200, 0.92),
    "pin_left": (640, 230, 0.94),
    "pin_right": (680, 230, 0.93),
    "knee_left": (340, 460, 0.87),
    "knee_right": (380, 460, 0.86),
    "hock_left": (480, 500, 0.90),
    "hock_right": (520, 500, 0.89),
    "pastern_left": (560, 560, 0.12),
    "pastern_right": (600, 560, 0.88),
    "hoof_left": (590, 585, 0.91),
    "hoof_right": (630, 585, 0.90),
    "chest_width_left": (290, 300, 0.05),
    "chest_width_right": (320, 300, 0.90),
    "rear_udder": (610, 420, 0.92),
    # NOTE: no udder/teat/rib keypoints provided - the 11 new contract traits that
    # need them (Udder x8, Angularity, Body Condition Score) will correctly come
    # back as not_measurable / not_scored below. That's expected right now.
}

needed = sorted({kp for t in TRAIT_REGISTRY for kp in t["required_keypoints"]})
missing = [k for k in needed if k not in keypoints]
print(f"keypoints provided: {len(keypoints)} | missing from registry: {len(missing)}\n")

measurements = measure_all_traits(keypoints, SCALE_FACTOR, SPECIES, SCALE_CONFIDENCE)
scores = score_all_traits(measurements, SPECIES)
eligibility = scoreability(measurements, SPECIES, quality_passed=True)
status = determine_status(eligibility, measurements)
weight_result = estimate_weight(measurements)
print(
    f"weight estimate: {weight_result.estimate_kg} kg "
    f"(range {weight_result.range_kg}, confidence {weight_result.confidence:.2f})"
)
# NOTE (REVIEW-ml-dev.md, Fix #3): heart_girth here is really a 2D chest-depth
# proxy, not true girth circumference - so this weight estimate is expected to
# undershoot a real animal's weight until that fix is applied. Not corrected in
# this test script; flagged so the number below isn't mistaken for accurate.

result = build_scoring_result(
    animal_id="356279812345",
    species=SPECIES,
    measurements=measurements,
    scores=scores,
    eligibility=eligibility,
    status=status,
    weight_result=weight_result,
)

contract_dict = to_contract_dict(
    result,
    measurements,
    keypoints,
    breed_registered="Gir",
    breed_verified=None,
    breed_verify_confidence=None,
    captured={"side_photo": True, "rear_photo": True, "gait_video": False},
)

print(f"internal status: {status}")
print(f"traits emitted in contract dict: {len(contract_dict['traits'])} (expect 20)\n")

problems = validate(contract_dict, mode="pipeline")

if not problems:
    print("VALIDATION: no problems found.")
else:
    print(f"VALIDATION: {len(problems)} problem(s) found:\n")
    for p in problems:
        print(f"  - {p}")