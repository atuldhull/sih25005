"""Paths and thresholds for the pose, tag-scale and breed models.

Same convention as config/detection.py: paths are relative to the repo root,
and the weights themselves are NOT in git. See models/README.md - they are
published as Kaggle datasets and fetched with scripts/fetch_models.py, which
also verifies a SHA256 so a truncated download cannot be mistaken for a model
bug.
"""

# ---- Section 5.2 : bovine keypoints -------------------------------------
POSE_MODEL_PATH = "models/pose/bovine41_hrnet.pt"
# Below this, a joint is treated as unavailable rather than uncertain. The
# measurement layer already refuses traits whose landmarks are missing, so a
# low bar here would produce confident measurements from guessed points.
POSE_MIN_KEYPOINT_CONFIDENCE = 0.30

# ---- Section 5.3 : breed / group verification ---------------------------
BREED_MODEL_PATH = "models/breed/breed_verifier.pt"
# The verifier carries its own measured thresholds inside the checkpoint, so
# nothing is hardcoded here. What IS a policy decision is whether we consult
# the exact-breed head at all: measured at 38.1% source-held-out with
# uninformative confidence, so the model disables it itself and only the
# group and species heads speak.
BREED_ENABLED = True

# ---- Ear-tag scale ------------------------------------------------------
# The round button on an NDDB tag is 27 mm. Under projection a circle becomes
# an ellipse whose MAJOR axis is tilt-invariant to first order, which is what
# makes a scale possible without knowing the camera. The outer panel is
# 55-69 mm depending on the supplier and must never be used for scale.
TAG_SCALE_METHOD = "button_ellipse_27mm"
# Refuse to claim a scale better than this. The method is validated on
# synthetic buttons of known size; on real NDDB tags it is not yet validated
# at all, and every class-C trait inherits this error.
TAG_SCALE_MIN_ERROR_FRAC = 0.04
