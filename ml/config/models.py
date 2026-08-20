"""Paths and thresholds for the pose, tag-scale and breed models.

Same convention as config/detection.py: paths are relative to the repo root,
and the weights themselves are NOT in git. See models/README.md - they are
published as Kaggle datasets and fetched with scripts/fetch_models.py, which
also verifies a SHA256 so a truncated download cannot be mistaken for a model
bug.
"""
from ml.config.detection import _at_root

# ---- Section 5.2 : bovine keypoints -------------------------------------
POSE_MODEL_PATH = _at_root("models/pose/bovine41_hrnet.pt")
# Below this, a joint is treated as unavailable rather than uncertain.
#
# Chosen from a measured sweep, not from feel. Re-gating 55 photographs at a
# range of thresholds and counting how many trait measurements come out, and
# how many of those land inside their calibrated rule band:
#
#     gate   usable joints   measurements   in-band rate
#     0.30        6.5              41           34.1%
#     0.25        7.5              46           39.1%
#     0.20        8.7              62           35.5%
#     0.15       10.8              86           37.2%
#     0.10       14.2             142           35.9%
#     0.05       15.3             167           34.1%
#
# The in-band rate is the control. If a looser gate were admitting misplaced
# joints, the extra measurements would scatter and that rate would fall toward
# chance. It does not move - so the measurements a strict gate was throwing
# away were as plausible as the ones it kept, and there were three times as
# many of them. 0.05 is where the rate finally starts to slip, so the knee at
# 0.10 is taken.
#
# What makes this safe is that a low-confidence joint is not an unmeasured
# one: bovine_pose_infer attaches err_frac per joint, scaled by that joint's
# historical PCK, so a weak landmark reports a wider interval rather than a
# false certainty. The gate decides what is unusable, not what is uncertain.
POSE_MIN_KEYPOINT_CONFIDENCE = 0.10

# ---- Section 5.3 : breed / group verification ---------------------------
BREED_MODEL_PATH = _at_root("models/breed/breed_verifier.pt")
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
