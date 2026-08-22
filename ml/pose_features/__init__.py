"""Pose and feature extraction module.

KEYPOINT ANNOTATION DECISION
----------------------------
Every trait in the measurement engine (measurement/traits.py + config/traits.py)
is keyed off the 24 canonical anatomical joints defined in this package's
keypoint_schema module (KEYPOINT_SCHEMA, stable indices 0-23). The future
bovine dataset for the next phase MUST be annotated against exactly this schema
(24 named joints), or an explicit mapping from the annotated format onto it
(i.e. a new source_format for normalize_keypoints) must be written.

This is the design decision that governs how the training skeleton dataset gets
annotated, and it fixes the seam between the RTMPose model output and the
measurement engine before any model work starts.
"""