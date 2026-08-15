"""Ear-tag physical specification (versioned) and identity-resolution config.

Tag dimensions are real-world spec, versioned per architecture Section 11.4 so a
design change only requires adding a new version entry. The camera model is a
placeholder intrinsic guess pending calibration.
"""

DEFAULT_TAG_VERSION = "v1"

# Physical tag dimensions in cm (versioned).
TAG_SPECS = {
    "v1": {
        "width_cm": 5.0,
        "height_cm": 3.5,
    },
}

# Placeholder camera intrinsics (px), pending per-device calibration.
CAMERA_MODEL = {
    "focal_length_px": 700.0,
    "principal_point_cx": 400.0,
    "principal_point_cy": 300.0,
}

# PaddleOCR minimum confidence for an ID to be accepted.
OCR_CONFIDENCE_THRESHOLD = 0.6

# Maximum mean reprojection error (px) accepted for a solvePnP scale estimate.
MAX_REPROJECTION_ERROR_PX = 8.0

# Local mock tag->animal lookup, standing in for the Pashu Aadhaar DB until a
# real connection exists (architecture Section 16 allows unresolved identity).
MOCK_IDENTITY_DB = {
    "TAG10001": {"animal_id": "US-001", "species": "cattle", "breed": "HF-Cross"},
    "TAG10002": {"animal_id": "US-002", "species": "cattle", "breed": "Sahiwal"},
    "TAG10003": {"animal_id": "US-003", "species": "buffalo", "breed": "Murrah"},
}