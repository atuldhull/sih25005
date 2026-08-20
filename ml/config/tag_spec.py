"""Ear-tag physical specification (versioned) and identity-resolution config.

Tag dimensions are real-world spec, versioned per architecture Section 11.4 so a
design change only requires adding a new version entry. The camera model is a
placeholder intrinsic guess pending calibration.
"""

DEFAULT_TAG_VERSION = "v1"

# Physical tag dimensions in cm (versioned).
#
# button_diameter_cm / barcode_line_cm / digit_line_cm are the invariant
# printed-feature measurements used for the actual solvePnP scale reference.
# These do NOT vary by tag manufacturer, unlike the outer panel below.
#
# panel_width_cm / panel_height_cm are kept ONLY for corner detection / pose
# framing (locating the tag region in the image) - they must NEVER be used
# as the scale reference, since the outer panel size varies by manufacturer
# (55-69mm per the audit) while the printed features do not.
TAG_SPECS = {
    "v1": {
        "button_diameter_cm": 2.7,
        "barcode_line_cm": 1.0,
        "digit_line_cm": 1.8,
        "panel_width_cm": 5.0,
        "panel_height_cm": 3.5,
    },
}


def get_camera_model(image_width_px: int, image_height_px: int) -> dict:
    """Derive camera intrinsics from the actual image dimensions at runtime.

    Replaces the old hardcoded 700/400/300 placeholder, which implied an
    800x600 image while real phone photos run 3000-4000px wide - solvePnP
    scale is directly proportional to focal length, so that hardcoded value
    was off by roughly 4-5x on real photos.

    The 1.0x width heuristic below is itself still a placeholder - it is
    *correct order of magnitude* (verified spec says focal length in pixels
    is roughly 0.8-1.2x image width for typical phone cameras) but is not a
    real per-device calibration. Replace with EXIF-derived focal length or
    real calibration data when available; that value was not specified
    anywhere in the repo or audit, so it is not invented here.
    """
    return {
        "focal_length_px": float(image_width_px),
        "principal_point_cx": image_width_px / 2.0,
        "principal_point_cy": image_height_px / 2.0,
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