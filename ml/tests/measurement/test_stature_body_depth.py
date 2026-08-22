"""Regression test for Stature / Body Depth using vertical distance
(Rev 2 audit item 9 / implementation item 13).

Stature and Body Depth are height measurements and must use vertical-only
(y-axis) pixel distance, not full Euclidean distance - Euclidean overstates
height whenever the two keypoints aren't perfectly vertically aligned (any
camera roll, or the lower point not sitting directly under the upper one).
Body Length is a genuine diagonal/straight-line distance and must be
unaffected - it still uses full Euclidean distance.

NOTE: this does not implement camera roll/tilt correction - see
_pixel_vertical_distance's docstring in ml/measurement/traits.py. That
remains BLOCKED pending a "level" reference signal that does not exist
anywhere in this pipeline yet.

NOTE: ml/config/rules.py's calibration bins for stature/body_depth were
deliberately left unchanged - this formula change will generally produce
smaller values than before (vertical distance <= Euclidean distance), which
may need re-calibration once real photos are available. Flagged, not
guessed at here.
"""

import math

from ml.measurement.traits import measure_trait


def _kp(x: float, y: float, confidence: float = 1.0):
    return (x, y, confidence)


def test_stature_uses_vertical_distance_only():
    kp = {"withers": _kp(300.0, 100.0), "hoof_left": _kp(340.0, 500.0)}
    scale = 0.3
    result = measure_trait("stature", kp, scale_factor=scale)
    expected_vertical = abs(500.0 - 100.0) * scale
    expected_euclidean = math.hypot(340.0 - 300.0, 500.0 - 100.0) * scale
    assert result.value == expected_vertical
    assert result.value < expected_euclidean


def test_body_depth_uses_vertical_distance_only():
    kp = {"withers": _kp(300.0, 100.0), "chest_bottom": _kp(330.0, 300.0)}
    scale = 0.3
    result = measure_trait("body_depth", kp, scale_factor=scale)
    expected_vertical = abs(300.0 - 100.0) * scale
    expected_euclidean = math.hypot(330.0 - 300.0, 300.0 - 100.0) * scale
    assert result.value == expected_vertical
    assert result.value < expected_euclidean


def test_body_length_still_uses_euclidean_distance():
    """Body Length is a genuine diagonal distance, not a height - must be
    unaffected by the stature/body_depth-specific vertical-distance change."""
    kp = {"chest_front": _kp(100.0, 200.0), "pin_right": _kp(500.0, 220.0)}
    scale = 0.3
    result = measure_trait("body_length", kp, scale_factor=scale)
    expected_euclidean = math.hypot(500.0 - 100.0, 220.0 - 200.0) * scale
    assert result.value == expected_euclidean
