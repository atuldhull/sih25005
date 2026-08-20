"""Regression test for tag spec values (Rev 2 audit item 6 / implementation item 7).

TAG_SPECS must use the invariant printed features (button/barcode/digit-line)
as the scale reference, never the outer panel (which varies 55-69mm by
manufacturer). CAMERA_MODEL must be derived at runtime from actual image
dimensions, not the old hardcoded 700/400/300 (which implied an 800x600
image against real 3000-4000px phone photos - roughly a 4-5x scale error).
"""

from ml.config.tag_spec import TAG_SPECS, get_camera_model


def test_tag_specs_use_printed_features_not_panel():
    v1 = TAG_SPECS["v1"]
    assert v1["button_diameter_cm"] == 2.7
    assert v1["barcode_line_cm"] == 1.0
    assert v1["digit_line_cm"] == 1.8
    # Panel dims are kept ONLY for corner detection, never as the scale spec.
    assert "width_cm" not in v1
    assert "height_cm" not in v1


def test_camera_model_derives_from_actual_image_size():
    model = get_camera_model(4000, 3000)
    assert model["focal_length_px"] == 4000.0
    assert model["principal_point_cx"] == 2000.0
    assert model["principal_point_cy"] == 1500.0

    # Must scale with the image, not be pinned to the old 800x600 placeholder.
    small = get_camera_model(800, 600)
    assert small["focal_length_px"] == 800.0
    assert small != model
