"""Tests for the canonical keypoint schema and normalize_keypoints() seam."""

import sys

import pytest



from ml.pose_features.keypoint_schema import (  # noqa: E402
    KEYPOINT_NAMES,
    KEYPOINT_SCHEMA,
    MISSING_JOINT,
    get_keypoint_index,
    get_keypoint_name,
    normalize_keypoints,
)


SAMPLE = {
    "withers": (300, 120, 0.93),
    "chest_front": (160, 340, 0.91),
    "hock_left": (480, 500, 0.90),
    "hoof_right": (630, 585, 0.90),
}


# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------
class TestSchema:
    def test_schema_has_41_contiguous_indices(self):
        assert len(KEYPOINT_SCHEMA) == 41
        indices = [e["index"] for e in KEYPOINT_SCHEMA]
        assert indices == list(range(41))

    def test_names_unique_and_match_lookup_tables(self):
        assert len(set(KEYPOINT_NAMES)) == 41

    
        assert all(get_keypoint_index(n) == i for i, n in enumerate(KEYPOINT_NAMES))
        assert all(get_keypoint_name(i) == n for i, n in enumerate(KEYPOINT_NAMES))

    def test_unknown_name_and_index_raise(self):
        with pytest.raises(KeyError):
            get_keypoint_index("not_a_joint")
        with pytest.raises(KeyError):
            get_keypoint_name(99)


# ---------------------------------------------------------------------------
# identity passthrough
# ---------------------------------------------------------------------------
class TestIdentity:
    def test_passthrough_preserves_provided_joints(self):
        out = normalize_keypoints(SAMPLE, "identity")
        assert out["withers"] == (300, 120, 0.93)
        assert out["chest_front"] == (160, 340, 0.91)
        assert out["hock_left"].confidence == pytest.approx(0.90)

    def test_all_41_canonical_keys_present_and_unprovided_are_zero_conf(self):
        out = normalize_keypoints(SAMPLE, "identity")
        assert set(out) == set(KEYPOINT_NAMES)
        for name in KEYPOINT_NAMES:
            if name not in SAMPLE:
                assert out[name] == MISSING_JOINT
                assert out[name].confidence == 0.0
                assert out[name].x == 0.0 and out[name].y == 0.0

    def test_missing_joint_does_not_crash(self):
        # Only one joint supplied -> the other 23 must degrade to confidence 0.0.
        out = normalize_keypoints({"withers": (1, 2, 0.99)}, "identity")
        assert out["withers"].confidence == pytest.approx(0.99)
        assert out["pastern_left"].confidence == 0.0

    def test_unknown_source_format_raises_not_implemented(self):
        for fmt in ("rtmpose_coco_animal", "rtmpose_custom_bovine"):
            with pytest.raises(NotImplementedError, match=fmt):
                normalize_keypoints(SAMPLE, fmt)

    def test_totally_unknown_source_format_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown source_format"):
            normalize_keypoints(SAMPLE, "rtmpose_pig_20k")

    def test_unknown_keypoint_name_rejected(self):
        with pytest.raises(ValueError, match="Unknown keypoint name"):
            normalize_keypoints({"muzzle": (1, 2, 0.9)}, "identity")

    def test_bad_value_shape_rejected(self):
        with pytest.raises(ValueError, match="3-element numeric"):
            normalize_keypoints({"withers": (300, 120)}, "identity")
        with pytest.raises(ValueError, match="3-element numeric"):
            normalize_keypoints({"withers": "not-a-joint"}, "identity")

    def test_non_dict_identity_input_rejected(self):
        with pytest.raises(ValueError, match="expects a dict"):
            normalize_keypoints([(1, 2, 0.9)], "identity")


# ---------------------------------------------------------------------------
# Interop with the measurement engine (the real consumer of this seam)
# ---------------------------------------------------------------------------
class TestMeasurementInterop:
    def test_normalized_output_drives_full_measurement(self):
        from ml.measurement.traits import measure_all_traits

        full = {
            "withers": (300, 120, 0.95), "back_mid": (430, 140, 0.95),
            "chest_front": (160, 340, 0.95), "chest_bottom": (300, 380, 0.95),
            "chest_width_left": (290, 300, 0.95), "chest_width_right": (320, 300, 0.95),
            "shoulder_left": (220, 210, 0.95), "shoulder_right": (260, 210, 0.95),
            "tail_head": (600, 170, 0.95), "hip_bone_left": (520, 180, 0.95),
            "hip_bone_right": (560, 180, 0.95), "hook_left": (540, 200, 0.95),
            "hook_right": (580, 200, 0.95), "pin_left": (640, 230, 0.95),
            "pin_right": (680, 230, 0.95), "knee_left": (340, 460, 0.95),
            "knee_right": (380, 460, 0.95), "hock_left": (480, 500, 0.95),
            "hock_right": (520, 500, 0.95), "pastern_left": (560, 560, 0.95),
            "pastern_right": (600, 560, 0.95), "hoof_left": (590, 585, 0.95),
            "hoof_right": (630, 585, 0.95), "rear_udder": (610, 420, 0.95),
        }
        normalized = normalize_keypoints(full, "identity")

        # Then the rear-view merge, which is what the pipeline does next and
        # is where rear-frame landmarks come from. They are deliberately NOT
        # canonical schema names - normalize_keypoints validates against the
        # 41 the checkpoint predicts, and these are copies made at merge time,
        # not predictions - so they are added here rather than above.
        #
        # Two of the twenty traits need them. rump_width and
        # rear_legs_rear_view compare the animal's LEFT side with its RIGHT,
        # and a side photograph shows the two sides 1.7% of the animal apart,
        # on top of each other. They can only be measured from a rear view.
        normalized.update({
            "rear_hook_left": (540, 200, 0.95),
            "rear_hook_right": (600, 200, 0.95),
            "rear_hip_bone_left": (520, 180, 0.95),
            "rear_hip_bone_right": (590, 180, 0.95),
            "rear_hock_left": (480, 500, 0.95),
            "rear_hock_right": (560, 500, 0.95),
        })
        measurements = measure_all_traits(normalized, 0.05, "cattle", 0.9)
        assert sum(1 for m in measurements if m.value is not None) == 20

    def test_zero_conf_joint_marks_dependent_traits_not_measurable(self):
        from ml.measurement.traits import measure_all_traits

        partial = {
            "withers": (300, 120, 0.95), "chest_bottom": (300, 380, 0.95),
            "chest_front": (160, 340, 0.95), "back_mid": (430, 140, 0.95),
            "shoulder_left": (220, 210, 0.95), "shoulder_right": (260, 210, 0.95),
            "hook_left": (540, 200, 0.95), "hook_right": (580, 200, 0.95),
            "pin_left": (640, 230, 0.95), "pin_right": (680, 230, 0.95),
            "chest_width_left": (290, 300, 0.95), "chest_width_right": (320, 300, 0.95),
            "rear_udder": (610, 420, 0.95),
            # left-side leg joints intentionally absent -> confidence 0.0
        }
        normalized = normalize_keypoints(partial, "identity")
        measurements = measure_all_traits(normalized, 0.05, "cattle", 0.9)
        hock_angle = next(m for m in measurements if m.trait_id == "hock_angle")
        assert hock_angle.value is None
        assert "not_measurable" in hock_angle.flags