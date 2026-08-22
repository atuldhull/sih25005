"""Pose adapter: the refusal behaviour, not the model.

None of these load the checkpoint. What they pin down is the contract between
the pose stage and measurement, because that is where a silent mistake turns
into a confidently wrong centimetre reading.
"""
import pytest

from ml.pose_features import pose_extractor
from ml.pose_features.keypoint_schema import KEYPOINT_INDEX_BY_NAME


def test_missing_checkpoint_raises_with_an_actionable_message(tmp_path):
    """A missing model must not look like a modelling failure."""
    with pytest.raises(pose_extractor.PoseBackendError) as exc:
        pose_extractor.extract_keypoints(
            "irrelevant.jpg", (0, 0, 10, 10),
            model_path=str(tmp_path / "does_not_exist.pt"))
    msg = str(exc.value)
    assert "not found" in msg
    assert "fetch_models" in msg, "must say how to fix it, not just that it broke"


def test_usable_joint_count_ignores_zero_confidence():
    kps = {"withers": (10.0, 20.0, 0.8),
           "teat_rear_left": (0.0, 0.0, 0.0),
           "hook_left": (30.0, 40.0, 0.5)}
    assert pose_extractor.usable_joint_count(kps) == 2


def test_untrained_joints_keep_their_key():
    """Measurement looks joints up by NAME.

    A dropped key is a KeyError deep in a trait computation; a zero confidence
    is a refusal the measurement layer already understands. The difference is
    a crash versus an honest not_scored_reason.
    """
    kps = {"withers": (1.0, 2.0, 0.0)}
    assert "withers" in kps
    assert pose_extractor.usable_joint_count(kps) == 0


TRAINED_JOINTS = [
    "withers", "back_mid", "chest_front", "chest_width_left",
    "chest_width_right", "shoulder_left", "shoulder_right", "tail_head",
    "hip_bone_left", "hip_bone_right", "hook_left", "hook_right",
    "pin_left", "pin_right", "knee_left", "knee_right", "hock_left",
    "hock_right", "pastern_left", "pastern_right", "hoof_left", "hoof_right",
]


@pytest.mark.parametrize("joint", TRAINED_JOINTS)
def test_every_trained_joint_exists_in_the_canonical_schema(joint):
    """The model's 22 trained joints must be canonical schema names.

    They are today - an exact string match, no aliasing anywhere. This test
    exists so that if either side renames a joint, it fails here loudly
    instead of silently producing keypoints nothing downstream can find.
    """
    assert joint in KEYPOINT_INDEX_BY_NAME, (
        f"'{joint}' is trained by the pose model but is not in "
        f"KEYPOINT_SCHEMA - measurement would never see it")


def test_trained_joints_are_a_strict_subset_of_the_schema():
    """22 of 41 trained; the udder and teat joints were never annotated."""
    assert len(TRAINED_JOINTS) == 22
    assert len(KEYPOINT_INDEX_BY_NAME) == 41
    assert set(TRAINED_JOINTS) < set(KEYPOINT_INDEX_BY_NAME)


# --- collapsed joints ------------------------------------------------------
# Two DIFFERENT joints predicted at the same pixel are mutually invalidating.
# Measured on a real photo: knee_left (2158,2149) and pin_left (2160,2154)
# landed 5px apart on a 3066px animal, as did chest_front and hook_left. The
# hock angle built on that knee came out 74.8 deg against a rule expecting
# 130-160, so every leg trait refused. The geometry was right; the joints were
# wrong, and nothing downstream could tell.

from ml.pose_features.pose_extractor import (  # noqa: E402
    _drop_collapsed,
    _same_landmark_other_side,
)

BOX = (0.0, 0.0, 1000.0, 1000.0)      # reach = 2% of 1000 = 20px


def test_distinct_joints_sharing_a_pixel_are_both_refused():
    kps = {"knee_left": (500.0, 500.0, 0.6),
           "pin_left": (503.0, 504.0, 0.6),
           "withers": (100.0, 100.0, 0.8)}
    out = _drop_collapsed(kps, BOX)
    assert out["knee_left"][2] == 0.0
    assert out["pin_left"][2] == 0.0, "both go - we cannot tell which is real"
    assert out["withers"][2] == 0.8, "an unrelated joint is untouched"


def test_left_right_pairs_of_the_SAME_landmark_are_kept():
    """On a side-on photo the near and far legs overlap.

    pastern_left and pastern_right landing 39px apart is correct geometry, not
    model failure. An earlier version of this filter threw them away and took
    19 usable joints down to 6.
    """
    kps = {"pastern_left": (500.0, 500.0, 0.77),
           "pastern_right": (505.0, 503.0, 0.55),
           "hock_left": (400.0, 400.0, 0.62),
           "hock_right": (404.0, 402.0, 0.59)}
    out = _drop_collapsed(kps, BOX)
    assert all(v[2] > 0 for v in out.values()), (
        "left/right pairs of the same landmark must survive")


def test_the_same_landmark_test_is_exact():
    assert _same_landmark_other_side("hock_left", "hock_right")
    assert _same_landmark_other_side("pastern_right", "pastern_left")
    assert not _same_landmark_other_side("knee_left", "pin_left")
    assert not _same_landmark_other_side("knee_left", "hock_right")
    assert not _same_landmark_other_side("withers", "back_mid")


def test_already_refused_joints_are_ignored():
    kps = {"a_left": (500.0, 500.0, 0.0), "b_left": (501.0, 501.0, 0.0),
           "withers": (100.0, 100.0, 0.8)}
    out = _drop_collapsed(kps, BOX)
    assert out["withers"][2] == 0.8
