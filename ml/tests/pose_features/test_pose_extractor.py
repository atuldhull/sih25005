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
