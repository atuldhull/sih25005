"""No trait may measure across two coordinate frames.

The keypoint dict handed to measurement holds landmarks from BOTH photographs.
Side-view joints are in the side photo's pixels; anything merged from the rear
photo is in the rear photo's pixels. The two shots were taken from different
distances, so a distance computed between one of each is meaningless.

It does not raise. udder_depth paired udder_floor (rear) with hock_left (side)
and produced 74.7 cm against a calibrated band of -10 to 25 - a units error
wearing the shape of a remarkable udder. udder_depth_ratio did the same across
a ratio's numerator and denominator.

The merge already carried a comment saying rear coordinates "must never be
mixed into a side-view distance". What made that safe was an assumption
alongside it - that every udder trait is measured entirely within the rear
view - and it was not true. This test replaces the assumption.
"""
import pytest

from ml.config.traits import TRAIT_REGISTRY
from ml.pose_features.silhouette_landmarks import (REAR_FRAME_ALIASES,
                                                   REAR_VIEW_JOINTS)

REAR_FRAME = set(REAR_VIEW_JOINTS) | set(REAR_FRAME_ALIASES.values())


@pytest.mark.parametrize("trait", TRAIT_REGISTRY,
                         ids=[t["trait_id"] for t in TRAIT_REGISTRY])
def test_a_trait_uses_one_frame_or_the_other(trait):
    joints = trait.get("required_keypoints", [])
    rear = [j for j in joints if j in REAR_FRAME]
    side = [j for j in joints if j not in REAR_FRAME]
    assert not (rear and side), (
        f"{trait['trait_id']} mixes coordinate frames: {rear} come from the "
        f"rear photograph and {side} from the side. A distance between them "
        f"is measured across two shots taken at two distances.")


@pytest.mark.parametrize("trait", [t for t in TRAIT_REGISTRY
                                   if t.get("view") == "rear"],
                         ids=[t["trait_id"] for t in TRAIT_REGISTRY
                              if t.get("view") == "rear"])
def test_a_rear_view_trait_uses_only_rear_frame_landmarks(trait):
    """A trait declared 'rear' that reads side-view landmarks is not measuring
    what its name says. rear_legs_rear_view compared the animal's left side
    with its right using points 1.7% of the animal apart in a side view."""
    stray = [j for j in trait.get("required_keypoints", [])
             if j not in REAR_FRAME]
    assert not stray, (
        f"{trait['trait_id']} is a rear-view trait reading side-view "
        f"landmarks: {stray}")


def test_every_alias_maps_to_a_real_side_view_joint():
    """An alias for a joint that does not exist would silently never merge."""
    from ml.pose_features.keypoint_schema import KEYPOINT_INDEX_BY_NAME
    for src, alias in REAR_FRAME_ALIASES.items():
        assert src in KEYPOINT_INDEX_BY_NAME, f"{src} is not a schema joint"
        assert alias.startswith("rear_"), (
            f"{alias} must be visibly rear-scoped, or the next person will "
            f"pair it with a side-view landmark")
        assert alias not in KEYPOINT_INDEX_BY_NAME, (
            f"{alias} is in the schema - aliases are copies made at merge "
            f"time, not predictions, and the schema mirrors the checkpoint")
