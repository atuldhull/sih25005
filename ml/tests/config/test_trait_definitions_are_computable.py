"""Every trait definition must be able to produce a number.

Four of the twenty contract traits were defined as class B - a RATIO of two
distances - with only two required keypoints. _compute_ratio needs four, so it
returned None on every image, forever. Those traits could never have measured,
no matter how good the landmarks were.

All four were udder traits waiting on annotation, which is what makes this
worth a test rather than a fix: nearly half of that labelling effort would
have produced nothing, and nobody would have known why, because "not
measurable" is exactly what a trait waiting on annotation looks like.
"""
import pytest

from ml.config.traits import CONTRACT_TRAITS, TRAIT_REGISTRY


@pytest.mark.parametrize("trait", TRAIT_REGISTRY,
                         ids=[t["trait_id"] for t in TRAIT_REGISTRY])
def test_the_keypoint_count_can_produce_the_trait_class(trait):
    n = len(trait.get("required_keypoints", []))
    cls = trait["trait_class"]
    if cls == "B":
        assert n >= 4, (
            f"{trait['trait_id']} is a ratio of two distances but has {n} "
            f"keypoints; _compute_ratio returns None below four, so this "
            f"trait can never measure")
    elif cls == "A":
        assert n in (2, 3, 4), (
            f"{trait['trait_id']} is an angle with {n} keypoints; "
            f"_compute_angle handles 2 or 3, and 4 reduces to 2")
    elif cls == "C":
        assert n >= 2, (
            f"{trait['trait_id']} is a distance with {n} keypoints")


# Landmarks a trait asks for that KEYPOINT_SCHEMA does not define. These are
# not an annotation backlog - they are a gap nobody has specified. A landmark
# absent from the schema can never be produced by the pose model, so the trait
# refuses forever with "not_measurable", which is indistinguishable from a
# trait merely waiting to be labelled.
#
# angularity is the only one. It asks for the rib angle at three heights, and
# the schema has no rib landmarks at all. Adding them is not free: the schema's
# 41 entries correspond to the trained checkpoint's own keypoint list, so
# extending it is a change to both, and annotating a rib angle from a side
# photograph is a harder judgement than a hock or a tail head.
#
# Listed here so the gap is visible and counted, rather than hidden inside a
# refusal that looks like every other refusal.
KNOWN_SCHEMA_GAPS = {
    "angularity": ["rib_angle_top", "rib_angle_mid", "rib_angle_bottom"],
}

# Deliberately absent from the schema, for a different reason: these are
# rear-frame COPIES of joints the side view also has, made when the two views
# are merged, so that a rear-view trait can use a hock without being measured
# against the side photograph's hock in a different coordinate frame. They are
# not predicted by anything, so they do not belong in a schema whose 41 entries
# mirror the trained checkpoint's keypoint list.
VIEW_SCOPED_ALIASES = {
    "rear_hock_left", "rear_hock_right",
    "rear_hip_bone_left", "rear_hip_bone_right",
    "rear_hook_left", "rear_hook_right",
}


@pytest.mark.parametrize("trait", TRAIT_REGISTRY,
                         ids=[t["trait_id"] for t in TRAIT_REGISTRY])
def test_every_required_keypoint_exists_in_the_schema(trait):
    """A landmark the schema does not define can never arrive, so the trait
    would refuse forever with a reason that looks like missing annotation."""
    from ml.pose_features.keypoint_schema import KEYPOINT_INDEX_BY_NAME
    allowed = set(KNOWN_SCHEMA_GAPS.get(trait["trait_id"], []))
    allowed |= VIEW_SCOPED_ALIASES
    for joint in trait.get("required_keypoints", []):
        if joint in allowed:
            continue
        assert joint in KEYPOINT_INDEX_BY_NAME, (
            f"{trait['trait_id']} needs '{joint}', which is not in "
            f"KEYPOINT_SCHEMA. If that is intended, add it to "
            f"KNOWN_SCHEMA_GAPS with the reason.")


def test_the_known_gaps_are_still_gaps():
    """If someone adds the rib landmarks, this fails and the exemption goes.

    A stale exemption is worse than none - it would let a real regression hide
    behind a note about a problem that had already been solved.
    """
    from ml.pose_features.keypoint_schema import KEYPOINT_INDEX_BY_NAME
    for trait_id, joints in KNOWN_SCHEMA_GAPS.items():
        still_missing = [j for j in joints if j not in KEYPOINT_INDEX_BY_NAME]
        assert still_missing == joints, (
            f"{trait_id}: some of {joints} are now in the schema - remove them "
            f"from KNOWN_SCHEMA_GAPS")


def test_all_twenty_contract_traits_are_computable():
    """The count the contract promises."""
    assert len(CONTRACT_TRAITS) == 20
    broken = [t["trait_id"] for t in CONTRACT_TRAITS
              if t["trait_class"] == "B"
              and len(t.get("required_keypoints", [])) < 4]
    assert not broken, f"structurally unmeasurable: {broken}"
