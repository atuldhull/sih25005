"""The brisket, taken from the silhouette instead of from the model.

This is the only derivation that OVERRIDES a landmark the pose model already
produced, so it needs more justification than the ones that merely fill a gap.

Audited over 30 photographs against where a cow's anatomy has to be
(ml/eval_landmark_placement.py), the model's chest_front landed in the right
region on 0 of 18 images it was confident on - median position 0.60 of the way
from head to tail, when the brisket is at most 0.30. It was being predicted
behind the middle of the animal, and that is not a tolerance question.

What it cost while it stood: chest_front to pin_left spanned 26% of the animal
where it should span about 70%, so body_length_to_height_ratio read 0.55
against a band of 0.9-1.4. After the substitution that median is 0.93, and
shoulder_angle moves from 25 degrees to 44 against a band of 45-65.
"""
import numpy as np
import pytest

from ml.pose_features.silhouette_landmarks import (
    DERIVED_CONFIDENCE_CAP,
    add_derived_landmarks,
    derive_chest_front,
)


def cow_mask(w=800, h=500, head=True, facing_left=True):
    """A barrel with a shallower head and neck projecting from one end."""
    m = np.zeros((h, w), np.uint8)
    m[120:380, 150:650] = 1                       # the barrel
    if head:
        if facing_left:
            m[170:260, 30:150] = 1                # neck and head, front-left
        else:
            m[170:260, 650:770] = 1
    return m


def facing_kps(left=True):
    """withers ahead of tail_head means facing left, per facing_sign."""
    return ({"withers": (250.0, 130.0, 0.8), "tail_head": (630.0, 140.0, 0.8)}
            if left else
            {"withers": (550.0, 130.0, 0.8), "tail_head": (170.0, 140.0, 0.8)})


def test_the_brisket_is_at_the_front_of_the_barrel_not_the_nose():
    """The head is shallower than the barrel, so it falls out on depth alone.

    Taking the front of the whole SILHOUETTE instead would land on the muzzle,
    which is further forward than any brisket.
    """
    m = cow_mask()
    got = derive_chest_front(m, facing_kps(True), (0, 0, 800, 500))
    assert got is not None
    assert got[0] == pytest.approx(150, abs=20), (
        f"expected the barrel's front edge near x=150, got {got[0]}")


def test_it_follows_the_animal_when_the_photograph_is_mirrored():
    """A mirrored photo is common, and a brisket placed at the wrong END of
    the animal would be far worse than no brisket at all."""
    m = cow_mask(facing_left=False)
    got = derive_chest_front(m, facing_kps(False), (0, 0, 800, 500))
    assert got is not None
    assert got[0] == pytest.approx(650, abs=20), (
        f"facing right, the brisket is the barrel's RIGHT edge, got {got[0]}")


def test_the_brisket_sits_low_on_the_chest_not_on_the_topline():
    m = cow_mask()
    got = derive_chest_front(m, facing_kps(True), (0, 0, 800, 500))
    assert got is not None
    assert got[1] > 120 + 0.5 * (380 - 120), (
        "the point of the chest is in the lower half of the body depth")


def test_it_refuses_when_the_facing_direction_is_unknown():
    """With no head or tail landmark there is no way to tell front from back,
    and a coin flip would put the brisket behind the udder."""
    assert derive_chest_front(cow_mask(), {}, (0, 0, 800, 500)) is None


def test_it_refuses_on_an_empty_silhouette():
    assert derive_chest_front(np.zeros((500, 800), np.uint8),
                              facing_kps(True), (0, 0, 800, 500)) is None
    assert derive_chest_front(None, facing_kps(True), (0, 0, 800, 500)) is None


# --- the override --------------------------------------------------------

def test_a_model_predicted_chest_front_is_REPLACED():
    """Unlike every other derivation here, this one overrides rather than
    fills. Keeping a landmark that is measurably always wrong is worse."""
    kps = dict(facing_kps(True))
    kps["chest_front"] = (480.0, 250.0, 0.9)      # mid-barrel, as the model does
    out, prov = add_derived_landmarks(kps, cow_mask(), (0, 0, 800, 500))
    assert prov.get("chest_front") == "derived_from_silhouette"
    assert out["chest_front"][0] < 300, (
        f"the mid-body prediction survived: {out['chest_front']}")


def test_the_derived_point_never_claims_full_confidence():
    out, _ = add_derived_landmarks(facing_kps(True), cow_mask(), (0, 0, 800, 500))
    assert out["chest_front"][2] == DERIVED_CONFIDENCE_CAP
    assert out["chest_front"][2] < 1.0, (
        "the barrel front is a little behind the true point of the chest, "
        "and the confidence has to say so")


def test_other_landmarks_are_left_alone():
    kps = dict(facing_kps(True))
    kps["hock_left"] = (600.0, 400.0, 0.71)
    out, prov = add_derived_landmarks(kps, cow_mask(), (0, 0, 800, 500))
    assert out["hock_left"] == (600.0, 400.0, 0.71)
    assert "hock_left" not in prov
    assert out["tail_head"] == kps["tail_head"]


def test_nothing_is_derived_without_a_silhouette():
    kps = dict(facing_kps(True))
    kps["chest_front"] = (480.0, 250.0, 0.9)
    out, prov = add_derived_landmarks(kps, None, (0, 0, 800, 500))
    assert prov == {}
    assert out["chest_front"] == (480.0, 250.0, 0.9), (
        "with no silhouette there is nothing better to offer, so the model's "
        "prediction stands rather than being discarded")
