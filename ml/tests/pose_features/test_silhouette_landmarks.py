"""Deriving chest_bottom from the silhouette - and refusing when it cannot.

chest_bottom (the brisket) was never annotated, so the pose model cannot
detect it. It gates body_depth, chest_depth and chest_width_to_depth_ratio.
On a side view it is simply where the body ends and only legs continue below,
so a real silhouette makes it a geometric lookup rather than a judgement.

Everything here is about the REFUSALS. A derived landmark that is wrong is
worse than one that is absent, because the trait built on it looks measured.
"""
import numpy as np
import pytest

from ml.pose_features.silhouette_landmarks import (
    DERIVED_CONFIDENCE_CAP,
    MIN_JOINTS_FOR_DERIVATION,
    PLAUSIBLE_DEPTH_FRAC,
    add_derived_landmarks,
    belly_line_y,
    derive_chest_bottom,
    facing_sign,
    front_leg_x,
)


def _cow_mask(w=400, h=300, body_top=60, body_bot=170, leg_bot=270):
    """A crude side-on cow: a wide barrel with two narrow legs below."""
    m = np.zeros((h, w), np.uint8)
    m[body_top:body_bot, 40:360] = 255           # barrel
    m[body_bot:leg_bot, 70:100] = 255            # front leg
    m[body_bot:leg_bot, 300:330] = 255           # hind leg
    return m


def _kps(faces_left=True, n=8, conf=0.7):
    """Enough joints to clear the quality gate, facing a known direction."""
    head_x, tail_x = (60, 340) if faces_left else (340, 60)
    k = {"withers": (head_x, 70.0, conf), "tail_head": (tail_x, 75.0, conf)}
    leg_x = 85 if faces_left else 315
    for i, name in enumerate(("knee_left", "knee_right", "pastern_left",
                              "pastern_right", "hoof_left", "hoof_right")):
        k[name] = (float(leg_x), 200.0 + i, conf)
    while len(k) < n:
        k[f"filler_{len(k)}"] = (200.0, 100.0, conf)
    return k


def test_belly_line_is_where_the_body_ends_not_the_hoof():
    """Scanning columns finds the HOOF, because legs attach to the body.

    The first version of this did exactly that and returned a chest depth of
    85% of the animal box. Row width separates body from legs cleanly.
    """
    m = _cow_mask(body_bot=170, leg_bot=270)
    y = belly_line_y(m)
    assert y is not None
    assert 160 <= y <= 175, f"belly line at {y}, expected the body bottom ~170"
    assert y < 260, "must not be the hoof"


def test_facing_is_detected_both_ways():
    assert facing_sign(_kps(faces_left=True)) == -1
    assert facing_sign(_kps(faces_left=False)) == 1


def test_facing_works_from_one_joint_plus_the_box():
    """A real photo with 13 usable joints failed because withers and
    chest_front both happened to be missing. One rear joint plus the box
    is enough."""
    only_tail = {"tail_head": (340.0, 75.0, 0.6)}
    assert facing_sign(only_tail, (40, 60, 360, 270)) == -1
    only_tail_left = {"tail_head": (60.0, 75.0, 0.6)}
    assert facing_sign(only_tail_left, (40, 60, 360, 270)) == 1


def test_front_leg_is_the_one_nearer_the_head():
    k = _kps(faces_left=True)
    x = front_leg_x(k)
    assert x is not None and x < 200, "facing left -> front leg is leftmost"


def test_a_weak_pose_is_refused_not_guessed():
    """One joint at low confidence cannot establish facing.

    A real photo did exactly this and produced a chest depth of 81% - an
    anatomically impossible value that would have reached the scorecard.
    """
    m = _cow_mask()
    weak = {"withers": (200.0, 70.0, 0.30)}
    assert len(weak) < MIN_JOINTS_FOR_DERIVATION
    assert derive_chest_bottom(m, weak, (40, 60, 360, 270)) is None


def test_an_implausible_depth_is_refused():
    """If the derived point implies a depth no animal has, refuse it.

    Here the withers is placed at the very top and the mask's body extends
    almost to the frame bottom, so the implied depth blows past the range.
    """
    m = np.zeros((300, 400), np.uint8)
    m[10:290, 40:360] = 255           # a 'body' filling nearly the whole box
    k = _kps()
    k["withers"] = (60.0, 12.0, 0.7)
    out = derive_chest_bottom(m, k, (40, 20, 360, 120))   # a short box
    assert out is None


def test_a_good_case_derives_a_plausible_point():
    m = _cow_mask()
    k = _kps()
    out = derive_chest_bottom(m, k, (40, 60, 360, 270))
    assert out is not None
    x, y, conf = out
    assert 160 <= y <= 175
    assert conf <= DERIVED_CONFIDENCE_CAP, "a derived point must never claim " \
                                           "the confidence of a detected one"
    lo, hi = PLAUSIBLE_DEPTH_FRAC
    frac = (y - k["withers"][1]) / (270 - 60)
    assert lo <= frac <= hi


def test_derived_points_are_marked_as_derived():
    m = _cow_mask()
    k2, prov = add_derived_landmarks(_kps(), m, (40, 60, 360, 270))
    assert prov.get("chest_bottom") == "derived_from_silhouette"
    assert k2["chest_bottom"][2] > 0


def test_no_mask_means_no_derivation():
    k2, prov = add_derived_landmarks(_kps(), None, (40, 60, 360, 270))
    assert prov == {}
    assert k2.get("chest_bottom") is None


def test_an_already_detected_joint_is_never_overwritten():
    """If the model ever learns chest_bottom, the real one must win."""
    m = _cow_mask()
    k = _kps()
    k["chest_bottom"] = (123.0, 456.0, 0.9)
    k2, prov = add_derived_landmarks(k, m, (40, 60, 360, 270))
    assert k2["chest_bottom"] == (123.0, 456.0, 0.9)
    assert "chest_bottom" not in prov
