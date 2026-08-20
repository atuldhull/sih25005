"""The 2D -> 3D volume estimator.

No weighed animals were available, so none of these tests claim the estimator
is ACCURATE. What they can do is pin the things that must hold regardless of
calibration, and those turn out to be most of the ways this could go wrong:

  - a volume built from a known shape must come out as that shape's volume
  - weight must scale as the CUBE of the linear scale, because a scale error
    that is merely large in one dimension is enormous in three
  - the width-to-depth ratio must be scale-free, which is the entire reason
    the rear photograph can be used without a tag of its own
  - and every missing input must produce a refusal with a reason, not a number

A weight is something a farmer may sell an animal on. A wrong one that looks
reasonable is worse than none.
"""
import math

import numpy as np
import pytest

from ml.weight import volume_3d as v3


# --- ellipse perimeter ----------------------------------------------------

def test_a_circle_comes_out_as_two_pi_r():
    """The degenerate case the approximation must get exactly right."""
    for r in (1.0, 7.5, 143.0):
        assert v3.ellipse_perimeter(r, r) == pytest.approx(2 * math.pi * r, rel=1e-9)


def test_a_flattened_section_is_not_understated():
    """The naive pi*(a+b) understates an eccentric ellipse, and a chest IS
    eccentric. Understating girth understates weight through a square."""
    a, b = 100.0, 55.0
    assert v3.ellipse_perimeter(a, b) > math.pi * (a + b)


def test_a_degenerate_axis_gives_zero_rather_than_a_nan():
    assert v3.ellipse_perimeter(0.0, 10.0) == 0.0
    assert v3.ellipse_perimeter(-3.0, 10.0) == 0.0


# --- fixtures: a solid of known volume ------------------------------------

def cylinder_masks(length_px=400, diameter_px=160, pad=40, extra_h=0):
    """A circular cylinder lying along x: side and rear views of one solid.

    Its true volume is pi*r^2*L, which the estimator must recover. A cylinder
    is the right fixture precisely because it has no taper - any disagreement
    is the integrator's error, not the shape's.
    """
    h = diameter_px + 2 * pad + extra_h
    w = length_px + 2 * pad
    side = np.zeros((h, w), np.uint8)
    side[pad:pad + diameter_px, pad:pad + length_px] = 1
    # from behind, the same cylinder is a circle -> width == depth, ratio 1
    rear = np.zeros((h, h), np.uint8)
    yy, xx = np.mgrid[0:h, 0:h]
    r = diameter_px / 2.0
    cy = cx = pad + r
    rear[((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r] = 1
    return side, rear


def cylinder_keypoints(length_px=400, pad=40):
    return {"chest_front": (float(pad), 100.0, 0.9),
            "pin_left": (float(pad + length_px), 100.0, 0.9)}


def test_a_cylinder_recovers_its_own_volume():
    """End to end against a shape whose volume is known in closed form."""
    L, D, pad = 400, 160, 40
    side, rear = cylinder_masks(L, D, pad)
    cm_per_px = 0.5
    got = v3.estimate(side, rear, cylinder_keypoints(L, pad), cm_per_px)
    assert got.measured, got.reason

    r_cm = (D / 2.0) * cm_per_px
    expected_m3 = (math.pi * r_cm ** 2 * (L * cm_per_px)) / 1.0e6
    assert got.volume_m3 == pytest.approx(expected_m3, rel=0.06), (
        f"recovered {got.volume_m3} m3, true {expected_m3} m3")


def test_a_cylinder_seen_end_on_has_ratio_one():
    _, rear = cylinder_masks()
    assert v3.width_depth_ratio(rear, None) == pytest.approx(1.0, abs=0.12)


# --- the invariants -------------------------------------------------------

def test_weight_scales_as_the_cube_of_the_scale():
    """Doubling cm_per_px must multiply the weight by eight.

    This is the property that makes tag-scale accuracy so consequential: a 4%
    error in centimetres per pixel is a 12.5% error in kilograms, and anything
    that silently broke the cube relationship would hide that.
    """
    side, rear = cylinder_masks()
    kps = cylinder_keypoints()
    a = v3.estimate(side, rear, kps, 0.25)
    b = v3.estimate(side, rear, kps, 0.50)
    assert a.measured and b.measured
    assert b.low_kg / a.low_kg == pytest.approx(8.0, rel=0.02)
    assert b.high_kg / a.high_kg == pytest.approx(8.0, rel=0.02)


def test_the_width_ratio_is_scale_free():
    """The whole reason the rear photograph needs no tag.

    The same animal photographed from twice as far away must give the same
    ratio, or mixing it with the side view's centimetres would be meaningless.
    """
    import cv2
    _, rear = cylinder_masks(400, 160, 40)
    small = cv2.resize(rear, (rear.shape[1] // 3, rear.shape[0] // 3),
                       interpolation=cv2.INTER_NEAREST)
    assert v3.width_depth_ratio(rear, None) == pytest.approx(
        v3.width_depth_ratio(small, None), rel=0.08)


def test_a_wider_animal_weighs_more():
    """Monotonicity: the rear view must actually influence the answer."""
    side, rear = cylinder_masks()
    kps = cylinder_keypoints()
    narrow = v3.estimate(side, rear, kps, 0.4)
    # stretch the rear silhouette sideways -> a wider, same-depth animal
    import cv2
    wide_rear = cv2.resize(rear, (rear.shape[1] * 2, rear.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
    wide = v3.estimate(side, wide_rear, kps, 0.4)
    assert wide.measured and narrow.measured
    assert wide.low_kg > narrow.low_kg * 1.5


def test_the_belly_cut_excludes_the_legs():
    """Without it the 'torso' is measured down to the hooves."""
    L, D, pad = 400, 160, 40
    side, _ = cylinder_masks(L, D, pad, extra_h=200)
    withlegs = side.copy()
    withlegs[pad + D:pad + D + 150, pad:pad + 40] = 1        # a leg
    withlegs[pad + D:pad + D + 150, pad + L - 40:pad + L] = 1
    cut = v3.depth_profile(withlegs, pad, pad + L, y_floor=pad + D)
    uncut = v3.depth_profile(withlegs, pad, pad + L, y_floor=None)
    assert cut.max() == pytest.approx(D, abs=2)
    assert uncut.max() > D * 1.5, "without the cut the legs are counted"


def test_a_hole_in_the_mask_does_not_thin_the_animal():
    """Depth spans top to bottom rather than counting set pixels, so a
    segmentation dropout reads as noise, not as a slimmer animal."""
    L, D, pad = 400, 160, 40
    side, _ = cylinder_masks(L, D, pad)
    holed = side.copy()
    holed[pad + 40:pad + 90, pad + 100:pad + 300] = 0
    assert (v3.depth_profile(holed, pad, pad + L).max()
            == pytest.approx(v3.depth_profile(side, pad, pad + L).max(), abs=1))


# --- refusals -------------------------------------------------------------

def test_no_scale_is_refused_with_a_reason():
    side, rear = cylinder_masks()
    got = v3.estimate(side, rear, cylinder_keypoints(), None)
    assert not got.measured
    assert got.low_kg is None and got.high_kg is None
    assert "no_scale" in got.reason


def test_a_missing_rear_view_is_refused_not_guessed():
    """Depth alone does not determine a volume. Assuming a width would be
    inventing the very dimension the second photograph exists to supply."""
    side, _ = cylinder_masks()
    got = v3.estimate(side, None, cylinder_keypoints(), 0.4)
    assert not got.measured
    assert "no_width_ratio" in got.reason


def test_missing_torso_landmarks_are_refused():
    side, rear = cylinder_masks()
    got = v3.estimate(side, rear, {"withers": (10.0, 10.0, 0.0)}, 0.4)
    assert not got.measured
    assert "no_torso_span" in got.reason


def test_no_silhouette_is_refused():
    _, rear = cylinder_masks()
    got = v3.estimate(None, rear, cylinder_keypoints(), 0.4)
    assert not got.measured
    assert "no_silhouette" in got.reason


def test_an_empty_silhouette_is_refused():
    L, pad = 400, 40
    empty = np.zeros((240, 480), np.uint8)
    _, rear = cylinder_masks()
    got = v3.estimate(empty, rear, cylinder_keypoints(L, pad), 0.4)
    assert not got.measured
    assert got.reason is not None


# --- the second opinion ---------------------------------------------------

def test_both_routes_are_reported_not_averaged():
    side, rear = cylinder_masks()
    got = v3.estimate(side, rear, cylinder_keypoints(), 0.4)
    assert got.method == "torso-volume-from-two-views"
    assert "girth-length" in got.cross_check
    assert got.heart_girth_cm is not None, (
        "heart girth was previously classed SMAL and never measured; the "
        "ellipse model is what makes it available")


def test_a_disagreement_widens_the_interval_rather_than_hiding():
    """Two methods that contradict each other must not be quietly averaged
    into a confident-looking middle."""
    side, rear = cylinder_masks()
    got = v3.estimate(side, rear, cylinder_keypoints(), 0.4)
    assert got.measured
    if "DISAGREES" in got.cross_check:
        girth_kg = float(got.cross_check.split(":")[1].split("kg")[0])
        assert got.low_kg <= girth_kg <= got.high_kg, (
            "a disagreeing cross-check must be spanned by the interval")


# --- is this even a side-on animal? ---------------------------------------
# Depth over length is anatomy: a bovine torso is roughly twice as long as it
# is deep. Checking it costs nothing and is not circular, because it is known
# independently of anything this pipeline measured. Measured on real
# photographs the value ranges from 0.52 on a genuine side view to 3.94 on a
# head-on one, so this gate decides whether the photograph was the one the
# method needs - and a head-on shot otherwise produces a beautifully formed
# volume that means nothing.

def _too_deep_masks():
    """A torso as deep as it is long: a head-on animal, not a side view."""
    side = np.zeros((520, 520), np.uint8)
    side[60:460, 60:460] = 1                      # square: depth == length
    _, rear = cylinder_masks()
    return side, rear


def test_a_head_on_photograph_is_refused():
    side, rear = _too_deep_masks()
    kps = {"chest_front": (60.0, 200.0, 0.9), "pin_left": (460.0, 200.0, 0.9)}
    got = v3.estimate(side, rear, kps, 0.4)
    assert not got.measured
    assert "not_a_side_view" in got.reason
    assert "deep" in got.reason, "the reason must say what was wrong with it"


def test_the_gate_is_on_geometry_not_on_the_answer():
    """The plausibility check must never look at the output weight.

    Gating on an expected weight range would make the estimator only ever
    confirm what it was told to expect, and a genuinely heavy or light animal
    would be refused for being unusual - which is exactly the case worth
    measuring.
    """
    import inspect
    src = inspect.getsource(v3.estimate)
    gate = src[src.index("PLAUSIBLE_DEPTH_TO_LENGTH"):src.index("dx_px =")]
    for forbidden in ("lo_kg", "hi_kg", "schaeffer", "DENSITY"):
        assert forbidden not in gate, (
            f"the plausibility gate refers to {forbidden} - it must decide "
            f"from geometry alone, before any weight exists")


# --- the barrel span ------------------------------------------------------

def test_the_barrel_span_ignores_a_projecting_head():
    """The head and neck are shallow next to the barrel, so a depth threshold
    drops them. Using keypoints instead gave a torso 225 px long on an animal
    835 px wide, making it 1.34 times deeper than long."""
    L, D, pad = 400, 160, 40
    side, _ = cylinder_masks(L, D, pad)
    withhead = side.copy()
    # a shallow neck projecting forward at a third of the body's depth
    withhead[pad + 30:pad + 80, pad - 150:pad] = 1
    span = v3._barrel_span(withhead, None)
    assert span is not None
    assert span[0] >= pad - 8, f"the neck was counted as barrel: {span}"
    assert (span[1] - span[0]) == pytest.approx(L, abs=12)


def test_the_barrel_span_is_preferred_over_the_keypoints():
    """Length enters the volume linearly and the scale enters it cubed, so the
    more reliable ruler must win."""
    L, D, pad = 400, 160, 40
    side, rear = cylinder_masks(L, D, pad)
    liar = {"chest_front": (float(pad + 150), 100.0, 0.9),   # far too short
            "pin_left": (float(pad + 250), 100.0, 0.9)}
    got = v3.estimate(side, rear, liar, 0.5)
    assert got.measured, got.reason
    assert got.body_length_cm == pytest.approx(L * 0.5, rel=0.06), (
        "the silhouette span should have overridden the keypoints")


def test_a_detached_patch_of_mask_cannot_extend_the_torso():
    """The longest CONTIGUOUS run, so a second animal or a fence post in the
    mask does not silently make the animal longer - and heavier."""
    L, D, pad = 400, 160, 40
    side, _ = cylinder_masks(L, D, pad)
    withblob = np.zeros((side.shape[0], side.shape[1] + 300), np.uint8)
    withblob[:, :side.shape[1]] = side
    withblob[pad:pad + D, side.shape[1] + 120:side.shape[1] + 260] = 1
    span = v3._barrel_span(withblob, None)
    assert span is not None
    assert (span[1] - span[0]) == pytest.approx(L, abs=12), (
        f"the detached blob extended the torso to {span}")
