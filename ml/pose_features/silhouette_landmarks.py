"""Derive body landmarks from the silhouette that were never annotated.

WHY THIS EXISTS
The keypoint model trains 22 of the 41 canonical joints. chest_bottom - the
brisket - is not among them, and it is the single most valuable missing one:
it gates body_depth, chest_depth and chest_width_to_depth_ratio.

Annotating it by hand across the training set is days of work. But the brisket
is not a subtle anatomical judgement: on a side view it is simply the lowest
point of the body outline at the front of the chest. Given a real silhouette
that is a geometric lookup, so this derives it instead.

WHAT MAKES IT RELIABLE OR NOT
The derivation is only as good as the joint it anchors on. The first attempt
anchored on chest_front, which is one of the WEAKEST trained joints
(PCK@0.02 = 0.429, third worst of 22) - on a real photo it returned 0.486
confidence and sat mid-body, so the derived point landed on the bottom of the
barrel rather than the brisket.

This anchors on the front leg instead. pastern (0.737), knee (0.721) and hoof
(0.703) are the strongest joints in the whole set, roughly 1.7x more accurate
than chest_front, and the brisket sits just ahead of the front leg. Anchoring
on a strong joint and walking a short distance beats anchoring on a weak joint
and not walking at all.

EVERY DERIVED POINT IS MARKED AS DERIVED
The returned confidence is deliberately capped below what a trained joint can
report. A derived landmark is evidence, not a measurement, and the trait that
consumes it should carry a wider interval because of it.
"""
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# Never let a derived point claim the confidence of a trained one. The cap is
# the anchor joint's own reliability multiplied by this: the derivation adds
# error, it cannot remove any.
DERIVED_CONFIDENCE_CAP = 0.55
# A body column is "solid" when this much of the sampled band is filled.
# Legs are far narrower than the chest, so this is what separates the two.
SOLID_FRACTION = 0.60
# How far ahead of the front leg to look for the brisket, as a fraction of
# the animal's box width.
BRISKET_SEARCH_FRAC = 0.18
# A row belongs to the BODY while it is at least this fraction as wide as
# the widest row. Below the belly only legs remain, so the width collapses
# well past this.
BELLY_WIDTH_FRAC = 0.45
# A derived point is only as good as the pose behind it. One weak joint is not
# enough to establish which way the animal faces, and a wrong facing puts the
# brisket behind the udder. Gir_212 had a single usable joint at 0.30 and
# produced a chest depth of 81% of the animal box - anatomically impossible.
MIN_JOINTS_FOR_DERIVATION = 4
MIN_FACING_CONFIDENCE = 0.35
# A cow's chest depth runs roughly 45-55% of wither height. Allowing 32-68%
# leaves room for posture, camera angle and a loose animal box while still
# rejecting a point that landed on the ground or the topline.
PLAUSIBLE_DEPTH_FRAC = (0.32, 0.68)


def _pt(kps: Dict[str, Tuple[float, float, float]], name: str):
    v = kps.get(name)
    if not v or len(v) < 3 or v[2] <= 0:
        return None
    return float(v[0]), float(v[1]), float(v[2])


def facing_sign(kps: Dict[str, Tuple[float, float, float]],
                animal_bbox: Optional[Sequence[float]] = None) -> Optional[int]:
    """-1 if the animal faces left in the image, +1 if it faces right.

    Worked out from the geometry rather than assumed. A mirrored photo is
    common and a hardcoded assumption would silently put the brisket behind
    the udder.

    Two routes, because requiring BOTH a head joint and a tail joint was too
    strict: a real photo with 13 usable joints still failed, because withers
    and chest_front both happened to be missing while tail_head and every leg
    joint were present.

      1. head joint vs tail joint, when both exist
      2. tail joint against the middle of the animal box - if the rear is in
         the right half, the animal faces left. This needs only ONE joint.
    """
    head = _pt(kps, "withers") or _pt(kps, "chest_front")
    tail = (_pt(kps, "tail_head") or _pt(kps, "pin_left")
            or _pt(kps, "pin_right") or _pt(kps, "hook_left")
            or _pt(kps, "hook_right"))
    if head is not None and tail is not None:
        return -1 if head[0] < tail[0] else 1
    if tail is not None and animal_bbox is not None:
        x1, _y1, x2, _y2 = (float(v) for v in animal_bbox)
        mid = 0.5 * (x1 + x2)
        # rear in the right half -> the head is on the left -> faces left
        return -1 if tail[0] > mid else 1
    if head is not None and animal_bbox is not None:
        x1, _y1, x2, _y2 = (float(v) for v in animal_bbox)
        mid = 0.5 * (x1 + x2)
        return -1 if head[0] < mid else 1
    return None


def front_leg_x(kps: Dict[str, Tuple[float, float, float]],
                animal_bbox: Optional[Sequence[float]] = None) -> Optional[float]:
    """x of the FRONT leg, chosen by which leg is nearer the head end.

    Uses the strong joints - pastern, knee, hoof - rather than a chest joint.
    """
    sign = facing_sign(kps, animal_bbox)
    if sign is None:
        return None
    xs = []
    for base in ("knee", "pastern", "hoof"):
        for side in ("left", "right"):
            p = _pt(kps, f"{base}_{side}")
            if p:
                xs.append(p[0])
    if not xs:
        return None
    # facing left (sign -1) means the head is at low x, so the front leg is
    # the leftmost; facing right, the rightmost
    return min(xs) if sign < 0 else max(xs)


def belly_line_y(mask: np.ndarray) -> Optional[int]:
    """The row where the body ends and only legs continue below.

    Scanning columns does not work: a leg attaches to the body, so the
    silhouette runs unbroken from topline to hoof and the lowest solid pixel
    at any column is the HOOF. The first attempt at this returned exactly
    that - a chest depth of 85% of the animal box.

    Row width separates them cleanly. Through the barrel the silhouette spans
    most of the animal's length; below the belly only two or four legs remain,
    so the width collapses. The belly line is where that collapse happens.
    """
    binary = mask > 0
    widths = binary.sum(axis=1)
    if widths.max() <= 0:
        return None
    body_rows = np.where(widths >= BELLY_WIDTH_FRAC * widths.max())[0]
    if len(body_rows) == 0:
        return None
    return int(body_rows.max())


def derive_chest_bottom(
    mask: np.ndarray,
    kps: Dict[str, Tuple[float, float, float]],
    animal_bbox: Sequence[float],
) -> Optional[Tuple[float, float, float]]:
    """(x, y, confidence) of the brisket, or None if it cannot be derived.

    mask is the binary silhouette from segment_animal, in FULL-IMAGE pixels -
    the same frame the keypoints use, so nothing is converted between them.
    """
    if mask is None or mask.size == 0:
        return None

    # Refuse on a weak pose rather than deriving from noise.
    usable = [v for v in kps.values()
              if isinstance(v, (tuple, list)) and len(v) >= 3 and v[2] > 0]
    if len(usable) < MIN_JOINTS_FOR_DERIVATION:
        return None
    if max((v[2] for v in usable), default=0.0) < MIN_FACING_CONFIDENCE:
        return None

    y = belly_line_y(mask)
    if y is None:
        return None

    # x: just ahead of the front leg if we can find it, else the front
    # quarter of the animal box. The brisket is at the chest end, not mid-body.
    sign = facing_sign(kps, animal_bbox)
    leg_x = front_leg_x(kps, animal_bbox)
    x1, _y1, x2, _y2 = (float(v) for v in animal_bbox)
    if sign is not None and leg_x is not None:
        reach = BRISKET_SEARCH_FRAC * max(1.0, x2 - x1)
        x = leg_x - reach if sign < 0 else leg_x + reach
    elif sign is not None:
        x = x1 + 0.25 * (x2 - x1) if sign < 0 else x2 - 0.25 * (x2 - x1)
    else:
        return None
    x = float(max(0, min(mask.shape[1] - 1, x)))

    # Clamp into the body rather than refusing. Walking a fixed fraction of
    # the box ahead of the front leg can overshoot past the chest and land on
    # empty background - on a tightly cropped animal it lands outside the
    # silhouette entirely. The brisket IS at the front edge of the body, so
    # clamping there is anatomically right, not a fudge.
    body_cols = np.where((mask[max(0, y - 25):y + 5] > 0).any(axis=0))[0]
    if len(body_cols) == 0:
        return None
    front, back = float(body_cols.min()), float(body_cols.max())
    margin = 0.04 * max(1.0, back - front)
    x = min(max(x, front + margin), back - margin)

    # and the silhouette must actually be present there
    col = mask[:, int(max(0, x - 12)):int(x) + 12] > 0
    if col.size == 0 or not col.any():
        return None

    anchor = max([(_pt(kps, f"{b}_{s}") or (0, 0, 0))[2]
                  for b in ("knee", "pastern", "hoof")
                  for s in ("left", "right")] or [0.0])

    # Plausibility gate: with withers available, the derived chest depth must
    # be a depth a real animal could have. This is what catches a point that
    # landed on the ground, on the topline, or on the wrong end of a
    # mis-detected facing direction.
    withers = _pt(kps, "withers")
    if withers is not None:
        box_h = max(1.0, float(animal_bbox[3]) - float(animal_bbox[1]))
        frac = (y - withers[1]) / box_h
        lo, hi = PLAUSIBLE_DEPTH_FRAC
        if not (lo <= frac <= hi):
            return None

    conf = round(min(DERIVED_CONFIDENCE_CAP, float(anchor) or 0.35), 3)
    return float(x), float(y), conf


# The brisket, from the silhouette rather than from the model.
#
# This is the one derivation that OVERRIDES a landmark the pose model already
# produced, so the evidence for it has to be strong, and it is. Audited over 30
# photographs against where a cow's anatomy has to be (see
# ml/eval_landmark_placement.py), chest_front lands in the right region on 0 of
# 18 images it was confident on - median position 0.60 of the way from head to
# tail, when the brisket is at most 0.30. It is being predicted behind the
# middle of the animal. The silhouette's barrel front lands at 0.26.
#
# What that costs while it stands: chest_front to pin_left spans 26% of the
# animal where it should span about 70%, so body_length_to_height_ratio reads
# 0.55 against a 0.9-1.4 band, and with a scale supplied body_length comes out
# at 73 cm on an animal whose barrel alone is 150 cm.
#
# The barrel front is not the brisket exactly - it is where the body first
# reaches most of its full depth, which is a little behind the point of the
# chest - so the derived point carries the same capped confidence as every
# other derived landmark, and the trait layer can see it is derived.
BARREL_DEPTH_FRAC = 0.75


def derive_chest_front(mask: np.ndarray,
                       kps: Dict[str, Tuple[float, float, float]],
                       animal_bbox: Sequence[float]
                       ) -> Optional[Tuple[float, float, float]]:
    """Front of the barrel, on whichever side the animal faces.

    Refuses rather than guessing when the silhouette gives no usable barrel or
    when the facing direction cannot be established - a brisket placed at the
    wrong END of the animal would be far worse than none.
    """
    sign = facing_sign(kps, animal_bbox)
    if sign is None or mask is None:
        return None
    cut = belly_line_y(mask)
    h, w = mask.shape[:2]
    cut = h if cut is None else max(1, min(h, int(cut)))

    depths = np.zeros(w, dtype=np.float64)
    for x in range(w):
        rows = np.flatnonzero(mask[:cut, x])
        depths[x] = (rows[-1] - rows[0] + 1) if rows.size else 0.0
    if depths.max() <= 0:
        return None
    idx = np.flatnonzero(depths >= BARREL_DEPTH_FRAC * depths.max())
    if idx.size < 8:
        return None
    runs = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
    run = max(runs, key=len)
    if run[-1] - run[0] <= 32:
        return None

    # sign is -1 when the animal faces LEFT, so the head end is the low x
    x_front = float(run[0]) if sign < 0 else float(run[-1])
    col = np.flatnonzero(mask[:cut, int(x_front)])
    if not col.size:
        return None
    # the brisket is low on the chest, not on the topline
    y = float(col[0]) + 0.72 * float(col[-1] - col[0])
    return (x_front, y, DERIVED_CONFIDENCE_CAP)


def add_derived_landmarks(
    kps: Dict[str, Tuple[float, float, float]],
    mask: Optional[np.ndarray],
    animal_bbox: Sequence[float],
) -> Tuple[Dict[str, Tuple[float, float, float]], Dict[str, str]]:
    """Fill derivable joints in-place-ish, and report what was derived.

    Returns (keypoints, provenance) where provenance maps a joint name to
    'derived_from_silhouette'. The trait layer can use that to widen the
    interval, and a reviewer can see at a glance which points were not
    actually detected.
    """
    out = dict(kps)
    prov: Dict[str, str] = {}
    if mask is None:
        return out, prov
    existing = out.get("chest_bottom")
    if existing is None or existing[2] <= 0:
        cb = derive_chest_bottom(mask, out, animal_bbox)
        if cb is not None:
            out["chest_bottom"] = cb
            prov["chest_bottom"] = "derived_from_silhouette"

    # chest_front is REPLACED, not merely filled - the only landmark here that
    # overrides a prediction the model made. See derive_chest_front for the
    # audit: 0 of 18 in the right place, median position 0.60 of the way to the
    # tail when the brisket is at most 0.30. Deriving it is strictly better
    # than keeping a landmark that is measurably always wrong, and it is
    # marked derived so nothing downstream mistakes it for a detection.
    cf = derive_chest_front(mask, out, animal_bbox)
    if cf is not None:
        out["chest_front"] = cf
        prov["chest_front"] = "derived_from_silhouette"
    return out, prov


# ---------------------------------------------------------------------------
# Rear-view landmarks
# ---------------------------------------------------------------------------
# Ten of the eleven blocked traits are udder or teat traits, and every one of
# them is view "rear". The app captures a rear photo and the pipeline was
# never running pose on it - so those joints could not appear even once they
# are annotated. That is plumbing, not labelling, and it has to exist either
# way.
#
# udder_floor is the cheapest of them: udder_depth needs that single joint and
# nothing else. On a rear view the udder is the mass hanging between the hind
# legs, so its floor is the lowest point of the central body column - the same
# row-profile idea that finds the belly line on a side view, restricted to the
# middle of the animal so the legs cannot win.

# Only these joints are taken from the rear photo. A joint that the side view
# already measures well must not be overwritten by a rear-view guess.
REAR_VIEW_JOINTS = frozenset({
    "udder_floor", "rear_udder", "rear_udder_top", "rear_udder_left",
    "rear_udder_right", "udder_cleft_top", "udder_cleft_bottom",
    "vulva_base", "teat_front_left", "teat_front_right",
    "teat_rear_left", "teat_rear_right",
    "teat_front_left_top", "teat_front_left_bottom",
    "teat_width_left_1", "teat_width_left_2",
})
# The udder sits in the middle of a rear view; the hind legs are at the edges.
UDDER_CENTRE_FRAC = 0.34

# Rear-frame copies of joints the SIDE view also has, kept under distinct names.
#
# udder_depth is the reason. ICAR measures it from the udder floor to the hock,
# and the hock is not an udder landmark - so the trait needed udder_floor,
# which is merged from the rear photo, together with hock_left, which is not,
# and stayed in SIDE-photo pixels. The distance between them was computed
# across two coordinate frames. It did not fail; it produced 74.7 cm against a
# calibrated band of -10 to 25, which reads as an animal with an extraordinary
# udder rather than as a units error.
#
# The docstring below already said rear coordinates "must never be mixed into
# a side-view distance". The assumption that made it safe - that every udder
# trait is measured entirely within the rear view - was simply not true of
# this one.
#
# These names are deliberately absent from KEYPOINT_SCHEMA: the schema's 41
# entries correspond to the trained checkpoint's own keypoint list, and these
# are not predicted, they are copies made at merge time.
# The left/right pairs matter for a second reason. A trait that compares the
# LEFT side of the animal with the RIGHT - cow-hock deviation, rump width -
# cannot be measured from a side photograph at all, because the two sides
# overlap there. Measured across 40 photographs, the horizontal separation
# between a left landmark and its right partner in a SIDE view:
#
#     hip_bone_left  <-> hip_bone_right     1.70% of the animal
#     hock_left      <-> hock_right         4.82%
#
# against a real rump that is 15-20% of body length wide. Those points are on
# top of each other. rear_legs_rear_view was computing a left-versus-right
# deviation from them and producing a spread of -16 to +15 degrees, which
# looked like conformation and was perspective noise. It is one of the few
# contract traits that scores, and it was scoring on nothing.
REAR_FRAME_ALIASES = {
    "hock_left": "rear_hock_left",
    "hock_right": "rear_hock_right",
    "hip_bone_left": "rear_hip_bone_left",
    "hip_bone_right": "rear_hip_bone_right",
    "hook_left": "rear_hook_left",
    "hook_right": "rear_hook_right",
}


def derive_udder_floor(mask: np.ndarray, animal_bbox: Sequence[float]
                       ) -> Optional[Tuple[float, float, float]]:
    """(x, y, confidence) of the udder floor from a REAR-view silhouette.

    The udder is the lowest central mass. Restricting to the middle third
    stops a hind leg - which reaches further down - from being mistaken for
    it, the same failure the side-view derivation hit with hooves.
    """
    if mask is None or mask.size == 0:
        return None
    x1, y1, x2, y2 = (float(v) for v in animal_bbox)
    cx = 0.5 * (x1 + x2)
    half = 0.5 * UDDER_CENTRE_FRAC * max(1.0, x2 - x1)
    lo = int(max(0, cx - half))
    hi = int(min(mask.shape[1], cx + half))
    if hi - lo < 6:
        return None
    centre = mask[:, lo:hi] > 0
    rows = np.where(centre.any(axis=1))[0]
    if len(rows) == 0:
        return None

    # the udder floor is the lowest row that is still SOLID across the centre
    widths = centre.sum(axis=1)
    solid = np.where(widths >= 0.5 * (hi - lo))[0]
    y = int(solid.max()) if len(solid) else int(rows.max())

    # An udder floor above the middle of the animal is not an udder - most
    # likely the silhouette is of something else, or the photo is not a rear
    # view at all.
    box_h = max(1.0, y2 - y1)
    if (y - y1) / box_h < 0.45:
        return None
    return float(cx), float(y), 0.40


def add_rear_view_landmarks(
    kps: Dict[str, Tuple[float, float, float]],
    rear_kps: Optional[Dict[str, Tuple[float, float, float]]],
    rear_mask: Optional[np.ndarray],
    rear_bbox: Optional[Sequence[float]],
) -> Tuple[Dict[str, Tuple[float, float, float]], Dict[str, str]]:
    """Merge rear-view joints into the side-view keypoints.

    ONLY joints in REAR_VIEW_JOINTS are taken, and only when the side view
    does not already have them - plus the REAR_FRAME_ALIASES, which are copies
    of joints the side view also has, kept under distinct names so a rear-view
    trait can use them without either value overwriting the other. Coordinates from a rear photo are in the REAR
    image's frame, which is a different frame from the side photo - so these
    are usable for measurements taken entirely within the rear view (all the
    udder traits are), and must never be mixed into a side-view distance.
    """
    out = dict(kps)
    prov: Dict[str, str] = {}

    if rear_kps:
        # rear-frame copies first, under their own names, so a rear-view trait
        # can reference a joint the side view also has without either
        # overwriting the other or being measured across two frames
        for src, alias in REAR_FRAME_ALIASES.items():
            v = rear_kps.get(src)
            if v and len(v) >= 3 and v[2] > 0:
                out[alias] = v
                prov[alias] = "detected_in_rear_view"

        for name, v in rear_kps.items():
            if name not in REAR_VIEW_JOINTS:
                continue
            if len(v) < 3 or v[2] <= 0:
                continue
            cur = out.get(name)
            if cur is None or cur[2] <= 0:
                out[name] = v
                prov[name] = "detected_in_rear_view"

    if rear_mask is not None and rear_bbox is not None:
        cur = out.get("udder_floor")
        if cur is None or cur[2] <= 0:
            uf = derive_udder_floor(rear_mask, rear_bbox)
            if uf is not None:
                out["udder_floor"] = uf
                prov["udder_floor"] = "derived_from_rear_silhouette"
    return out, prov
