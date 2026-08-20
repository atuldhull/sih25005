"""Weight from a reconstructed torso volume: the 2D -> 3D step.

WHAT THIS DOES
Two photographs of an animal are two orthogonal projections of one solid. The
side view gives, at every station along the back, how DEEP the animal is
(dorsal to ventral). The rear view gives how WIDE it is. Treat each station as
an ellipse with those two axes, stack the ellipses along the body, and the
result is a volume - a real three-dimensional quantity recovered from flat
pictures. Multiply by a density and it is a weight.

    V = sum over stations of  (pi/4) * depth(x) * width(x) * dx

THE SCALE PROBLEM, AND WHY THE RATIO SOLVES IT
Converting pixels to centimetres needs the ear tag, and the tag is in the SIDE
photograph. The rear photograph is a different shot from a different distance,
so its pixels mean something else entirely, and no measurement taken from it in
pixels can be mixed with one taken from the side.

What CAN cross between them is a ratio. Width divided by depth is
dimensionless - it is the same number whether measured in rear-photo pixels,
side-photo pixels, or centimetres. So the rear photograph is used only to
answer "how much wider than deep is this animal", and that ratio is then
applied to the side photograph's depth profile, which does carry a scale. The
rear photo never needs a tag of its own.

    k = width_rear_px / depth_rear_px          (dimensionless)
    width(x) = k * depth(x)
    V = (pi/4) * k * sum depth(x)^2 * dx

HEART GIRTH FALLS OUT OF THE SAME MODEL
Heart girth is the circumference of the chest just behind the fore leg. Under
this model that station is an ellipse of axes depth and k*depth, so its
perimeter is computable - which matters because heart_girth was previously
classed SMAL and never measured at all, and Schaeffer's formula depends on it.
That gives a SECOND weight estimate from the same photographs by a completely
different route, and the two are reported against each other rather than
averaged. Two methods agreeing is evidence; one method alone is a guess.

WHAT IS ASSUMED, PLAINLY
  - Elliptical cross-sections. A real torso is not an ellipse; it is flatter
    over the back and rounder at the barrel.
  - A constant width-to-depth ratio along the body, calibrated where the rear
    view can see it. A real animal narrows toward the shoulder.
  - Body density. Live cattle are close to water but not exactly, and gut fill
    moves it. This is carried as a RANGE rather than a number, which is where
    most of the reported interval comes from.
  - The torso only. Head, neck and legs are excluded deliberately - including
    a leg would add volume that weighs almost nothing per unit length and is
    projected end-on anyway.

None of these is calibrated against weighed animals, because no weighbridge
data was available. So this estimator reports an interval, names its method,
and is cross-checked; it does not claim an accuracy it has not been shown to
have.
"""
import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# Live cattle sit close to the density of water. The spread is real - gut fill
# alone moves an adult animal by tens of kilograms through the day - so it is
# carried as an interval instead of being collapsed to a single number that
# would look more precise than it is.
DENSITY_KG_PER_M3 = (900.0, 1010.0)

# Schaeffer's formula in metric: weight_kg = girth_cm^2 * length_cm / 10838.4,
# where 10838.4 = 300 * 2.54^3 * 2.20462 converts the imperial
# girth_in^2 * length_in / 300 form.
SCHAEFFER_DIVISOR = 10838.4

# Stations along the body. More than this and each slice is thinner than the
# silhouette's own edge accuracy; fewer and the barrel is under-sampled.
N_STATIONS = 48

# A station whose depth is a small fraction of the deepest one is off the end
# of the torso - tail, or a gap in the mask - and contributes noise.
MIN_STATION_DEPTH_FRAC = 0.25

# Where the barrel meets the legs the rear silhouette loses at least this
# fraction of its width within a row. Below it the narrowing is a taper, not a
# join, and nothing is cut.
ABRUPT_DROP_FRAC = 0.25

# Heart girth is measured just behind the fore leg, expressed as a fraction of
# the torso length back from its front edge.
HEART_STATION_FRAC = 0.18

# A bovine torso is roughly twice as long as it is deep. Outside this band the
# reconstruction is not of a side-on animal at all - a three-quarter view, a
# head-on shot, or a silhouette that has captured a second animal - and the
# volume it produces is meaningless while looking perfectly well-formed.
#
# Measured on arbitrary web photographs the ratio ranges from 0.52 on a genuine
# side view to 3.94 on a head-on one, so this gate is doing most of the work of
# deciding whether the two photographs were the ones the method needs.
PLAUSIBLE_DEPTH_TO_LENGTH = (0.30, 0.65)

# A column is part of the barrel if it is at least this deep relative to the
# deepest one. The head and neck project forward at much less than this.
BARREL_DEPTH_FRAC = 0.75

# The two routes should not disagree wildly. Beyond this they are reported as
# disagreeing rather than reconciled, because a quiet average of two estimates
# that contradict each other is the least informative thing to show.
CROSS_CHECK_TOLERANCE = 0.25


@dataclass
class VolumeWeight:
    """A weight estimate with its interval, its method, and its second opinion."""
    low_kg: Optional[float]
    high_kg: Optional[float]
    method: Optional[str]
    cross_check: Optional[str]
    volume_m3: Optional[float] = None
    heart_girth_cm: Optional[float] = None
    body_length_cm: Optional[float] = None
    width_depth_ratio: Optional[float] = None
    reason: Optional[str] = None

    @property
    def measured(self) -> bool:
        return self.low_kg is not None and self.high_kg is not None


def ellipse_perimeter(a: float, b: float) -> float:
    """Ramanujan's second approximation, for semi-axes a and b.

    Accurate to better than one part in a million for the eccentricities a
    torso produces, and unlike the naive pi*(a+b) it does not understate a
    flattened section - which is the shape a real chest actually is.
    """
    if a <= 0 or b <= 0:
        return 0.0
    h = ((a - b) ** 2) / ((a + b) ** 2)
    return math.pi * (a + b) * (1.0 + (3.0 * h) / (10.0 + math.sqrt(4.0 - 3.0 * h)))


def depth_profile(mask: np.ndarray, x0: int, x1: int,
                  y_floor: Optional[int] = None,
                  n: int = N_STATIONS) -> Optional[np.ndarray]:
    """Vertical extent of the silhouette at n stations between x0 and x1.

    y_floor, when given, is the belly line: everything below it is leg and is
    excluded. Without that cut the profile measures from the back down to the
    hooves and the "torso" becomes half again as deep as the animal.
    """
    if mask is None or x1 <= x0:
        return None
    h, w = mask.shape[:2]
    x0, x1 = max(0, int(x0)), min(w, int(x1))
    if x1 - x0 < n:
        return None
    cut = h if y_floor is None else max(1, min(h, int(y_floor)))

    xs = np.linspace(x0, x1 - 1, n).astype(int)
    out = np.zeros(n, dtype=np.float64)
    for i, x in enumerate(xs):
        col = mask[:cut, x]
        rows = np.flatnonzero(col)
        # Span top-to-bottom rather than counting set pixels: a hole in the
        # mask should not read as a thinner animal.
        out[i] = (rows[-1] - rows[0] + 1) if rows.size else 0.0
    return out


def _barrel_bottom(widths: np.ndarray) -> Optional[int]:
    """Row where the barrel ends and the legs begin, or None if it does not.

    The giveaway is not that the legs are narrow - the top and bottom of any
    rounded body are narrow too - it is that the transition is ABRUPT. A
    barrel tapers smoothly into its own edge; where it meets the legs the
    silhouette loses half its width within a row or two.

    Cutting on narrowness alone would trim a smooth shape as well, which is
    what an earlier fixed-fraction rule did: it reported a circle as being
    1.6x wider than deep.
    """
    if widths.size < 12:
        return None
    w_max = float(widths.max())
    if w_max <= 0:
        return None
    # only look below the widest row - the barrel is above it
    start = int(np.argmax(widths))
    lower = widths[start:]
    if lower.size < 6:
        return None
    drops = -np.diff(lower.astype(np.float64))
    i = int(np.argmax(drops))
    if drops[i] < ABRUPT_DROP_FRAC * w_max:
        return None                       # a smooth taper: nothing to cut
    # and the narrowing has to persist, or it is a notch in the mask
    below = lower[i + 1:]
    if below.size and float(np.median(below)) > 0.75 * w_max:
        return None
    return start + i + 1


def width_depth_ratio(rear_mask: np.ndarray,
                      rear_bbox: Optional[Sequence[float]] = None) -> Optional[float]:
    """How much wider than deep the animal is, from the rear photograph.

    Both quantities are in that photograph's own pixels, so the ratio is
    dimensionless and carries across to the side photograph unchanged. This is
    the only thing taken from the rear view, and it is why the rear photo does
    not need a tag or a scale of its own.

    The torso is an ellipse under this model, and an ellipse's bounding box IS
    its two axes - so once the legs are removed the ratio is just the box.
    """
    if rear_mask is None:
        return None
    ys, xs = np.nonzero(rear_mask)
    if ys.size < 64:
        return None

    y0, y1 = int(ys.min()), int(ys.max())
    widths = np.zeros(y1 - y0 + 1, dtype=np.float64)
    for i, y in enumerate(range(y0, y1 + 1)):
        row = np.flatnonzero(rear_mask[y])
        widths[i] = (row[-1] - row[0] + 1) if row.size else 0.0

    cut = _barrel_bottom(widths)
    barrel = widths[:cut] if cut else widths
    if barrel.size < 6:
        return None

    width = float(barrel.max())
    depth = float(barrel.size)
    if depth <= 0 or width <= 0:
        return None
    r = width / depth
    # A torso is wider than it is deep, but not by much, and never much
    # narrower either. Outside this the mask has caught something else.
    return r if 0.45 <= r <= 2.2 else None


def _barrel_span(mask: np.ndarray, y_floor: Optional[int]
                 ) -> Optional[Tuple[float, float]]:
    """Front and rear of the barrel, taken from the silhouette itself.

    Preferred over the keypoint span because the landmarks it would need -
    chest_front especially, at PCK@0.02 0.429 - are among the least reliable,
    and the length enters the volume linearly and the SCALE cubed. On one real
    photograph the keypoint span gave a torso 225 px long against a silhouette
    835 px wide, which made the animal 1.34 times deeper than it was long.

    The barrel is the run of columns that are nearly as deep as the deepest.
    Head and neck project forward at a fraction of the body's depth, so they
    fall out on their own.
    """
    if mask is None:
        return None
    h, w = mask.shape[:2]
    cut = h if y_floor is None else max(1, min(h, int(y_floor)))
    depths = np.zeros(w, dtype=np.float64)
    for x in range(w):
        rows = np.flatnonzero(mask[:cut, x])
        depths[x] = (rows[-1] - rows[0] + 1) if rows.size else 0.0
    if depths.max() <= 0:
        return None
    idx = np.flatnonzero(depths >= BARREL_DEPTH_FRAC * depths.max())
    if idx.size < 8:
        return None
    # the longest CONTIGUOUS run, so a detached patch of mask cannot extend it
    runs = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
    run = max(runs, key=len)
    return (float(run[0]), float(run[-1])) if run[-1] - run[0] > 32 else None


def _torso_span(keypoints: Dict[str, Tuple[float, float, float]]
                ) -> Optional[Tuple[float, float]]:
    """Front and rear x of the torso, from whichever landmarks are available.

    Not the bounding box: that includes the head and the tail, and a head is
    perhaps a fifth of the animal's length carrying none of the barrel.
    """
    def px(name):
        v = keypoints.get(name)
        return v[0] if v and v[2] > 0 else None

    front = px("chest_front") or px("shoulder_left") or px("withers")
    back = px("pin_left") or px("pin_right") or px("tail_head") or px("hook_left")
    if front is None or back is None:
        return None
    lo, hi = (front, back) if front < back else (back, front)
    return (lo, hi) if hi - lo > 32 else None


def estimate(side_mask: np.ndarray,
             rear_mask: Optional[np.ndarray],
             keypoints: Dict[str, Tuple[float, float, float]],
             cm_per_px: Optional[float],
             belly_y: Optional[int] = None,
             rear_bbox: Optional[Sequence[float]] = None) -> VolumeWeight:
    """Reconstruct the torso volume and turn it into a weight interval.

    Refuses, with a reason, whenever an input it genuinely needs is missing -
    a weight is a number a farmer may sell an animal on, and a plausible
    fabricated one is worse than "not measured".
    """
    if cm_per_px is None:
        return VolumeWeight(None, None, None, None,
                            reason="no_scale: weight needs centimetres, which "
                                   "come from the ear tag")
    if side_mask is None:
        return VolumeWeight(None, None, None, None,
                            reason="no_silhouette: segmentation did not run")

    span = _torso_span(keypoints)
    if span is None:
        return VolumeWeight(None, None, None, None,
                            reason="no_torso_span: need a front landmark "
                                   "(chest/shoulder/withers) and a rear one "
                                   "(pin/hook/tail_head)")

    ratio = width_depth_ratio(rear_mask, rear_bbox) if rear_mask is not None else None
    if ratio is None:
        return VolumeWeight(None, None, None, None,
                            reason="no_width_ratio: the rear photograph did "
                                   "not give a usable silhouette, and depth "
                                   "alone does not determine a volume")

    # The silhouette is the better ruler here; keypoints are the fallback.
    span = _barrel_span(side_mask, belly_y) or span

    x0, x1 = span
    prof = depth_profile(side_mask, int(x0), int(x1), y_floor=belly_y)
    if prof is None or not np.any(prof):
        return VolumeWeight(None, None, None, None,
                            reason="no_depth_profile: the silhouette is empty "
                                   "across the torso")

    # Drop stations that are clearly off the end of the barrel.
    keep = prof >= MIN_STATION_DEPTH_FRAC * prof.max()
    if keep.sum() < 8:
        return VolumeWeight(None, None, None, None,
                            reason="torso_too_short: fewer than eight usable "
                                   "stations along the body")

    length_px = float(x1 - x0)

    # Is this even a side-on bovine torso? Depth over length is anatomy, known
    # independently of anything this pipeline measured, so it can be checked
    # without circularity - unlike the output weight, which must not be gated
    # against an expected range or the estimator would only ever confirm what
    # it was told to expect.
    d_over_l = float(prof.max()) / length_px if length_px > 0 else 0.0
    lo_r, hi_r = PLAUSIBLE_DEPTH_TO_LENGTH
    if not (lo_r <= d_over_l <= hi_r):
        return VolumeWeight(
            None, None, None, None,
            reason=(f"not_a_side_view: the reconstructed torso is "
                    f"{d_over_l:.2f} times as deep as it is long, outside the "
                    f"{lo_r}-{hi_r} a bovine torso occupies. The side "
                    f"photograph is probably not square-on to the animal."))

    dx_px = length_px / (len(prof) - 1)

    # V = (pi/4) * k * sum d(x)^2 dx, in pixels cubed, then scaled once.
    vol_px3 = (math.pi / 4.0) * ratio * float(np.sum((prof[keep] ** 2))) * dx_px
    cm3 = vol_px3 * (cm_per_px ** 3)
    m3 = cm3 / 1.0e6

    lo_kg = m3 * DENSITY_KG_PER_M3[0]
    hi_kg = m3 * DENSITY_KG_PER_M3[1]

    # --- the independent second route ------------------------------------
    heart_idx = int(round(HEART_STATION_FRAC * (len(prof) - 1)))
    heart_depth_px = float(prof[heart_idx]) if prof[heart_idx] > 0 else float(prof.max())
    girth_cm = ellipse_perimeter(heart_depth_px * cm_per_px / 2.0,
                                 ratio * heart_depth_px * cm_per_px / 2.0)
    length_cm = length_px * cm_per_px
    schaeffer_kg = (girth_cm ** 2) * length_cm / SCHAEFFER_DIVISOR

    mid = 0.5 * (lo_kg + hi_kg)
    agree = mid > 0 and abs(schaeffer_kg - mid) / mid <= CROSS_CHECK_TOLERANCE
    cross = (f"girth-length: {schaeffer_kg:.0f} kg"
             + ("" if agree else " - DISAGREES with the volume estimate"))

    if not agree:
        # Do not quietly average two methods that contradict each other. Widen
        # to span both and let the reader see the disagreement in the number.
        lo_kg = min(lo_kg, schaeffer_kg)
        hi_kg = max(hi_kg, schaeffer_kg)

    return VolumeWeight(
        low_kg=round(lo_kg, 1),
        high_kg=round(hi_kg, 1),
        method="torso-volume-from-two-views",
        cross_check=cross,
        volume_m3=round(m3, 4),
        heart_girth_cm=round(girth_cm, 1),
        body_length_cm=round(length_cm, 1),
        width_depth_ratio=round(ratio, 3),
    )
