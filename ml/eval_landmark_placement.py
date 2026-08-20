"""Do the landmarks land where that anatomy actually is?

PCK measures distance from an annotated point. It cannot tell you that the
annotations themselves are being reproduced in the wrong PLACE on the animal,
and it says nothing at all if a joint is confidently predicted somewhere no
brisket has ever been.

This checks something PCK cannot: whether each landmark falls in the region of
the silhouette where that part of a cow has to be. A brisket is at the front.
A pin bone is at the back, high up. Withers are on the topline. Hooves are on
the ground. None of that needs ground truth - it follows from the animal's own
outline, so it can be run on any photograph.

Each joint is given a box in silhouette-relative coordinates, generously sized
so that only a real failure trips it:

    x = 0.0 at the front of the animal, 1.0 at the rear
    y = 0.0 at the topline,             1.0 at the ground

    python -m ml.eval_landmark_placement <folder> [limit]
"""
import collections
import statistics
import sys
from pathlib import Path

import numpy as np

from ml.detection.detector import detect_animal, segment_animal
from ml.pose_features.pose_extractor import extract_keypoints
from ml.pose_features.silhouette_landmarks import (
    add_derived_landmarks, facing_sign)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# (x_low, x_high, y_low, y_high), fractions of the silhouette, front-to-rear.
# Deliberately loose - these are "anywhere near right", not tolerances.
EXPECTED = {
    "withers":          (0.20, 0.55, 0.00, 0.30),
    "back_mid":         (0.35, 0.70, 0.00, 0.30),
    "tail_head":        (0.80, 1.00, 0.00, 0.35),
    "chest_front":      (0.00, 0.30, 0.20, 0.70),
    "chest_bottom":     (0.15, 0.50, 0.45, 0.85),
    "shoulder_left":    (0.10, 0.40, 0.15, 0.60),
    "shoulder_right":   (0.10, 0.40, 0.15, 0.60),
    "hip_bone_left":    (0.65, 0.95, 0.05, 0.45),
    "hip_bone_right":   (0.65, 0.95, 0.05, 0.45),
    "hook_left":        (0.65, 0.95, 0.05, 0.45),
    "hook_right":       (0.65, 0.95, 0.05, 0.45),
    "pin_left":         (0.78, 1.00, 0.10, 0.50),
    "pin_right":        (0.78, 1.00, 0.10, 0.50),
    "knee_left":        (0.05, 0.40, 0.50, 0.85),
    "knee_right":       (0.05, 0.40, 0.50, 0.85),
    "hock_left":        (0.60, 0.95, 0.50, 0.85),
    "hock_right":       (0.60, 0.95, 0.50, 0.85),
    "pastern_left":     (0.00, 1.00, 0.75, 1.00),
    "pastern_right":    (0.00, 1.00, 0.75, 1.00),
    "hoof_left":        (0.00, 1.00, 0.85, 1.00),
    "hoof_right":       (0.00, 1.00, 0.85, 1.00),
}


def _relative(kp, mask, sign):
    """Position within the silhouette, oriented front-to-rear."""
    ys, xs = np.nonzero(mask)
    if ys.size < 64:
        return None
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    if x1 <= x0 or y1 <= y0:
        return None
    fx = (kp[0] - x0) / (x1 - x0)
    fy = (kp[1] - y0) / (y1 - y0)
    # facing_sign returns -1 for an animal facing LEFT - head already at x=0,
    # nothing to do - and +1 for one facing right, where x must be flipped so
    # that 0 is always the head. Getting this backwards mirrors the whole
    # report and puts every landmark at the wrong end of the animal.
    return ((1.0 - fx) if sign is not None and sign > 0 else fx), fy


def run(folder: str, limit: int = 40):
    files = [p for p in sorted(Path(folder).rglob("*"))
             if p.suffix.lower() in IMAGE_SUFFIXES][:limit]
    inside = collections.Counter()
    total = collections.Counter()
    xs_seen = collections.defaultdict(list)
    ys_seen = collections.defaultdict(list)

    for f in files:
        try:
            a = detect_animal(str(f))
            if a is None:
                continue
            kps = extract_keypoints(str(f), a.bbox)
            mask, degraded = segment_animal(str(f), a.bbox)
            if degraded:
                continue
            # Audit what the PIPELINE uses, not what the model alone returned -
            # some landmarks are derived from the silhouette, and one
            # (chest_front) is deliberately replaced. Skipping this step made
            # the report describe a stage that nothing downstream sees.
            kps, _prov = add_derived_landmarks(kps, mask, a.bbox)
            sign = facing_sign(kps, a.bbox)
        except Exception:
            continue
        for name, box in EXPECTED.items():
            v = kps.get(name)
            if not v or v[2] <= 0:
                continue
            rel = _relative(v, mask, sign)
            if rel is None:
                continue
            fx, fy = rel
            total[name] += 1
            xs_seen[name].append(fx)
            ys_seen[name].append(fy)
            lo_x, hi_x, lo_y, hi_y = box
            if lo_x <= fx <= hi_x and lo_y <= fy <= hi_y:
                inside[name] += 1

    print(f"\n{len(files)} images. x=0 at the head, 1 at the tail; "
          f"y=0 on the topline, 1 on the ground.\n")
    print(f"{'landmark':<18}{'n':>4}{'in place':>10}{'median x':>10}"
          f"{'expected x':>13}{'median y':>10}{'expected y':>13}")
    print("-" * 78)
    rows = sorted(EXPECTED, key=lambda n: (inside[n] / total[n]) if total[n] else -1)
    for name in rows:
        n = total[name]
        if not n:
            print(f"{name:<18}{0:>4}{'never seen':>10}")
            continue
        lo_x, hi_x, lo_y, hi_y = EXPECTED[name]
        print(f"{name:<18}{n:>4}{inside[name]/n*100:>9.0f}%"
              f"{statistics.median(xs_seen[name]):>10.2f}"
              f"{f'{lo_x:.2f}-{hi_x:.2f}':>13}"
              f"{statistics.median(ys_seen[name]):>10.2f}"
              f"{f'{lo_y:.2f}-{hi_y:.2f}':>13}")

    ok = [n for n in EXPECTED if total[n] and inside[n] / total[n] >= 0.5]
    bad = [n for n in EXPECTED if total[n] and inside[n] / total[n] < 0.5]
    print(f"\nland where they should ({len(ok)}): {', '.join(sorted(ok))}")
    print(f"do NOT ({len(bad)}): {', '.join(sorted(bad))}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else ".",
        int(sys.argv[2]) if len(sys.argv) > 2 else 40)
