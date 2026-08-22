"""Test alternative geometries for traits whose measurements miss their band.

A trait whose values sit consistently OUTSIDE its calibrated range is usually
not measuring an unusual animal - it is measuring a different quantity from the
one the range was written for. Three have already been found this way
(fore_leg_set, shoulder_angle, and foot_angle before it), and each time the
signature was the same: the median parked far outside the band, and a different
reading of the same landmarks put it inside.

This script proposes candidate geometries per trait and reports where each one's
distribution lands. A candidate is only believable if its median moves INSIDE
the band while the alternatives stay out - a candidate that merely widens the
spread until it overlaps the band has explained nothing.

    python -m ml.eval_geometry <folder> [limit]
"""
import math
import statistics
import sys
from pathlib import Path

from ml.detection.detector import detect_animal
from ml.pose_features import pose_extractor
from ml.pose_features.pose_extractor import _drop_collapsed

GATE = 0.10
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _interior(v, a, b):
    """Angle at vertex v between the rays to a and b, in degrees."""
    av = (a[0] - v[0], a[1] - v[1])
    bv = (b[0] - v[0], b[1] - v[1])
    ma, mb = math.hypot(*av), math.hypot(*bv)
    if not ma or not mb:
        return None
    c = (av[0] * bv[0] + av[1] * bv[1]) / (ma * mb)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _from_horizontal(p, q):
    return math.degrees(math.atan2(q[1] - p[1], abs(q[0] - p[0])))


def _mid(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


# Each candidate: (label, required joints, fn(joint->xy) -> value or None)
CANDIDATES = {
    "hock_angle  (band 130..160)": [
        ("current: interior at hock, knee->pastern",
         ["knee_left", "hock_left", "pastern_left"],
         lambda g: _interior(g("hock_left"), g("knee_left"), g("pastern_left"))),
        ("rear leg: interior at hock, hip_bone->pastern",
         ["hip_bone_left", "hock_left", "pastern_left"],
         lambda g: _interior(g("hock_left"), g("hip_bone_left"), g("pastern_left"))),
        ("rear leg: interior at hock, hip_bone->hoof",
         ["hip_bone_left", "hock_left", "hoof_left"],
         lambda g: _interior(g("hock_left"), g("hip_bone_left"), g("hoof_left"))),
        ("supplement of current",
         ["knee_left", "hock_left", "pastern_left"],
         lambda g: 180.0 - _interior(g("hock_left"), g("knee_left"), g("pastern_left"))),
    ],
    "rump_angle  (band 0..15)": [
        ("current: hook_mid->pin_mid from horizontal",
         ["hook_left", "pin_left", "hook_right", "pin_right"],
         lambda g: _from_horizontal(_mid(g("hook_left"), g("hook_right")),
                                    _mid(g("pin_left"), g("pin_right")))),
        ("single side: hook_left->pin_left from horizontal",
         ["hook_left", "pin_left"],
         lambda g: _from_horizontal(g("hook_left"), g("pin_left"))),
        ("complement of current (from vertical)",
         ["hook_left", "pin_left", "hook_right", "pin_right"],
         lambda g: 90.0 - abs(_from_horizontal(_mid(g("hook_left"), g("hook_right")),
                                               _mid(g("pin_left"), g("pin_right"))))),
        ("tail_head->pin_mid from horizontal",
         ["tail_head", "pin_left", "pin_right"],
         lambda g: _from_horizontal(g("tail_head"),
                                    _mid(g("pin_left"), g("pin_right")))),
    ],
    "foot_angle  (band 40..65)": [
        ("current: pastern->hoof from horizontal",
         ["pastern_left", "hoof_left"],
         lambda g: _from_horizontal(g("pastern_left"), g("hoof_left"))),
        ("complement (from vertical)",
         ["pastern_left", "hoof_left"],
         lambda g: 90.0 - abs(_from_horizontal(g("pastern_left"), g("hoof_left")))),
        ("hock->hoof from horizontal",
         ["hock_left", "hoof_left"],
         lambda g: _from_horizontal(g("hock_left"), g("hoof_left"))),
    ],
}

BANDS = {"hock_angle  (band 130..160)": (130, 160),
         "rump_angle  (band 0..15)": (0, 15),
         "foot_angle  (band 40..65)": (40, 65)}


def run(folder: str, limit: int = 60):
    files = [p for p in sorted(Path(folder).rglob("*"))
             if p.suffix.lower() in IMAGE_SUFFIXES][:limit]
    model = pose_extractor._get_model()
    frames = []
    for f in files:
        try:
            a = detect_animal(str(f))
            if a is None:
                continue
            raw = model.extract(str(f), a.bbox)
        except Exception:
            continue
        k = {n: (kp.x, kp.y, kp.confidence if kp.confidence >= GATE else 0.0)
             for n, kp in raw.items()}
        frames.append(_drop_collapsed(k, a.bbox))
    print(f"{len(frames)} images, keypoint gate {GATE}\n")

    for trait, cands in CANDIDATES.items():
        lo, hi = BANDS[trait]
        print(trait)
        for label, need, fn in cands:
            vals = []
            for k in frames:
                if not all(k.get(n, (0, 0, 0))[2] > 0 for n in need):
                    continue
                try:
                    v = fn(lambda n: k[n][:2])
                except Exception:
                    v = None
                if v is not None:
                    vals.append(v)
            if not vals:
                print(f"    {label:<48}  no samples")
                continue
            inb = sum(1 for v in vals if lo <= v <= hi)
            print(f"    {label:<48}  n={len(vals):3d}  median {statistics.median(vals):7.1f}"
                  f"  in-band {inb:3d} ({inb/len(vals)*100:3.0f}%)")
        print()


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else ".",
        int(sys.argv[2]) if len(sys.argv) > 2 else 60)
