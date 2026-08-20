"""Where should the keypoint confidence gate sit?

A gate that is too strict produces nothing; one that is too loose produces
confident nonsense. Neither failure announces itself, so the threshold has to
be chosen against evidence rather than taste.

THE TEST
There is no keypoint ground truth for these photographs, so correctness cannot
be measured directly. What can be measured is whether the extra measurements a
looser gate admits are PLAUSIBLE - do they land inside the trait's calibrated
rule band, or scatter outside it?

That works as a discriminator because the bands were set from breed-standard
anatomy and are narrow relative to the space of possible values. A genuinely
located joint produces a value inside the band most of the time. A joint the
model has misplaced produces a value drawn from something much wider, and only
lands in-band by luck. So:

    in-band rate holds up as the gate loosens  -> the joints were real and the
                                                  strict gate was discarding them
    in-band rate falls as the gate loosens     -> the gate is admitting noise

The absolute rate is not the interesting number - some traits are genuinely
often out of band. The SHAPE of the curve is what decides the threshold.

    python -m ml.eval_confidence_gate <folder> [limit]
"""
import collections
import sys
from pathlib import Path

from ml.config.rules import SPECIES_RULES
from ml.detection.detector import detect_animal
from ml.measurement import traits as traits_mod
from ml.measurement.traits import measure_all_traits
from ml.pose_features import pose_extractor
from ml.pose_features.pose_extractor import _drop_collapsed

GATES = [0.30, 0.25, 0.20, 0.15, 0.10, 0.05]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _regate(raw, gate, bbox):
    """Re-apply a different confidence gate to already-computed keypoints.

    The model runs once per image; only the threshold changes. Re-running
    inference per gate would be slower and would prove nothing extra.
    """
    out = {}
    for name, kp in raw.items():
        c = float(kp.confidence)
        out[name] = (float(kp.x), float(kp.y), c if c >= gate else 0.0)
    return _drop_collapsed(out, bbox)


def run(folder: str, limit: int = 40):
    files = [p for p in sorted(Path(folder).rglob("*"))
             if p.suffix.lower() in IMAGE_SUFFIXES][:limit]
    bands = SPECIES_RULES.get("cattle", {})

    cached = []
    model = pose_extractor._get_model()
    for f in files:
        try:
            animal = detect_animal(str(f))
            if animal is None:
                continue
            cached.append((model.extract(str(f), animal.bbox), animal.bbox))
        except Exception:
            continue
    print(f"{len(cached)} images\n")

    print(f"{'gate':>6}{'joints':>9}{'measured':>10}{'in band':>10}"
          f"{'in-band rate':>14}   per-trait in-band")
    print("-" * 100)
    for gate in GATES:
        joints = meas = inband = 0
        per_trait = collections.Counter()
        per_trait_tot = collections.Counter()
        # Measurement re-gates independently of the pose stage, so the
        # effective threshold is the stricter of the two. Sweeping only one
        # of them changes nothing, which is exactly what the first run of
        # this script showed.
        traits_mod.KEYPOINT_CONFIDENCE_THRESHOLD = gate
        for raw, bbox in cached:
            kps = _regate(raw, gate, bbox)
            joints += sum(1 for v in kps.values() if v[2] > 0)
            for m in measure_all_traits(kps, None, "cattle", 0.0):
                if m.value is None:
                    continue
                meas += 1
                per_trait_tot[m.trait_id] += 1
                b = bands.get(m.trait_id)
                if b and b["min"] <= m.value <= b["max"]:
                    inband += 1
                    per_trait[m.trait_id] += 1
        rate = (inband / meas * 100) if meas else float("nan")
        top = "  ".join(
            f"{t}:{per_trait[t]}/{per_trait_tot[t]}"
            for t, _ in per_trait_tot.most_common(4))
        print(f"{gate:>6.2f}{joints/max(1,len(cached)):>9.1f}{meas:>10}"
              f"{inband:>10}{rate:>13.1f}%   {top}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else ".",
        int(sys.argv[2]) if len(sys.argv) > 2 else 40)
