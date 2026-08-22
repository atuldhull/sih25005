"""How much of the trait table can this pipeline actually fill, and why not the rest?

Run over a folder of photographs and report, per trait, how often it measures,
how often it scores, and when it does not, which specific reason stopped it.
That last column is the point: "not_measurable" aggregated over twenty traits
tells you nothing, while "this trait needs two landmarks nobody annotated" and
"this trait's rule band does not match the quantity we compute" are different
problems with different fixes.

    python -m ml.eval_coverage <folder> [limit]
"""
import collections
import sys
from pathlib import Path

from ml.config.traits import CONTRACT_TRAITS, TRAIT_REGISTRY
from ml.detection.detector import detect_animal
from ml.measurement.traits import measure_all_traits
from ml.pose_features.pose_extractor import extract_keypoints, usable_joint_count
from ml.scoring.scorer import score_all_traits

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _trait_by_id():
    return {t["trait_id"]: t for t in TRAIT_REGISTRY}


def run(folder: str, limit: int = 40):
    files = [p for p in sorted(Path(folder).rglob("*"))
             if p.suffix.lower() in IMAGE_SUFFIXES][:limit]
    if not files:
        print(f"no images under {folder}")
        return

    defs = _trait_by_id()
    measured = collections.Counter()
    scored = collections.Counter()
    values = collections.defaultdict(list)
    missing_joints = collections.Counter()
    joint_counts = []
    ok_images = 0

    for f in files:
        try:
            animal = detect_animal(str(f))
            if animal is None:
                continue
            kps = extract_keypoints(str(f), animal.bbox)
        except Exception as exc:
            print(f"  {f.name}: {type(exc).__name__}: {exc}")
            continue
        ok_images += 1
        joint_counts.append(usable_joint_count(kps))
        live = {n for n, v in kps.items() if v[2] > 0}

        results = measure_all_traits(kps, None, "cattle", 0.0)
        scores = {s.trait_id: s for s in score_all_traits(results, "cattle")}
        for m in results:
            if m.value is not None:
                measured[m.trait_id] += 1
                values[m.trait_id].append(m.value)
                if scores[m.trait_id].score_1_9 is not None:
                    scored[m.trait_id] += 1
            else:
                # Which required landmark was actually missing? This is the
                # column that turns "not_measurable" into a work item.
                need = defs.get(m.trait_id, {}).get("required_keypoints", [])
                for j in need:
                    if j not in live:
                        missing_joints[j] += 1

    n = max(1, ok_images)
    print(f"\n{ok_images} images, mean usable joints "
          f"{sum(joint_counts)/n:.1f} of 22 trained\n")
    print(f"{'trait':<28}{'class':>6}{'scale':>7}{'measured':>10}{'scored':>8}"
          f"   {'median':>9}  rule band")
    print("-" * 96)
    for t in TRAIT_REGISTRY:
        tid = t["trait_id"]
        from ml.config.rules import SPECIES_RULES as RULES
        band = RULES.get("cattle", {}).get(tid)
        band_s = f"{band['min']:g}..{band['max']:g}" if band else "-"
        vs = sorted(values[tid])
        med = f"{vs[len(vs)//2]:9.1f}" if vs else f"{'-':>9}"
        print(f"{tid:<28}{t['trait_class']:>6}"
              f"{('yes' if t.get('required_scale') else 'no'):>7}"
              f"{measured[tid]:>7}/{ok_images:<3}{scored[tid]:>6}"
              f"   {med}  {band_s}")

    print(f"\nmost-often-missing landmarks (blocking a trait that needed it):")
    for joint, cnt in missing_joints.most_common(12):
        print(f"   {cnt:5d}  {joint}")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    run(folder, lim)
