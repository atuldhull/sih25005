"""Person 2 self-check - run this BEFORE pushing ml/pipeline.py:

  venv\\Scripts\\python ..\\contract\\check_pipeline.py        (from server/)
  venv\\Scripts\\python contract\\check_pipeline.py            (from repo root)

It performs EXACTLY the adoption test the server's scoring_loader runs:
import ml.pipeline, score a realistic animal, validate the result
against the frozen contract, and check that at least one trait scored.
The verdict tells you what the live server would do with this build -
no server, no MongoDB, no push needed.

Optionally score real files:
  ... check_pipeline.py --side path\\to\\side.jpg --rear path\\to\\rear.jpg
(defaults to the bundled demo photos in server/demo_assets/)
"""
import argparse
import importlib
import sys
import traceback
from datetime import date, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "contract"))
from validate_result import validate  # noqa: E402

# realistic BPA records, same shape seed.py writes (no DB needed).
# 351 is the demo's live outbreak trigger - see the note printed below.
def _animal(aid, species, breed, days_since_calving, owner, village):
    calving = date.today() - timedelta(days=days_since_calving)
    return {"_id": aid, "species": species, "breed": breed,
            "sex": "female", "dob": (calving - timedelta(days=1095)
                                     ).isoformat(),
            "lactation_no": 1, "last_calving_date": calving.isoformat(),
            "owner": owner, "village": village}


SAMPLES = [
    _animal("356279812345", "cattle", "Gir", 70, "Ramesh Kumar", "Anand"),
    _animal("356279812351", "buffalo", "Mehsana", 40, "Kiran Chaudhary",
            "Anand"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default=str(REPO / "server" / "demo_assets"
                                          / "side.jpg"))
    ap.add_argument("--rear", default=str(REPO / "server" / "demo_assets"
                                          / "rear.jpg"))
    ap.add_argument("--video", default=None)
    args = ap.parse_args()

    print("pipeline self-check (mirrors the server's scoring_loader)")
    print("=" * 58)

    # 1. import - the loader retries this every 30 s on the server
    sys.path.insert(0, str(REPO))
    try:
        importlib.invalidate_caches()
        mod = importlib.import_module("ml.pipeline")
        score = mod.score_animal
        print("[OK]   ml.pipeline imports and has score_animal")
    except Exception:
        print("[FAIL] ml.pipeline NOT importable - the server would keep "
              "using the baseline engine. The error:")
        traceback.print_exc()
        return 1

    ok = True
    for animal in SAMPLES:
        tag = animal["_id"]
        print(f"\nscoring {animal['breed']} {tag} "
              f"({args.side!r}, {args.rear!r}, video={args.video!r})")
        try:
            result = score(args.side, args.rear, args.video, animal)
        except Exception:
            print("[FAIL] score_animal CRASHED - the server would fall "
                  "back to baseline for this call:")
            traceback.print_exc()
            ok = False
            continue

        if not isinstance(result, dict):
            print(f"[FAIL] score_animal returned {type(result).__name__}, "
                  "not a dict - the server would keep the baseline. It "
                  "must return the contract/scoring_result.json shape.")
            ok = False
            continue

        try:
            problems = validate(result, mode="pipeline")
        except Exception as e:
            print(f"[FAIL] the contract validator could not even read this "
                  f"result ({type(e).__name__}: {e}) - check that 'traits' "
                  "is a list of dicts and every field exists.")
            ok = False
            continue

        if problems:
            print(f"[FAIL] {len(problems)} contract violation(s) - the "
                  "server would REJECT this output and use baseline:")
            for p in problems[:12]:
                print(f"       - {p}")
            if len(problems) > 12:
                print(f"       ... and {len(problems) - 12} more")
            ok = False
        else:
            print("[OK]   contract valid (pipeline mode)")

        traits = result.get("traits")
        scored = sum(1 for t in traits
                     if isinstance(t, dict) and t.get("score") is not None) \
            if isinstance(traits, list) else 0
        if scored > 0:
            print(f"[OK]   {scored}/20 traits scored")
        else:
            print("[FAIL] 0/20 traits scored - valid shape but the server "
                  "keeps baseline until real scores exist")
            ok = False

        if tag == "356279812351":
            symptoms = {s.get("symptom") for s in
                        result.get("symptom_vector", []) or []}
            if "skin_nodules" in symptoms:
                print("[OK]   demo note: 351 reports skin_nodules - the "
                      "live outbreak moment still fires")
            else:
                print("[NOTE] 351 reports no skin_nodules. Fine for "
                      "production - but the demo's LIVE OUTBREAK moment "
                      "depends on it. Run server\\preflight.py after "
                      "landing; coordinate with Person 3 before demo day.")

    print("\n" + "=" * 58)
    if args.video is None:
        print("[NOTE] the gait-video argument was NOT exercised. The "
              "server passes a real .mp4 path whenever the app attaches "
              "a clip (and None when it does not) - re-run with "
              "--video <file.mp4> to cover that call shape too.")
    if ok:
        print("VERDICT: the server would ADOPT this pipeline "
              "(hot-swap within 30 s of landing, no restart).")
        return 0
    print("VERDICT: the server would KEEP THE BASELINE engine. "
          "Fix the [FAIL] lines above before pushing.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
