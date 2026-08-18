"""Reset and seed the DEMO STORY. Run before any demo/rehearsal:

  venv\\Scripts\\python demo_seed.py

This is the STAGE RESET button. Each run:
  - re-seeds the animals (calls seed.py itself - no ordering trap)
  - WIPES sessions, vet alerts and the overlay cache
  - rebuilds the demo story deterministically

The story it creates:
1. GIR 356279812345 ("the star"): calving date is rewritten to 70 days
   ago, so its three history sessions (5 weeks of data) all fall inside
   the day 30-90 scoring window AND the animal is still eligible for a
   LIVE re-scan on stage today. Rising weight trend: 392 -> 405 -> 418.
2. Village Anand outbreak: three different animals flagged with
   skin_nodules in the last week. The THIRD one triggers the herd
   alert (3 affected), so the officer feed already shows OUTBREAK.
3. One escalated lameness session for Murrah buffalo 356279812347.

LIVE TRIGGER (the on-stage moment): score animal 356279812351
(Mehsana buffalo, moved to Anand, day-40 eligible) from the console.
The baseline engine deterministically detects skin_nodules for it
(tag % 5 == 1), the herd count reaches 4 distinct animals, and the
OUTBREAK banner + escalation fire live in front of the judges.

Every stored result passes the contract validator in FULL mode, and
demo sessions get real photos so trait overlays render on an actual
animal, not a placeholder.
"""
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path

from pymongo import MongoClient

import reports
import seed
import vkg
# the BASELINE engine on purpose, never the ml/ hot-swap: the seeded
# story must stay byte-identical no matter what pipeline landed today
from scoring import score_animal
import scoring_loader  # noqa: F401 - imported for the contract sys.path
from validate_result import validate

client = MongoClient("mongodb://127.0.0.1:27017")
db = client["sih25005"]

HERE = Path(__file__).parent
ASSETS = HERE / "demo_assets"
UPLOADS = HERE / "uploads"
OVERLAY_CACHE = HERE / "overlays_cache"

STAR = "356279812345"
LIVE_TRIGGER = "356279812351"


def _place_photos(session_id: str) -> dict:
    """Copy the bundled demo photos into this session's upload folder
    so /overlays renders trait points on a real animal."""
    files = {}
    dest = UPLOADS / session_id
    for name in ("side.jpg", "rear.jpg"):
        src = ASSETS / name
        if not src.exists():
            print(f"  WARN demo_assets/{name} missing - overlays will "
                  "use the neutral placeholder")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest / name)
        files[name] = str(dest / name)
    return files


def make_session(animal_id: str, days_ago: int, session_id: str,
                 weight_mid: int | None = None,
                 symptoms: list | None = None):
    animal = db.animals.find_one({"_id": animal_id})
    if animal is None:
        print(f"  SKIP {animal_id} - not in animals collection")
        return
    symptoms = symptoms or []
    day = date.today() - timedelta(days=days_ago)

    result = score_animal("demo_side.jpg", "demo_rear.jpg", "demo_gait.mp4",
                          animal)
    result["engine"] = "baseline"
    result["session_id"] = session_id

    # eligibility as it was ON THE SESSION DATE, same wording as rules.py
    calving = datetime.strptime(animal["last_calving_date"],
                                "%Y-%m-%d").date()
    dim = (day - calving).days
    if not 30 <= dim <= 90:
        print(f"  WARN {session_id}: day {dim} is outside the 30-90 window")
    result["eligible"] = True
    result["eligible_reason"] = f"first lactation, day {dim} after calving"
    if not result.get("breed_registered"):
        result["breed_registered"] = animal.get("breed")
    result["captured_at"] = f"{day.isoformat()}T11:30:00+05:30"

    if weight_mid is not None:
        result["weight_kg"] = {"low": weight_mid - 12, "high": weight_mid + 12,
                               "method": "girth-length-regression",
                               "cross_check": None}

    # normalize the health layer for EVERY session - same path as the
    # live server, so nothing in a stored doc renders as 'undefined'.
    # health_flags use the ENGINE's vocabulary so seeded and live
    # sessions read identically on the history badges
    flag_map = {"skin_nodules": "visible_abnormality",
                "gait_asymmetry": "locomotion_abnormal",
                "hoof_abnormality": "locomotion_abnormal"}
    result["symptom_vector"] = symptoms
    result["health_flags"] = sorted({flag_map.get(s["symptom"],
                                                  s["symptom"])
                                     for s in symptoms})
    risks = vkg.estimate_risks(symptoms)
    result["risk_report"] = risks
    herd_alerts = []
    for s in symptoms:
        others = vkg.herd_symptom_count(db, animal["village"], s["symptom"],
                                        exclude_animal=animal_id)
        if others + 1 >= vkg.OUTBREAK_MIN_ANIMALS:
            herd_alerts.append({"symptom": s["symptom"],
                                "village": animal["village"],
                                "animals_affected_14d": others + 1})
    result["herd_alerts"] = herd_alerts
    result["reports"] = reports.build_reports(animal, risks, symptoms,
                                              herd_alerts)
    result["escalated"] = vkg.needs_escalation(risks) or bool(herd_alerts)

    problems = validate(result, mode="full")
    if problems:
        print(f"  WARN {session_id} violates the contract: {problems[:3]}")

    weight = result.get("weight_kg") or {}
    mid = None
    if isinstance(weight.get("low"), (int, float)) and \
            isinstance(weight.get("high"), (int, float)):
        mid = (weight["low"] + weight["high"]) // 2
    db.sessions.replace_one({"session_id": session_id}, {
        "session_id": session_id, "animal_id": animal_id,
        "date": day.isoformat(),
        "weight_kg_mid": mid, "health_flags": result["health_flags"],
        "files": _place_photos(session_id), "result": dict(result),
    }, upsert=True)
    if result["escalated"]:
        # exact same alert schema the live server writes
        db.vet_alerts.replace_one({"session_id": session_id}, {
            "session_id": session_id,
            "animal_id": animal_id,
            "village": animal["village"],
            "date": day.isoformat(),
            "top_risks": [r.get("label") or r["condition"]
                          for r in risks[:3]],
            "herd_alerts": list(herd_alerts),
            "report_vet": result["reports"]["vet"],
        }, upsert=True)
    print(f"  session {session_id}: {animal['breed']} {animal_id[-4:]} "
          f"day -{days_ago}, weight_mid={mid}, "
          f"escalated={result['escalated']}, herd={len(herd_alerts)}")


def main():
    print("stage reset: re-seeding animals, wiping sessions + alerts + "
          "overlay cache")
    seed.main()
    db.sessions.delete_many({})
    db.vet_alerts.delete_many({})
    shutil.rmtree(OVERLAY_CACHE, ignore_errors=True)
    # every upload dir is an orphan once sessions are wiped - clear
    # rehearsal leftovers and 413-refused partials too
    shutil.rmtree(UPLOADS, ignore_errors=True)

    # the outbreak needs 3+ flagged animals in ONE village, plus a 4th
    # eligible animal held back as the live on-stage trigger
    db.animals.update_many(
        {"_id": {"$in": ["356279812355", "356279812350", LIVE_TRIGGER]}},
        {"$set": {"village": "Anand"}})
    # star: calving 70 days ago -> history at days 35/52/67 (all inside
    # the 30-90 window) and still eligible for a live re-scan today
    db.animals.update_one(
        {"_id": STAR},
        {"$set": {"last_calving_date":
                  (date.today() - timedelta(days=70)).isoformat()}})

    print("demo story: the star animal - rising weight trend")
    make_session(STAR, 35, "demo-star-1", weight_mid=392)
    make_session(STAR, 18, "demo-star-2", weight_mid=405)
    make_session(STAR, 3, "demo-star-3", weight_mid=418)

    print("demo story: village Anand outbreak (skin nodules x3 animals)")
    nodules = [{"symptom": "skin_nodules", "confidence": 0.82,
                "region": "skin", "source": "photo"}]
    make_session("356279812346", 5, "demo-outbreak-1", weight_mid=380,
                 symptoms=nodules)
    make_session("356279812355", 4, "demo-outbreak-2", weight_mid=401,
                 symptoms=nodules)
    make_session("356279812350", 2, "demo-outbreak-3", weight_mid=366,
                 symptoms=nodules)

    print("demo story: individual lameness escalation (Murrah buffalo)")
    make_session("356279812347", 1, "demo-lame-1", weight_mid=498,
                 symptoms=[{"symptom": "gait_asymmetry", "confidence": 0.81,
                            "region": "legs", "source": "video"},
                           {"symptom": "hoof_abnormality", "confidence": 0.74,
                            "region": "legs", "source": "photo"}])

    n = db.sessions.count_documents({"session_id": {"$regex": "^demo-"}})
    a = db.vet_alerts.count_documents({})
    print(f"done: {n} demo sessions, {a} vet alerts in the feed")

    # pre-render every overlay now: first tap on stage is instant, and
    # any render problem surfaces HERE instead of live
    from overlays import render_overlay
    rendered = 0
    for sess in db.sessions.find({"session_id": {"$regex": "^demo-"}}):
        files = sess.get("files", {})
        out_dir = OVERLAY_CACHE / sess["session_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        for t in sess["result"]["traits"]:
            if t["score"] is None or not t.get("overlay_points"):
                continue
            source = (files.get("rear.jpg") if t["view"] in ("rear", "video")
                      else files.get("side.jpg"))
            slug = "".join(c for c in t["name"].lower() if c.isalnum())
            render_overlay(source, t, out_dir / f"{slug}.jpg")
            rendered += 1
    print(f"pre-rendered {rendered} trait overlays into the cache")
    print()
    print("RUNBOOK - the live moments:")
    print(f"  1. History/trend + chatbot: use star Gir {STAR}")
    print(f"  2. LIVE OUTBREAK: score {LIVE_TRIGGER} (Mehsana buffalo, "
          "Anand) from the console -> skin nodules detected -> 4th "
          "animal in the village -> OUTBREAK banner fires on stage")
    print("  3. Officer feed (/alerts tab) already shows the outbreak "
          "cluster + the lameness case")


if __name__ == "__main__":
    main()
