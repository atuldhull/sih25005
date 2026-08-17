"""Seed the DEMO STORY on top of seed.py's animals. Run before any
demo/rehearsal (idempotent - safe to re-run):

  venv\\Scripts\\python demo_seed.py

The story it creates:
1. GIR 356279812345 ("the star"): three sessions across 5 weeks with a
   RISING weight trend -> history/trend answers in the chatbot and
   dashboard have real data.
2. Village Anand outbreak: three different animals flagged with
   skin_nodules in the last week -> the vet officer's alert feed shows
   a live outbreak signal during the demo.
3. One escalated session for buffalo 356279812347 (lameness) -> the
   alerts feed has an individual case too.

Uses the same scoring path as the live server, so every stored result
is contract-true.
"""
from datetime import date, timedelta

from pymongo import MongoClient

import reports
import vkg
from scoring_loader import score_animal

client = MongoClient("mongodb://127.0.0.1:27017")
db = client["sih25005"]


def make_session(animal_id: str, days_ago: int, session_id: str,
                 weight_mid: int | None = None,
                 symptoms: list | None = None):
    animal = db.animals.find_one({"_id": animal_id})
    if animal is None:
        print(f"  SKIP {animal_id} - run seed.py first")
        return
    result = score_animal("demo_side.jpg", "demo_rear.jpg", "demo_gait.mp4",
                          animal)
    result["session_id"] = session_id
    result["eligible"] = True
    result["eligible_reason"] = "demo session"
    if not result.get("breed_registered"):
        result["breed_registered"] = animal.get("breed")
    if weight_mid is not None:
        result["weight_kg"] = {"low": weight_mid - 12, "high": weight_mid + 12,
                               "method": "girth-length-regression",
                               "cross_check": None}
    if symptoms is not None:
        result["symptom_vector"] = symptoms
        result["risk_report"] = vkg.estimate_risks(symptoms)
        result["health_flags"] = [s["symptom"] for s in symptoms]
        result["escalated"] = vkg.needs_escalation(result["risk_report"])
        result["reports"] = reports.build_reports(animal,
                                                  result["risk_report"],
                                                  symptoms, [])
    day = (date.today() - timedelta(days=days_ago)).isoformat()
    w = result.get("weight_kg") or {}
    mid = None
    if isinstance(w.get("low"), (int, float)) and \
            isinstance(w.get("high"), (int, float)):
        mid = (w["low"] + w["high"]) // 2
    db.sessions.replace_one({"session_id": session_id}, {
        "session_id": session_id, "animal_id": animal_id, "date": day,
        "weight_kg_mid": mid, "health_flags": result.get("health_flags", []),
        "files": {}, "result": dict(result),
    }, upsert=True)
    if result.get("escalated"):
        db.vet_alerts.replace_one({"session_id": session_id}, {
            "session_id": session_id, "animal_id": animal_id,
            "village": animal["village"], "date": day,
            "top_risks": [r["condition"] for r in result["risk_report"][:3]],
            "herd_alerts": [],
            "report_vet": result["reports"]["vet"],
        }, upsert=True)
    print(f"  session {session_id}: {animal['breed']} {animal_id[-4:]} "
          f"day -{days_ago}, weight_mid={mid}, "
          f"escalated={result.get('escalated', False)}")


def main():
    print("demo story: the star animal - rising weight trend")
    make_session("356279812345", 35, "demo-star-1", weight_mid=392)
    make_session("356279812345", 18, "demo-star-2", weight_mid=405)
    make_session("356279812345", 3, "demo-star-3", weight_mid=418)

    print("demo story: village Anand outbreak (skin nodules x3 animals)")
    # the outbreak signal needs 3+ animals in the SAME village - move two
    # demo animals into Anand so the herd machinery has a real cluster
    db.animals.update_many({"_id": {"$in": ["356279812355", "356279812350"]}},
                           {"$set": {"village": "Anand"}})
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


if __name__ == "__main__":
    main()
