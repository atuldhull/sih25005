"""Tests for the knowledge-graph screening layer. Needs MongoDB running.

Run:  venv\\Scripts\\python test_vkg.py
"""
from datetime import date

import reports
import vkg
from fastapi.testclient import TestClient
from main import app, db

client = TestClient(app)


def sv(*pairs):
    return [{"symptom": s, "confidence": c, "region": "test", "source": "test"}
            for s, c in pairs]


def main():
    risks = vkg.estimate_risks(sv(("skin_nodules", 0.9)))
    assert risks[0]["condition"] == "lumpy_skin_disease", risks
    assert risks[0]["urgency"] == "high"
    assert risks[0]["action"] == "refer to vet urgently"
    assert vkg.needs_escalation(risks)
    print(f"PASS  skin nodules -> LSD top risk ({risks[0]['risk']}, "
          f"score {risks[0]['score']}), escalates")

    risks = vkg.estimate_risks(sv(("gait_asymmetry", 0.8), ("hoof_abnormality", 0.7)))
    assert risks[0]["condition"] == "hoof_lameness"
    assert set(risks[0]["because_of"]) == {"gait_asymmetry", "hoof_abnormality"}
    print(f"PASS  gait + hoof -> hoof lameness top (score {risks[0]['score']}), "
          "traceable to both symptoms")

    assert vkg.estimate_risks([]) == []
    assert vkg.estimate_risks(sv(("made_up_symptom", 0.9))) == []
    print("PASS  empty / unknown symptoms -> no risks, no crash")

    # same input twice -> byte-identical output (determinism)
    a = vkg.estimate_risks(sv(("udder_swelling", 0.75)))
    b = vkg.estimate_risks(sv(("udder_swelling", 0.75)))
    assert a == b and a[0]["condition"] == "mastitis"
    print("PASS  deterministic: same symptoms -> identical risk report")

    animal = db.animals.find_one({"_id": "356279812345"})
    rep = reports.build_reports(animal, a, sv(("udder_swelling", 0.75)), [])
    assert "Mastitis" in rep["farmer"] and "Mastitis" in rep["vet"]
    assert reports.DISCLAIMER in rep["farmer"] and reports.DISCLAIMER in rep["vet"]
    print("PASS  farmer + vet reports built, both carry the disclaimer")

    # herd signal: 3 distinct Anand animals with skin_nodules recently
    db.sessions.delete_many({"session_id": {"$regex": "^vkgtest-"}})
    db.animals.delete_many({"_id": {"$regex": "^99999999"}})
    today = date.today().isoformat()
    for i, aid in enumerate(["999999990001", "999999990002", "999999990003"]):
        db.animals.insert_one({"_id": aid, "village": "VKGTestVillage",
                               "species": "cattle", "breed": "Gir",
                               "lactation_no": 1, "last_calving_date": today})
        db.sessions.insert_one({
            "session_id": f"vkgtest-{i}", "animal_id": aid, "date": today,
            "result": {"symptom_vector": sv(("skin_nodules", 0.8))},
        })
    count = vkg.herd_symptom_count(db, "VKGTestVillage", "skin_nodules")
    assert count == 3, count
    print("PASS  herd signal: 3 distinct animals in village counted -> outbreak")
    db.sessions.delete_many({"session_id": {"$regex": "^vkgtest-"}})
    db.animals.delete_many({"_id": {"$regex": "^99999999"}})

    r = client.get("/alerts")
    assert r.status_code == 200 and "alerts" in r.json()
    print(f"PASS  /alerts feed serves ({len(r.json()['alerts'])} alerts stored)")


if __name__ == "__main__":
    main()
