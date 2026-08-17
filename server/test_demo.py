"""Tests for the demo console + demo story. Needs MongoDB running.

Run:  venv\\Scripts\\python test_demo.py
"""
import shutil
from pathlib import Path

import demo_seed
from fastapi.testclient import TestClient
from main import app, db
from validate_result import validate  # path set up via scoring_loader import

client = TestClient(app)


def main():
    demo_seed.main()
    print("PASS  demo story seeded (stage reset + rebuild)")

    r = client.get("/session/demo-star-3")
    assert r.status_code == 200, r.text
    doc = r.json()
    assert doc["animal_id"] == "356279812345"
    assert len(doc["result"]["traits"]) == 20
    assert "_id" not in doc and "files" not in doc
    print("PASS  GET /session returns the stored scorecard (private fields "
          "stripped)")

    assert client.get("/session/nope-000").status_code == 404
    print("PASS  unknown session -> 404")

    # every stored demo result must be contract-true in FULL mode, carry
    # a REAL eligibility reason (in the 30-90 window), and never leak
    # the engine-debug string
    for sess in db.sessions.find({"session_id": {"$regex": "^demo-"}}):
        res = sess["result"]
        problems = validate(res, mode="full")
        assert not problems, f"{sess['session_id']}: {problems[:3]}"
        assert res["eligible_reason"].startswith("first lactation, day"), \
            res["eligible_reason"]
        day = int(res["eligible_reason"].split("day ")[1].split()[0])
        assert 30 <= day <= 90, f"{sess['session_id']} day {day} out of window"
        assert res["engine"] == "baseline" or res["engine"] == "ml-pipeline", \
            res["engine"]
        for key in ("risk_report", "herd_alerts", "reports", "escalated"):
            assert key in res, f"{sess['session_id']} missing {key}"
    print("PASS  all demo results contract-valid (full mode), real "
          "eligibility reasons, no debug leak")

    h = client.get("/animal/356279812345/history").json()["sessions"]
    star = [s for s in h if s["session_id"].startswith("demo-star")]
    assert len(star) == 3
    weights = [s["weight_kg_mid"] for s in sorted(star, key=lambda s: s["date"])]
    assert weights == sorted(weights) and weights[0] < weights[-1], weights
    print(f"PASS  rising weight trend in history: {weights}")

    # the star must still be scorable LIVE today (calving rewritten to -70)
    a = client.get("/animal/356279812345").json()
    assert a["eligible"], a["eligible_reason"]
    print(f"PASS  star still eligible for a live re-scan ({a['eligible_reason']})")

    # overlays must render on the bundled REAL photos, not a placeholder
    scored = next(t["name"] for t in doc["result"]["traits"]
                  if t["score"] is not None and t["view"] == "side")
    ov = client.get(f"/overlays/demo-star-3/{scored.replace(' ', '_')}.jpg")
    assert ov.status_code == 200 and len(ov.content) > 20000, \
        f"{ov.status_code}, {len(ov.content)} bytes - placeholder-sized?"
    print(f"PASS  overlay renders on the real demo photo "
          f"({len(ov.content)//1024} KB)")

    alerts = client.get("/alerts").json()["alerts"]
    assert any(a["animal_id"] == "356279812347" for a in alerts), \
        "lameness escalation missing from officer feed"
    assert any(a.get("herd_alerts") for a in alerts), \
        "no alert carries the herd/outbreak signal"
    assert all("session_id" in a for a in alerts)
    print(f"PASS  officer feed has the outbreak + lameness cases "
          f"({len(alerts)} alerts)")

    r2 = client.get("/demo")
    assert r2.status_code == 200 and "Scoring Console" in r2.text
    print(f"PASS  /demo console serves ({len(r2.text)} bytes)")

    # outbreak cluster: 3 distinct Anand animals already flagged
    import vkg
    n = vkg.herd_symptom_count(db, "Anand", "skin_nodules")
    assert n >= 3, f"outbreak cluster incomplete: {n}"
    print(f"PASS  outbreak cluster live: {n} affected animals in Anand")

    # ---- dress rehearsal of the LIVE ON-STAGE TRIGGER ----
    # scoring 356279812351 (moved to Anand, tag%5==1 -> nodules) must
    # raise the village outbreak to 4 animals, live, via POST /session
    sid = "test-live-trigger"
    db.sessions.delete_one({"session_id": sid})
    db.vet_alerts.delete_one({"session_id": sid})
    files = {"side_photo": ("side.jpg", b"x" * 100, "image/jpeg"),
             "rear_photo": ("rear.jpg", b"x" * 100, "image/jpeg")}
    rr = client.post("/session", data={"animal_id": "356279812351",
                                       "device_session_id": sid}, files=files)
    assert rr.status_code == 200, rr.text
    live = rr.json()
    assert live["symptom_vector"] and \
        live["symptom_vector"][0]["symptom"] == "skin_nodules", \
        live["symptom_vector"]
    assert live["herd_alerts"], "live outbreak signal did not fire"
    assert live["herd_alerts"][0]["animals_affected_14d"] == 4, \
        live["herd_alerts"]
    assert live["escalated"] is True
    assert live["risk_report"] and live["risk_report"][0].get("label")
    assert not validate(live, mode="full")
    print(f"PASS  LIVE TRIGGER rehearsal: scoring ...351 fires the outbreak "
          f"({live['herd_alerts'][0]['animals_affected_14d']} animals, "
          f"top risk: {live['risk_report'][0]['label']})")

    # oversized upload must be refused with 413, not fill the disk
    big = {"side_photo": ("side.jpg", b"x" * (10 * 1024 * 1024 + 1),
                          "image/jpeg"),
           "rear_photo": ("rear.jpg", b"x" * 100, "image/jpeg")}
    rb = client.post("/session", data={"animal_id": "356279812345",
                                       "device_session_id": "test-too-big"},
                     files=big)
    assert rb.status_code == 413, rb.status_code
    print("PASS  oversized photo refused with 413")

    # cleanup the rehearsal so the officer feed is demo-clean afterwards
    db.sessions.delete_many({"session_id": {"$regex": "^test-"}})
    db.vet_alerts.delete_many({"session_id": {"$regex": "^test-"}})
    for p in (Path(__file__).parent / "uploads").glob("test-*"):
        shutil.rmtree(p, ignore_errors=True)
    for p in (Path(__file__).parent / "overlays_cache").glob("test-*"):
        shutil.rmtree(p, ignore_errors=True)
    print("PASS  rehearsal artifacts cleaned up")


if __name__ == "__main__":
    main()
