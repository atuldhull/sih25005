"""Tests for the demo console + demo story. Needs MongoDB running.

Run:  venv\\Scripts\\python test_demo.py
"""
import demo_seed
from fastapi.testclient import TestClient
from main import app, db

client = TestClient(app)


def main():
    demo_seed.main()
    print("PASS  demo story seeded (idempotent)")

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

    h = client.get("/animal/356279812345/history").json()["sessions"]
    star = [s for s in h if s["session_id"].startswith("demo-star")]
    assert len(star) == 3
    weights = [s["weight_kg_mid"] for s in sorted(star, key=lambda s: s["date"])]
    assert weights == sorted(weights) and weights[0] < weights[-1], weights
    print(f"PASS  rising weight trend in history: {weights}")

    alerts = client.get("/alerts").json()["alerts"]
    assert any(a["animal_id"] == "356279812347" for a in alerts), \
        "lameness escalation missing from officer feed"
    print(f"PASS  officer feed populated ({len(alerts)} alerts)")

    r2 = client.get("/demo")
    assert r2.status_code == 200 and "Scoring Console" in r2.text
    print(f"PASS  /demo console serves ({len(r2.text)} bytes)")

    # outbreak check: 3 distinct Anand animals with skin_nodules -> the
    # herd signal fires for any NEW Anand session during the live demo
    import vkg
    n = vkg.herd_symptom_count(db, "Anand", "skin_nodules")
    assert n >= 3, f"outbreak cluster incomplete: {n}"
    print(f"PASS  outbreak cluster live: {n} affected animals in Anand")


if __name__ == "__main__":
    main()
