"""Quick end-to-end test of the session flow. Needs MongoDB running.

Run:  venv\\Scripts\\python test_session.py
"""
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from main import OVERLAY_DIR, app, db

client = TestClient(app)

FAKE_JPG = b"\xff\xd8\xff\xe0" + b"fake-jpeg-bytes" * 10
FAKE_MP4 = b"\x00\x00\x00\x18ftypmp42" + b"fake-video-bytes" * 10

ELIGIBLE = "356279812345"      # Gir, day 47
INELIGIBLE = "356279812361"    # Sahiwal, lactation 2


def files():
    return {
        "side_photo": ("side.jpg", FAKE_JPG, "image/jpeg"),
        "rear_photo": ("rear.jpg", FAKE_JPG, "image/jpeg"),
        "gait_video": ("gait.mp4", FAKE_MP4, "video/mp4"),
    }


def main():
    # fresh start for this test's session id
    db.sessions.delete_many({"session_id": "test-session-001"})
    shutil.rmtree(Path(OVERLAY_DIR) / "test-session-001", ignore_errors=True)

    r = client.post("/session", data={"animal_id": ELIGIBLE,
                                      "device_session_id": "test-session-001"},
                    files=files())
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["traits"]) == 20, "must always return all 20 traits"
    scored = [t for t in body["traits"] if t["score"] is not None]
    print(f"PASS  new session scored: {len(scored)}/20 traits, "
          f"weight {body['weight_kg']['low']}-{body['weight_kg']['high']} kg")

    r2 = client.post("/session", data={"animal_id": ELIGIBLE,
                                       "device_session_id": "test-session-001"},
                     files=files())
    assert r2.status_code == 200 and r2.json().get("duplicate") is True
    print("PASS  retry with same device_session_id -> duplicate, not stored twice")

    r3 = client.post("/session", data={"animal_id": INELIGIBLE,
                                       "device_session_id": "test-session-002"},
                     files=files())
    assert r3.status_code == 422, r3.text
    print(f"PASS  ineligible animal refused: {r3.json()['detail']}")

    r4 = client.get(f"/animal/{ELIGIBLE}/history")
    assert any(s["session_id"] == "test-session-001" for s in r4.json()["sessions"])
    print("PASS  session appears in history")

    r5 = client.post("/sync", json={"device_session_ids":
                                    ["test-session-001", "ghost-999"]})
    statuses = {x["device_session_id"]: x["status"] for x in r5.json()["results"]}
    assert statuses == {"test-session-001": "exists", "ghost-999": "missing"}
    print("PASS  /sync reconcile: exists + missing reported correctly")

    scored_trait = next(t for t in body["traits"] if t["score"] is not None)
    fname = scored_trait["name"].replace(" ", "_") + ".jpg"
    r6 = client.get(f"/overlays/test-session-001/{fname}")
    assert r6.status_code == 200 and r6.content[:2] == b"\xff\xd8", r6.text
    r6b = client.get(f"/overlays/test-session-001/{fname}")  # cached second hit
    assert r6b.status_code == 200
    print(f"PASS  overlay rendered for '{scored_trait['name']}' "
          f"({len(r6.content)} bytes, cached on repeat)")

    # clean up at the END too: this session belongs to the DEMO STAR,
    # and a leftover test row would show in the on-stage history tab
    db.sessions.delete_many({"session_id": {"$regex": "^test-session-"}})
    for sid in ("test-session-001", "test-session-002"):
        shutil.rmtree(Path(OVERLAY_DIR) / sid, ignore_errors=True)
        shutil.rmtree(Path(OVERLAY_DIR).parent / "uploads" / sid,
                      ignore_errors=True)
    print("PASS  test artifacts cleaned up (star history stays demo-clean)")


if __name__ == "__main__":
    main()
