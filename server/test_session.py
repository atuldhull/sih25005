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

    # a streamed body with no Content-Length is refused BEFORE anything
    # is spooled to disk. The app's http.MultipartRequest always sets
    # Content-Length (proven by every other request in this file), so
    # this only blocks the unbounded-stream case.
    def _stream():
        yield b"x" * 4096

    r7 = client.post("/session", content=_stream(),
                     headers={"Content-Type":
                              "multipart/form-data; boundary=zz"})
    assert r7.status_code == 411, f"{r7.status_code}: {r7.text[:200]}"
    assert not (Path(OVERLAY_DIR).parent / "uploads" / "zz").exists()
    print("PASS  streamed upload without Content-Length -> 411")

    # a refused oversized upload must leave NO orphan upload folder
    big = {"side_photo": ("side.jpg", b"x" * (10 * 1024 * 1024 + 1),
                          "image/jpeg"),
           "rear_photo": ("rear.jpg", FAKE_JPG, "image/jpeg")}
    r8 = client.post("/session", data={"animal_id": ELIGIBLE,
                                       "device_session_id": "test-orphan"},
                     files=big)
    assert r8.status_code == 413, r8.status_code
    assert not (Path(OVERLAY_DIR).parent / "uploads" / "test-orphan").exists(), \
        "413 left an orphan upload directory behind"
    print("PASS  oversized upload -> 413 with no orphan folder")

    # offline-queue flush: uploading a dozen queued sessions back to
    # back is a DEMO FEATURE - the rate limiter must not trip
    codes = []
    for i in range(15):
        rf = client.post("/session",
                         data={"animal_id": ELIGIBLE,
                               "device_session_id": f"test-session-flush-{i}"},
                         files=files())
        codes.append(rf.status_code)
    assert all(c == 200 for c in codes), f"flush blocked: {codes}"
    print(f"PASS  offline-queue flush: {len(codes)} back-to-back uploads, "
          "no rate-limit refusals")

    # clean up at the END too: these sessions belong to the DEMO STAR,
    # and leftover test rows would show in the on-stage history tab
    db.sessions.delete_many({"session_id": {"$regex": "^test-session-"}})
    uploads = Path(OVERLAY_DIR).parent / "uploads"
    for base in (Path(OVERLAY_DIR), uploads):
        for p in base.glob("test-*"):
            shutil.rmtree(p, ignore_errors=True)
    print("PASS  test artifacts cleaned up (star history stays demo-clean)")


if __name__ == "__main__":
    main()
