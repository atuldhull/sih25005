"""The exact field names the phone app must POST to /session.

Written after replaying the app's real request and getting HTTP 422. This is
executable documentation: if either side renames a field, this fails with the
name that broke rather than the app silently failing to upload in the field.

  venv\Scripts\python test_app_contract.py

Mismatches found on 2026-08-20 against app-dev abafa82:

  app sends          server expects       consequence
  -----------------  -------------------  ------------------------------------
  tag_id             animal_id            422, upload rejected outright
  (not sent)         device_session_id    422, upload rejected outright
  video              gait_video           silently ignored - the field is
                                          optional, so the server returns 200
                                          and the gait video never arrives

The third is the dangerous one: it does not fail, it just quietly drops the
video the farmer recorded, and captured.gait_video comes back false.

device_session_id cannot simply be generated server-side. It is the
idempotency key for the offline queue - the same id on a retry means "this is
the same session, do not store it twice" - so a generated one would turn every
retry into a duplicate record.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "contract"))

from fastapi.testclient import TestClient  # noqa: E402

from main import app, db  # noqa: E402

client = TestClient(app)
JPG = b"\xff\xd8\xff\xe0" + b"x" * 200
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"x" * 200
SID = "app-contract-test-001"

REQUIRED_FORM_FIELDS = {"animal_id", "device_session_id"}
REQUIRED_FILE_FIELDS = {"side_photo", "rear_photo"}
OPTIONAL_FILE_FIELDS = {"gait_video"}


def _cleanup():
    db.sessions.delete_many({"session_id": {"$regex": "^app-contract-test-"}})


def check(ok, name, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    return ok


def main():
    _cleanup()
    failures = 0

    r = client.post("/session",
                    data={"animal_id": "356279812345", "device_session_id": SID},
                    files={"side_photo": ("s.jpg", JPG, "image/jpeg"),
                           "rear_photo": ("r.jpg", JPG, "image/jpeg"),
                           "gait_video": ("g.mp4", MP4, "video/mp4")})
    failures += not check(r.status_code == 200,
                          "correct field names are accepted",
                          f"got HTTP {r.status_code}")
    if r.status_code == 200:
        b = r.json()
        failures += not check(b.get("captured", {}).get("gait_video") is True,
                              "gait_video arrives when named correctly")
        # the app's model declares `final bool breedVerified` - non-nullable.
        # A null here is a runtime TypeError on the phone, not a blank field.
        failures += not check(isinstance(b.get("breed_verified"), bool),
                              "breed_verified is a bool, never null",
                              f"got {type(b.get('breed_verified')).__name__} "
                              f"- the app declares it non-nullable")
        failures += not check(b.get("breed_verify_status") in
                              ("unverified", "agree", "disagree"),
                              "breed_verify_status carries the honest state",
                              f"got {b.get('breed_verify_status')!r}")
        # every overlay point must be int - the app does List<int>.from(p)
        bad = [t["name"] for t in b.get("traits", [])
               for p in (t.get("overlay_points") or [])
               if not all(isinstance(v, int) for v in p)]
        failures += not check(not bad, "overlay points are integers",
                              f"float points would crash List<int>.from: {bad[:3]}")
        # eligible_reason is non-nullable in the app's model
        failures += not check(isinstance(b.get("eligible_reason"), str),
                              "eligible_reason is a string, never null")
    _cleanup()

    # The app's OWN field names must work: it sends tag_id, no session id,
    # and names the video 'video'. Before the compatibility shim this
    # returned 422 and the app could not upload at all.
    def app_post():
        return client.post("/session", data={"tag_id": "356279812345"},
                           files={"side_photo": ("s.jpg", JPG, "image/jpeg"),
                                  "rear_photo": ("r.jpg", JPG, "image/jpeg"),
                                  "video": ("g.mp4", MP4, "video/mp4")})

    r2 = app_post()
    failures += not check(r2.status_code == 200,
                          "the app's OWN field names are accepted",
                          f"got HTTP {r2.status_code}")
    if r2.status_code == 200:
        b2 = r2.json()
        # 'video' used to be silently dropped: the field is optional, so the
        # server returned 200 and the farmer's recording simply vanished.
        failures += not check(b2.get("captured", {}).get("gait_video") is True,
                              "a video named 'video' is NOT silently dropped")
        sid = b2.get("session_id")
        failures += not check(bool(sid), "a session id was derived")

        # The derived id is the offline queue's idempotency key. A retry of
        # the SAME upload must collapse to the same session, or every retry
        # becomes a duplicate record.
        r3 = app_post()
        failures += not check(
            r3.status_code == 200 and r3.json().get("session_id") == sid,
            "retrying the same upload is idempotent",
            f"first {sid}, retry {r3.json().get('session_id') if r3.status_code == 200 else r3.status_code}")

        # Different content must NOT collapse onto the same session.
        r4 = client.post("/session", data={"tag_id": "356279812345"},
                         files={"side_photo": ("s.jpg", JPG + b"different",
                                               "image/jpeg"),
                                "rear_photo": ("r.jpg", JPG, "image/jpeg"),
                                "video": ("g.mp4", MP4, "video/mp4")})
        failures += not check(
            r4.status_code == 200 and r4.json().get("session_id") != sid,
            "a retaken photo becomes a NEW session, not a duplicate")
        db.sessions.delete_many({"session_id": {"$regex": "^auto-"}})
    _cleanup()

    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{failures} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
