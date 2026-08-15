"""Concurrency smoke test against a REAL uvicorn server (not the
TestClient): fires simultaneous chats, lookups, a session upload and
pings, and asserts nothing blocks anything else. The chat model is
pointed at a nonexistent name so answers come from the instant
template path - this tests OUR threading, not Ollama's speed.

Run:  venv\\Scripts\\python test_concurrency.py     (needs MongoDB)
"""
import concurrent.futures
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

PORT = 8001
BASE = f"http://127.0.0.1:{PORT}"
SERVER_DIR = Path(__file__).parent
FAKE_JPG = b"\xff\xd8\xff\xe0" + b"fake" * 20


def wait_up(timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(f"{BASE}/ping", timeout=2.0).status_code == 200:
                return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    env = dict(os.environ)
    env["SIH_CHAT_MODEL"] = "nonexistent-model-for-test"
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--port", str(PORT)],
        cwd=SERVER_DIR, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert wait_up(), "server did not come up on port 8001"
        print("PASS  test server up")

        def chat(i):
            r = httpx.post(f"{BASE}/chat", timeout=30.0,
                           json={"animal_id": "356279812345",
                                 "message": f"what is her weight? ({i})"})
            return r.status_code

        def lookup(i):
            return httpx.get(f"{BASE}/animal/356279812346", timeout=30.0).status_code

        def session(i):
            files = {"side_photo": ("s.jpg", FAKE_JPG, "image/jpeg"),
                     "rear_photo": ("r.jpg", FAKE_JPG, "image/jpeg")}
            r = httpx.post(f"{BASE}/session", timeout=60.0,
                           data={"animal_id": "356279812350",
                                 "device_session_id": f"conc-test-{i}"},
                           files=files)
            return r.status_code

        def ping(i):
            return httpx.get(f"{BASE}/ping", timeout=10.0).status_code

        jobs = [chat] * 4 + [lookup] * 2 + [session] * 2 + [ping] * 2
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            codes = list(ex.map(lambda f_i: f_i[0](f_i[1]),
                                [(f, i) for i, f in enumerate(jobs)]))
        wall = time.time() - start

        assert all(c == 200 for c in codes), codes
        assert wall < 20, f"10 concurrent requests took {wall:.1f}s"
        print(f"PASS  10 concurrent requests (4 chat, 2 lookup, 2 session, "
              f"2 ping) all 200 in {wall:.1f}s")

        # duplicate-retry race: same device_session_id fired twice at once
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f1 = ex.submit(session, "race")
            f2 = ex.submit(session, "race")
            c1, c2 = f1.result(), f2.result()
        assert c1 == 200 and c2 == 200
        r = httpx.get(f"{BASE}/animal/356279812350/history", timeout=10.0).json()
        count = sum(1 for s in r["sessions"]
                    if s["session_id"] == "conc-test-race")
        assert count == 1, f"race produced {count} sessions for one id"
        print("PASS  simultaneous duplicate uploads -> exactly one stored session")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        # clean up test sessions
        from pymongo import MongoClient
        db = MongoClient("mongodb://127.0.0.1:27017")["sih25005"]
        db.sessions.delete_many({"session_id": {"$regex": "^conc-test-"}})
        print("cleaned up test sessions")


if __name__ == "__main__":
    main()
