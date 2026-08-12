import re
from datetime import date, datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pymongo import MongoClient

from overlays import render_overlay
from scoring import score_animal

app = FastAPI(title="SIH25005 Backend")

client = MongoClient("mongodb://127.0.0.1:27017", serverSelectionTimeoutMS=3000)
db = client["sih25005"]

UPLOAD_DIR = Path(__file__).parent / "uploads"
OVERLAY_DIR = Path(__file__).parent / "overlays_cache"


def _slug(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


def check_eligibility(animal: dict) -> tuple[bool, str]:
    """NDDB rule: type-classify only in FIRST lactation, day 30-90
    after calving. Refusing to score outside this window is a
    feature, not a limitation - the reason string is shown in-app."""
    lact = animal["lactation_no"]
    calving = datetime.strptime(animal["last_calving_date"], "%Y-%m-%d").date()
    days = (date.today() - calving).days

    if lact != 1:
        return False, f"not in first lactation (currently lactation {lact})"
    if days < 30:
        return False, f"only {days} days since calving - scoring allowed from day 30"
    if days > 90:
        return False, f"{days} days since calving - past the day-90 window"
    return True, f"first lactation, day {days} after calving"


@app.get("/ping")
def ping():
    return {"status": "ok", "service": "sih25005-server"}


@app.get("/animal/{animal_id}")
def get_animal(animal_id: str):
    animal = db.animals.find_one({"_id": animal_id})
    if animal is None:
        raise HTTPException(status_code=404, detail="animal not found in BPA records")

    eligible, reason = check_eligibility(animal)
    return {
        "animal_id": animal["_id"],
        "species": animal["species"],
        "breed": animal["breed"],
        "dob": animal["dob"],
        "lactation_no": animal["lactation_no"],
        "last_calving_date": animal["last_calving_date"],
        "owner": animal["owner"],
        "village": animal["village"],
        "eligible": eligible,
        "eligible_reason": reason,
    }


@app.post("/session")
async def create_session(
    animal_id: str = Form(...),
    device_session_id: str = Form(...),
    side_photo: UploadFile = File(...),
    rear_photo: UploadFile = File(...),
    gait_video: UploadFile = File(None),
):
    animal = db.animals.find_one({"_id": animal_id})
    if animal is None:
        raise HTTPException(status_code=404, detail="animal not found in BPA records")

    # idempotency: the app may retry after a half-failed upload.
    # Same device_session_id -> return the stored result, store nothing twice.
    existing = db.sessions.find_one({"session_id": device_session_id})
    if existing is not None:
        response = dict(existing["result"])
        response["duplicate"] = True
        return response

    eligible, reason = check_eligibility(animal)
    if not eligible:
        raise HTTPException(status_code=422, detail=f"animal not eligible: {reason}")

    session_dir = UPLOAD_DIR / device_session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    for filename, upload in [("side.jpg", side_photo), ("rear.jpg", rear_photo),
                             ("gait.mp4", gait_video)]:
        if upload is not None:
            (session_dir / filename).write_bytes(await upload.read())
            saved[filename] = str(session_dir / filename)

    result = score_animal(saved.get("side.jpg"), saved.get("rear.jpg"),
                          saved.get("gait.mp4"), animal)
    result["session_id"] = device_session_id
    result["eligible"] = eligible
    result["eligible_reason"] = reason

    weight = result["weight_kg"]
    db.sessions.insert_one({
        "session_id": device_session_id,
        "animal_id": animal_id,
        "date": date.today().isoformat(),
        "weight_kg_mid": (weight["low"] + weight["high"]) // 2,
        "health_flags": result["health_flags"],
        "files": saved,
        "result": dict(result),
    })
    return result


class SyncCheckRequest(BaseModel):
    device_session_ids: list[str]


@app.post("/sync")
def sync_check(body: SyncCheckRequest):
    """Reconcile the app's offline queue: for each queued id, say
    whether the server already has it. exists -> app marks it synced;
    missing -> app re-uploads via POST /session (which is retry-safe)."""
    results = []
    for sid in body.device_session_ids:
        exists = db.sessions.find_one({"session_id": sid}) is not None
        results.append({"device_session_id": sid,
                        "status": "exists" if exists else "missing"})
    return {"results": results}


@app.get("/overlays/{session_id}/{trait_file}")
def get_overlay(session_id: str, trait_file: str):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", session_id):
        raise HTTPException(status_code=404, detail="unknown session")

    target = _slug(trait_file.rsplit(".", 1)[0])
    cached = OVERLAY_DIR / session_id / f"{target}.jpg"
    if cached.exists():
        return FileResponse(cached, media_type="image/jpeg")

    session = db.sessions.find_one({"session_id": session_id})
    if session is None:
        raise HTTPException(status_code=404, detail="unknown session")

    trait = next((t for t in session["result"]["traits"]
                  if _slug(t["name"]) == target), None)
    if trait is None:
        raise HTTPException(status_code=404, detail="unknown trait name")
    if not trait.get("overlay_points"):
        raise HTTPException(status_code=404,
                            detail="trait was not scored - no overlay available")

    files = session.get("files", {})
    source = (files.get("rear.jpg") if trait["view"] in ("rear", "video")
              else files.get("side.jpg"))
    render_overlay(source, trait, cached)
    return FileResponse(cached, media_type="image/jpeg")


@app.get("/animal/{animal_id}/history")
def get_history(animal_id: str):
    if db.animals.find_one({"_id": animal_id}) is None:
        raise HTTPException(status_code=404, detail="animal not found in BPA records")

    sessions = []
    for s in db.sessions.find({"animal_id": animal_id}).sort("date", -1):
        sessions.append({
            "session_id": s["session_id"],
            "date": s["date"],
            "weight_kg_mid": s.get("weight_kg_mid"),
            "health_flags": s.get("health_flags", []),
        })
    return {"animal_id": animal_id, "sessions": sessions}
