from datetime import date, datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pymongo import MongoClient

from scoring import score_animal

app = FastAPI(title="SIH25005 Backend")

client = MongoClient("mongodb://127.0.0.1:27017", serverSelectionTimeoutMS=3000)
db = client["sih25005"]

UPLOAD_DIR = Path(__file__).parent / "uploads"


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
