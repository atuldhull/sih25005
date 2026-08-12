from datetime import date, datetime

from fastapi import FastAPI, HTTPException
from pymongo import MongoClient

app = FastAPI(title="SIH25005 Backend")

client = MongoClient("mongodb://127.0.0.1:27017", serverSelectionTimeoutMS=3000)
db = client["sih25005"]


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
