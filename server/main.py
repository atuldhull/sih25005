import os
import re
import time
from collections import defaultdict, deque
from datetime import date
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

import threading
import uuid

import chat
import reports
import vkg
import voice
from overlays import render_overlay
from rules import check_eligibility
from scoring_loader import engine_status, score_animal

app = FastAPI(title="SIH25005 Backend")

# hackathon LAN: allow browser-based clients too (Flutter web builds,
# quick HTML test pages) - native apps ignore CORS entirely
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# cheap insurance against quota-burn/CPU loops from strangers on the
# network: per-IP sliding-window limits on the expensive endpoints.
# Generous enough that no honest demo ever hits them.
_RATE_LIMITS = {"/chat": (20, 60.0), "/chat/voice": (6, 60.0)}
_rate_lock = threading.Lock()
_rate_hits: dict = defaultdict(deque)


@app.middleware("http")
async def _rate_limit(request, call_next):
    lim = _RATE_LIMITS.get(request.url.path)
    if lim and request.client:
        max_hits, window = lim
        bucket = _rate_hits[(request.client.host, request.url.path)]
        now = time.time()
        with _rate_lock:
            while bucket and now - bucket[0] > window:
                bucket.popleft()
            if len(bucket) >= max_hits:
                return JSONResponse(status_code=429, content={
                    "detail": "too many requests from this device - "
                              "please wait a minute"})
            bucket.append(now)
    return await call_next(request)

client = MongoClient("mongodb://127.0.0.1:27017", serverSelectionTimeoutMS=3000)
db = client["sih25005"]

UPLOAD_DIR = Path(__file__).parent / "uploads"
OVERLAY_DIR = Path(__file__).parent / "overlays_cache"
VOICE_DIR = Path(__file__).parent / "voice_cache"


@app.on_event("startup")
def _startup():
    # sort indexes (sessions are day-granular, _id breaks same-day ties);
    # session_id is UNIQUE so concurrent phone retries cannot double-insert
    try:
        db.sessions.create_index([("date", -1), ("_id", -1)])
        db.animals.create_index("village")
    except Exception:
        pass  # Mongo down: endpoints will surface it per-request
    try:
        db.sessions.drop_index("session_id_1")  # replace old non-unique index
    except Exception:
        pass
    try:
        db.sessions.create_index("session_id", unique=True)
    except Exception as e:
        print(f"[warm] WARNING: unique session index failed: {e}")

    # warm the slow pieces in the background and SAY what happened -
    # a silent warm failure means every chat quietly falls to templates
    def warm():
        try:
            import rag
            stats = rag.ensure_index(db)
            print(f"[warm] knowledge index: {stats}")
        except Exception as e:
            print(f"[warm] knowledge index failed: {e}")
        try:
            voice._get_whisper()
            print("[warm] whisper STT model loaded")
        except Exception as e:
            print(f"[warm] whisper unavailable: {e}")
        try:
            import httpx
            r = httpx.post(f"{chat.OLLAMA_URL}/api/chat", timeout=120.0, json={
                "model": chat.CHAT_MODEL, "stream": False,
                "messages": [{"role": "user", "content": "hi"}],
                "options": {"num_predict": 1},
            })
            print(f"[warm] ollama {chat.CHAT_MODEL}: HTTP {r.status_code}")
        except Exception as e:
            print(f"[warm] ollama unavailable: {e}")
        try:
            # privacy sweep: farmer voice clips older than ~6 hours
            import time
            cutoff = time.time() - 6 * 3600
            for f in VOICE_DIR.glob("*.wav"):
                if f.stat().st_mtime < cutoff:
                    f.unlink(missing_ok=True)
        except Exception:
            pass
    threading.Thread(target=warm, daemon=True).start()


def _slug(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


@app.get("/ping")
def ping():
    import llm_providers
    chain = llm_providers.status()
    if not os.environ.get("SIH_DEBUG"):
        # don't advertise how many keys the team holds to the network
        chain = {"cloud_configured": chain["gemini_keys"] > 0
                 or bool(chain["compat_providers"])}
    return {"status": "ok", "service": "sih25005-server",
            "scoring_engine": engine_status(),
            "llm_chain": chain}


@app.get("/animals")
def list_animals():
    """Roster for demo UIs and the app's animal picker."""
    out = []
    try:
        for a in db.animals.find({}).sort("_id", 1):
            eligible, _ = check_eligibility(a)
            out.append({"animal_id": a["_id"], "species": a["species"],
                        "breed": a["breed"], "village": a["village"],
                        "eligible": eligible})
    except Exception:
        raise HTTPException(status_code=503,
                            detail="database unreachable - start MongoDB "
                                   "(run_server.bat does this)")
    return {"animals": out}


@app.get("/chat-ui")
def chat_ui():
    """The professional chat interface - self-contained, offline-safe."""
    return FileResponse(Path(__file__).parent / "static" / "chat.html",
                        media_type="text/html")


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
def create_session(
    animal_id: str = Form(...),
    device_session_id: str = Form(...),
    side_photo: UploadFile = File(...),
    rear_photo: UploadFile = File(...),
    gait_video: UploadFile = File(None),
):
    # sync on purpose: FastAPI runs sync handlers in a threadpool, so
    # when the real ML pipeline (seconds of CPU) replaces the fake
    # engine, /ping and /chat keep answering while a session scores
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
            (session_dir / filename).write_bytes(upload.file.read())
            saved[filename] = str(session_dir / filename)

    result = score_animal(saved.get("side.jpg"), saved.get("rear.jpg"),
                          saved.get("gait.mp4"), animal)
    result["session_id"] = device_session_id
    result["eligible"] = eligible
    result["eligible_reason"] = reason

    # screening layer: deterministic risk estimation from the symptom
    # vector (Person 2's detectors fill it; the VKG reasons over it)
    risks = vkg.estimate_risks(result["symptom_vector"])
    result["risk_report"] = risks

    herd_alerts = []
    for s in result["symptom_vector"]:
        others = vkg.herd_symptom_count(db, animal["village"], s["symptom"])
        if others + 1 >= vkg.OUTBREAK_MIN_ANIMALS:
            herd_alerts.append({"symptom": s["symptom"],
                                "village": animal["village"],
                                "animals_affected_14d": others + 1})
    result["herd_alerts"] = herd_alerts
    result["reports"] = reports.build_reports(animal, risks,
                                              result["symptom_vector"], herd_alerts)
    result["escalated"] = vkg.needs_escalation(risks) or bool(herd_alerts)
    if result["escalated"]:
        db.vet_alerts.insert_one({
            "animal_id": animal_id,
            "village": animal["village"],
            "date": date.today().isoformat(),
            "top_risks": [r["condition"] for r in risks[:3]],
            "herd_alerts": list(herd_alerts),
            "report_vet": result["reports"]["vet"],
        })

    weight = result["weight_kg"]
    try:
        db.sessions.insert_one({
            "session_id": device_session_id,
            "animal_id": animal_id,
            "date": date.today().isoformat(),
            "weight_kg_mid": (weight["low"] + weight["high"]) // 2,
            "health_flags": result["health_flags"],
            "files": saved,
            "result": dict(result),
        })
    except DuplicateKeyError:
        # a concurrent retry won the race - return the winner's result
        winner = db.sessions.find_one({"session_id": device_session_id})
        response = dict(winner["result"])
        response["duplicate"] = True
        return response
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


class ChatRequest(BaseModel):
    animal_id: str
    message: str = Field(..., max_length=500)


@app.post("/chat")
def chat_with_record(body: ChatRequest):
    """Feature (i): ask anything about ONE animal, answered only from
    its record + care advice. Hindi in -> Hindi out."""
    animal = db.animals.find_one({"_id": body.animal_id})
    if animal is None:
        raise HTTPException(status_code=404, detail="animal not found in BPA records")
    return chat.answer(db, animal, body.message)


@app.post("/chat/voice")
def chat_with_voice(animal_id: str = Form(...),
                    audio: UploadFile = File(...)):
    """Spoken question in, grounded answer out - plus a spoken reply
    WAV when a matching Windows voice exists. Same grounding rules as
    /chat. Deliberately a sync endpoint: transcription + TTS take
    seconds, and FastAPI runs sync handlers in a threadpool so other
    requests keep flowing."""
    animal = db.animals.find_one({"_id": animal_id})
    if animal is None:
        raise HTTPException(status_code=404, detail="animal not found in BPA records")

    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    stem = uuid.uuid4().hex
    in_path = VOICE_DIR / f"{stem}.in.wav"
    data = audio.file.read(2 * 1024 * 1024 + 1)
    if len(data) > 2 * 1024 * 1024:
        raise HTTPException(status_code=413,
                            detail="recording too large - keep questions "
                                   "under ~1 minute")
    in_path.write_bytes(data)

    try:
        heard, _ = voice.transcribe(str(in_path))
    except voice.EngineUnavailable as e:
        raise HTTPException(status_code=503,
                            detail=f"speech engine unavailable: {e}")
    finally:
        in_path.unlink(missing_ok=True)  # farmer's recording: don't keep it
    if not heard:
        raise HTTPException(status_code=422,
                            detail="could not understand the audio - please "
                                   "record again closer to the phone")
    if len(heard) > 500:
        heard = heard[:500]

    response = chat.answer(db, animal, heard)
    response["heard"] = heard

    reply_path = VOICE_DIR / f"{stem}.reply.wav"
    if voice.synthesize(response["answer"], response["language"], str(reply_path)):
        response["audio_url"] = f"/voice-audio/{stem}.reply.wav"
    else:
        response["audio_url"] = None
    return response


@app.get("/voice-audio/{fname}")
def get_voice_audio(fname: str):
    if not re.fullmatch(r"[a-f0-9]+\.reply\.wav", fname):
        raise HTTPException(status_code=404, detail="unknown audio")
    path = VOICE_DIR / fname
    if not path.exists():
        raise HTTPException(status_code=404, detail="unknown audio")
    return FileResponse(path, media_type="audio/wav")


@app.get("/alerts")
def get_alerts():
    """The vet officer's mock notification feed: escalated screenings,
    newest first. In production this would push to the officer's BPA
    dashboard - in the demo it's this endpoint + a screen in the app."""
    return {"alerts": list(db.vet_alerts.find({}, {"_id": 0})
                           .sort("date", -1).limit(20))}


@app.get("/animal/{animal_id}/history")
def get_history(animal_id: str):
    if db.animals.find_one({"_id": animal_id}) is None:
        raise HTTPException(status_code=404, detail="animal not found in BPA records")

    sessions = []
    for s in db.sessions.find({"animal_id": animal_id}).sort([("date", -1), ("_id", -1)]):
        sessions.append({
            "session_id": s["session_id"],
            "date": s["date"],
            "weight_kg_mid": s.get("weight_kg_mid"),
            "health_flags": s.get("health_flags", []),
        })
    return {"animal_id": animal_id, "sessions": sessions}
