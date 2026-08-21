import hashlib
import os
import re
import time
from collections import defaultdict, deque
from datetime import date, datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, PyMongoError

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

# cheap insurance against quota-burn/CPU loops from strangers on the
# network: per-IP sliding-window limits on the expensive endpoints.
# Generous enough that no honest demo ever hits them.
# /session is generous on purpose: flushing an offline queue of a dozen
# captured animals in one go is a DEMO FEATURE and must never trip it.
# The real disk protection is the 411/413 body caps, not this.
_RATE_LIMITS = {"/chat": (20, 60.0), "/chat/voice": (6, 60.0),
                "/transcribe": (10, 60.0), "/session": (40, 60.0)}
_rate_lock = threading.Lock()
_rate_hits: dict = defaultdict(deque)

# refuse oversized bodies BEFORE they get buffered (browsers always send
# Content-Length for form uploads); the per-file caps inside /session
# still cover anything that slips past
_BODY_CAPS = {"/session": 50 * 1024 * 1024,
              "/chat/voice": 8 * 1024 * 1024,
              "/transcribe": 8 * 1024 * 1024}

# session ids become folder names on disk AND appear in /overlays and
# /session URLs - one charset, enforced on write and read alike
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


@app.middleware("http")
async def _rate_limit(request, call_next):
    cap = _BODY_CAPS.get(request.url.path)
    if cap and request.method == "POST":
        cl = request.headers.get("content-length", "")
        if not cl.isdigit():
            # no/odd Content-Length = chunked streaming, which would be
            # spooled to disk unbounded BEFORE the handler runs. Every
            # real browser/app form post sends Content-Length.
            return JSONResponse(status_code=411, content={
                "detail": "length required - streamed uploads are not "
                          "accepted"})
        if int(cl) > cap:
            return JSONResponse(status_code=413, content={
                "detail": f"upload too large - max {cap // (1024 * 1024)} "
                          "MB per request"})
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


# CORS is registered AFTER the gate above so it ends up OUTERMOST in the
# stack: without this, the 411/413/429 short-circuit responses would
# carry no Access-Control-Allow-Origin and a browser client would see an
# opaque network error instead of the reason.
# hackathon LAN: allow browser-based clients too (Flutter web builds,
# quick HTML test pages) - native apps ignore CORS entirely
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

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
                "keep_alive": "4h",   # stay resident for the whole demo
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


@app.get("/demo")
def demo_console():
    """Judge-facing scoring console: run sessions, browse scorecards
    with overlay proof, weight trends, and the vet officer's alert
    feed - a complete demo without the mobile app."""
    return FileResponse(Path(__file__).parent / "static" / "demo.html",
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
    side_photo: UploadFile = File(...),
    rear_photo: UploadFile = File(...),
    animal_id: str = Form(None),
    device_session_id: str = Form(None),
    gait_video: UploadFile = File(None),
    # --- compatibility with the field names the app actually sends -------
    # The Flutter client posts tag_id and names the video 'video'. Replaying
    # its real request returned 422 on animal_id and device_session_id, so
    # the app could not upload a session at all; and because gait_video is
    # optional, a video named 'video' was silently DROPPED - a 200 with the
    # farmer's recording quietly discarded, which is worse than a refusal.
    #
    # tag_id and animal_id are the same thing: the 12-digit ear-tag number
    # IS the _id in the BPA records, so this is an alias, not a guess.
    tag_id: str = Form(None),
    video: UploadFile = File(None),
    # The close-up of the ear tag. Optional, and absent from the app today -
    # ScanTagScreen captures the tag NUMBER, not a photograph of it.
    #
    # It is accepted here because everything measured in centimetres depends
    # on it. In the side photograph the tag is a thumbnail and the detector
    # frequently does not find it at all; in a close-up the tag fills the
    # frame, no detector is needed, and the printed 18 mm digit row gives a
    # scale directly. ml/tag_intelligence/panel_transfer.py then carries that
    # scale to the side photograph using the tag as a bridge.
    #
    # Without it: the class-C traits, heart girth and the weight all refuse,
    # honestly, and the angle traits still work. With it they become
    # measurable. Both names are accepted so the app can use either.
    tag_photo: UploadFile = File(None),
    tag_image: UploadFile = File(None),
):
    # sync on purpose: FastAPI runs sync handlers in a threadpool, so
    # when the real ML pipeline (seconds of CPU) replaces the fake
    # engine, /ping and /chat keep answering while a session scores
    animal_id = animal_id or tag_id
    gait_video = gait_video if gait_video is not None else video
    if not animal_id:
        raise HTTPException(
            status_code=422,
            detail="animal_id (or tag_id) is required")

    # The client may not send a session id. Derive one from the CONTENT
    # instead of generating a random one, because this id is the offline
    # queue's idempotency key: a retry of the same upload must collapse to
    # the same session, and a random id would turn every retry into a
    # duplicate record. Same animal + same bytes = same id; retake a photo
    # and the bytes change, so it correctly becomes a new session.
    #
    # THE EAR-TAG CLOSE-UP HAS TO BE IN THIS HASH. It was left out, and it is
    # the one photograph the entire centimetre scale is derived from - so
    # swapping it changed nothing: the server recognised the side, rear and
    # video, called it a duplicate, and returned the scorecard computed from
    # the OLD tag in 0.2 seconds without looking at the new one.
    #
    # In the field that is the worst possible case. A farmer whose session
    # refused every centimetre trait for want of a readable tag walks back to
    # the animal, takes a better close-up, resubmits - and gets the same
    # refusal handed straight back, with no way to tell that nothing was
    # re-measured. It was also quietly invalidating our own A/B tests of tag
    # images, which is how it was found.
    if not device_session_id:
        h = hashlib.sha256()
        h.update(str(animal_id).encode())
        tag_for_hash = tag_photo if tag_photo is not None else tag_image
        for up in (side_photo, rear_photo, gait_video, tag_for_hash):
            if up is None:
                continue
            up.file.seek(0)
            for chunk in iter(lambda: up.file.read(1 << 20), b""):
                h.update(chunk)
            up.file.seek(0)
        device_session_id = f"auto-{h.hexdigest()[:24]}"

    # this id becomes a FOLDER NAME and a URL path segment. Unvalidated,
    # a timestamp id ("...T10:30:00") is an illegal Windows path (500 on
    # every retry) and "../.." would escape the uploads folder entirely.
    if not _SESSION_ID_RE.fullmatch(device_session_id):
        raise HTTPException(
            status_code=422,
            detail="device_session_id must be 1-64 characters of "
                   "A-Z a-z 0-9 - _ (no dots, spaces, colons or slashes)")
    try:
        animal = db.animals.find_one({"_id": animal_id})
    except PyMongoError:
        # a JSON 503 with the actual fix, not a text/plain 500 the
        # console would mislabel as a connection problem
        raise HTTPException(status_code=503,
                            detail="database unavailable - restart MongoDB "
                                   "(run_server.bat starts it)")
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

    # size-check EVERYTHING before writing ANYTHING, so a refused
    # upload never leaves a half-written orphan dir behind
    tag_close_up = tag_photo if tag_photo is not None else tag_image
    caps = {"side.jpg": 10 * 1024 * 1024, "rear.jpg": 10 * 1024 * 1024,
            "gait.mp4": 25 * 1024 * 1024, "tag.jpg": 10 * 1024 * 1024}
    blobs = {}
    for filename, upload in [("side.jpg", side_photo), ("rear.jpg", rear_photo),
                             ("gait.mp4", gait_video), ("tag.jpg", tag_close_up)]:
        if upload is not None:
            data = upload.file.read(caps[filename] + 1)
            if len(data) > caps[filename]:
                raise HTTPException(
                    status_code=413,
                    detail=f"{filename.split('.')[0]} file too large - max "
                           f"{caps[filename] // (1024 * 1024)} MB. Use a "
                           "smaller photo / record a shorter clip (~8 s).")
            blobs[filename] = data
    session_dir = UPLOAD_DIR / device_session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    for filename, data in blobs.items():
        (session_dir / filename).write_bytes(data)
        saved[filename] = str(session_dir / filename)

    result = score_animal(saved.get("side.jpg"), saved.get("rear.jpg"),
                          saved.get("gait.mp4"), animal,
                          tag_img=saved.get("tag.jpg"))
    result["session_id"] = device_session_id
    result["eligible"] = eligible
    result["eligible_reason"] = reason
    # the server owns the BPA record and the session metadata - fill
    # anything the engine left blank, so swapping engines can never hand
    # the app a null in a field its models declare non-nullable
    if not result.get("breed_registered"):
        result["breed_registered"] = animal.get("breed")
    if result.get("animal_id") is None:
        result["animal_id"] = animal_id
    # captured_at is server-injected: the pipeline honestly does not know
    # wall-clock time, so it leaves this None and the server fills it.
    if not result.get("captured_at"):
        result["captured_at"] = datetime.now().astimezone().isoformat(
            timespec="seconds")
    if result.get("synced") is None:
        result["synced"] = True

    # BREED VERIFICATION - the one place the two branches genuinely
    # disagreed, so the reasoning is recorded here rather than resolved
    # silently.
    #
    # ml-dev left breed_verified as None: setting it to anything else is
    # fabricating a value. The measurements agree. Exact-breed verification
    # on the data we can legally use scores 38.1% source-held-out, and its
    # confidence carries no information - tightening the threshold from
    # 100% to 30% coverage moves accuracy only +5.6 points. The trained
    # model disables its own breed head for exactly this reason.
    #
    # server-dev coerced None to False, because the app's models declare
    # this field non-nullable and a null is a hard decode failure on the
    # phone. That constraint is real and the server cannot fix it alone.
    #
    # Resolution until Person 1 can accept a null: keep the wire value a
    # bool so the app cannot crash, and carry the honest state alongside it
    # in breed_verify_status, which the app may ignore safely.
    #
    # False here means NOT VERIFIED. It does not mean "contradicted". The
    # app must not render it as a breed mismatch - that would accuse a
    # correctly registered animal on every single record.
    if result.get("breed_verified") is None:
        result["breed_verified"] = False
        result["breed_verify_confidence"] = 0.0
        result.setdefault("breed_verify_status", "unverified")
    else:
        result.setdefault(
            "breed_verify_status",
            "agree" if result["breed_verified"] else "disagree")

    # pixel coordinates are integers on the wire: the app parses them as
    # ints and a float would be a hard decode failure on the phone
    for t in result.get("traits", []):
        pts = t.get("overlay_points")
        if isinstance(pts, list):
            t["overlay_points"] = [
                [int(round(p[0])), int(round(p[1]))] for p in pts
                if isinstance(p, (list, tuple)) and len(p) == 2]

    # screening layer: deterministic risk estimation from the symptom
    # vector (Person 2's detectors fill it; the VKG reasons over it)
    risks = vkg.estimate_risks(result["symptom_vector"])
    result["risk_report"] = risks

    herd_alerts = []
    for s in result["symptom_vector"]:
        others = vkg.herd_symptom_count(db, animal["village"], s["symptom"],
                                        exclude_animal=animal_id)
        if others + 1 >= vkg.OUTBREAK_MIN_ANIMALS:
            herd_alerts.append({"symptom": s["symptom"],
                                "village": animal["village"],
                                "animals_affected_14d": others + 1})
    result["herd_alerts"] = herd_alerts
    result["reports"] = reports.build_reports(
        animal, risks, result["symptom_vector"], herd_alerts,
        screened=bool(result.get("vet_screened", False)))
    result["escalated"] = vkg.needs_escalation(risks) or bool(herd_alerts)
    if result["escalated"]:
        # upsert by session_id: a retried upload refreshes its own alert
        # instead of stacking duplicates in the officer's feed; store the
        # human label, not the internal condition id
        db.vet_alerts.replace_one({"session_id": device_session_id}, {
            "session_id": device_session_id,
            "animal_id": animal_id,
            "village": animal["village"],
            "date": date.today().isoformat(),
            # Which engine produced the findings behind this alert. The
            # baseline engine INVENTS symptoms - skin_nodules at confidence
            # 0.82, or gait_asymmetry - and those flow through the knowledge
            # graph into needs_escalation and land here, in a veterinary
            # officer's feed, about an animal nothing examined. Someone could
            # drive to a farm over it.
            #
            # The alert is still raised, because the escalation path is a real
            # feature that has to be demonstrable, but it is labelled so that
            # nobody acts on a placeholder. Suppressing it instead would hide
            # the feature; labelling it keeps the demo honest.
            "demonstration": not str(result.get("engine", "")).startswith("ml"),
            "top_risks": [r.get("label") or r["condition"]
                          for r in risks[:3]],
            "herd_alerts": list(herd_alerts),
            "report_vet": result["reports"]["vet"],
        }, upsert=True)

    weight = result.get("weight_kg")
    # low/high are honestly None when weight could not be measured (Heart
    # Girth needs a 3D model that does not exist yet). Do not fabricate a
    # midpoint, and do not crash on None + None.
    weight_mid = None
    if isinstance(weight, dict) and             isinstance(weight.get("low"), (int, float)) and             isinstance(weight.get("high"), (int, float)):
        weight_mid = (weight["low"] + weight["high"]) // 2
    try:
        db.sessions.insert_one({
            "session_id": device_session_id,
            "animal_id": animal_id,
            "date": date.today().isoformat(),
            "weight_kg_mid": weight_mid,
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
    if not _SESSION_ID_RE.fullmatch(session_id):
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
    # render to a private temp file, then swap it in atomically: two
    # judges tapping the same trait at once must never see a half-written
    # JPEG (cached.exists() turns true the moment PIL opens for writing)
    cached.parent.mkdir(parents=True, exist_ok=True)
    tmp = cached.with_name(f"{cached.stem}.{uuid.uuid4().hex[:8]}.tmp")
    render_overlay(source, trait, tmp)
    os.replace(tmp, cached)          # atomic on the same volume
    return FileResponse(cached, media_type="image/jpeg")


class ChatRequest(BaseModel):
    animal_id: str
    message: str = Field(..., max_length=500)
    language: str = "auto"   # auto | en | hi | kn - forces reply language
    speak: bool = False      # true -> also synthesize a spoken reply


@app.post("/chat")
def chat_with_record(body: ChatRequest):
    """Feature (i): ask anything about ONE animal, answered only from
    its record + care advice, in the requested language."""
    animal = db.animals.find_one({"_id": body.animal_id})
    if animal is None:
        raise HTTPException(status_code=404, detail="animal not found in BPA records")
    override = body.language if body.language in ("en", "hi", "kn") else None
    response = chat.answer(db, animal, body.message, lang_override=override)
    response["audio_url"] = None
    if body.speak:
        VOICE_DIR.mkdir(parents=True, exist_ok=True)
        base = str(VOICE_DIR / f"{uuid.uuid4().hex}.reply")
        written = voice.synthesize(response["answer"], response["language"], base)
        if written:
            response["audio_url"] = f"/voice-audio/{Path(written).name}"
    return response


@app.post("/transcribe")
def transcribe_only(audio: UploadFile = File(...),
                    language: str = Form("auto")):
    """Speech-to-text WITHOUT answering: the UI puts the text in the
    chat box so the farmer can review/correct it before sending."""
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    in_path = VOICE_DIR / f"{uuid.uuid4().hex}.in"
    data = audio.file.read(2 * 1024 * 1024 + 1)
    if len(data) > 2 * 1024 * 1024:
        raise HTTPException(status_code=413,
                            detail="recording too large - keep questions "
                                   "under ~1 minute")
    in_path.write_bytes(data)
    try:
        heard, lang = voice.transcribe(str(in_path), lang_hint=language)
    except voice.EngineUnavailable as e:
        raise HTTPException(status_code=503,
                            detail=f"speech engine unavailable: {e}")
    finally:
        in_path.unlink(missing_ok=True)
    if not heard:
        raise HTTPException(status_code=422,
                            detail="could not hear any words - please record "
                                   "again closer to the phone")
    return {"heard": heard[:500], "language": lang}


@app.post("/chat/voice")
def chat_with_voice(animal_id: str = Form(...),
                    audio: UploadFile = File(...),
                    language: str = Form("auto")):
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
        heard, _ = voice.transcribe(str(in_path), lang_hint=language)
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

    override = language if language in ("en", "hi", "kn") else None
    response = chat.answer(db, animal, heard, lang_override=override)
    response["heard"] = heard

    written = voice.synthesize(response["answer"], response["language"],
                               str(VOICE_DIR / f"{stem}.reply"))
    response["audio_url"] = f"/voice-audio/{Path(written).name}" if written else None
    return response


@app.get("/voice-audio/{fname}")
def get_voice_audio(fname: str):
    if not re.fullmatch(r"[a-f0-9]+\.reply\.(wav|mp3)", fname):
        raise HTTPException(status_code=404, detail="unknown audio")
    path = VOICE_DIR / fname
    if not path.exists():
        raise HTTPException(status_code=404, detail="unknown audio")
    return FileResponse(path, media_type="audio/mpeg" if fname.endswith(".mp3")
                        else "audio/wav")


@app.get("/alerts")
def get_alerts():
    """The vet officer's mock notification feed: escalated screenings,
    newest first. In production this would push to the officer's BPA
    dashboard - in the demo it's this endpoint + a screen in the app."""
    return {"alerts": list(db.vet_alerts.find({}, {"_id": 0})
                           .sort([("date", -1), ("_id", -1)]).limit(20))}


@app.get("/session/{session_id}")
def get_session(session_id: str):
    """Full stored scorecard for one session - powers the demo
    dashboard and any 'open old session' screen in the app."""
    s = db.sessions.find_one({"session_id": session_id}, {"_id": 0,
                                                          "files": 0})
    if s is None:
        raise HTTPException(status_code=404, detail="unknown session")
    return s


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
            # WITHOUT THIS THE HISTORY SCREEN DRAWS INVENTED WEIGHTS AS A TREND.
            #
            # The scorecard, the chat, the reports and the alert feed all
            # disclose the baseline engine. This endpoint did not send the
            # field at all, so the one screen that plots weight over time had
            # no way to know, and rendered three placeholder figures - 392,
            # 405, 418 kg from random.Random(animal_id) - as a confident
            # rising line with a green figure beside each row.
            #
            # Measured on animal 356279812345: the assistant refuses that same
            # weight outright ("did not produce a real weight measurement")
            # while History shows it climbing. Same animal, same app, opposite
            # claims - and the trend is the more persuasive of the two,
            # because a line going up looks like evidence.
            "measured": str(s.get("result", {}).get("engine", "")).startswith("ml"),
        })
    return {"animal_id": animal_id, "sessions": sessions}
