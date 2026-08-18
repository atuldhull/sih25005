"""Stage-day GO / NO-GO checker. Run this BEFORE walking on stage:

  venv\\Scripts\\python preflight.py

Checks every dependency the demo needs and prints one verdict at the
end. FAIL = the demo will visibly break; WARN = a fallback will kick
in (voice/LLM quality drops but nothing dies). Works whether the
server is already running or not.
"""
import json
import os
import shutil
import socket
import sys
import urllib.request
from datetime import date, datetime
from pathlib import Path

from pymongo import MongoClient


def _http_json(url, timeout=2.5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8"))

HERE = Path(__file__).parent
STAR = "356279812345"
LIVE_TRIGGER = "356279812351"

_results = []  # (level, name, detail)


def check(level, name, detail=""):
    _results.append((level, name, detail))
    mark = {"PASS": "[OK]  ", "WARN": "[WARN]", "FAIL": "[FAIL]"}[level]
    print(f"{mark} {name}" + (f" - {detail}" if detail else ""))


def lan_ips():
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))          # no packet sent; picks a route
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if not ip.startswith("127."):
                ips.add(ip)
    except OSError:
        pass
    return sorted(ips)


def main():
    print("Pashu Mitra - stage preflight\n" + "=" * 46)

    # 1. MongoDB
    db = None
    try:
        client = MongoClient("mongodb://127.0.0.1:27017",
                             serverSelectionTimeoutMS=2500)
        client.admin.command("ping")
        db = client["sih25005"]
        check("PASS", "MongoDB reachable")
    except Exception:
        check("FAIL", "MongoDB down",
              "run_server.bat starts it, or: "
              r"D:\mongodb-windows-x86_64-8.3.2"
              r"\mongodb-win32-x86_64-windows-8.3.2\bin\mongod.exe "
              r"--dbpath D:\mongodb\data")

    # 2. API server (connect probe, NOT bind: on Windows a 127.0.0.1
    # bind can succeed even while uvicorn holds 0.0.0.0:8000)
    server_up = False
    try:
        status, _ = _http_json("http://127.0.0.1:8000/ping", timeout=6)
        server_up = status == 200
        check("PASS", "API server answering on :8000")
    except Exception:
        try:
            socket.create_connection(("127.0.0.1", 8000), timeout=1.5).close()
            check("FAIL", "something listens on :8000 but /ping fails",
                  "the server may be stuck mid-start (ml import?) or "
                  "another program holds the port - check the server window")
        except OSError:
            check("WARN", "API server not running",
                  "start it with run_server.bat (port 8000 is free)")

    # 3. Ollama + models. EXACT tag match - the runtime requests the
    # exact model name and a prefix variant would 404 at runtime
    chat_model = os.environ.get("SIH_CHAT_MODEL", "qwen2.5:7b")
    embed_model = os.environ.get("SIH_EMBED_MODEL", "nomic-embed-text")
    try:
        _, tags = _http_json("http://127.0.0.1:11434/api/tags")
        names = {m["name"] for m in tags.get("models", [])}
        flat = " ".join(sorted(names))
        for want, why in [(chat_model, "local LLM answers"),
                          (embed_model, "semantic RAG search")]:
            if want in names or f"{want}:latest" in names:
                check("PASS", f"ollama model {want}")
            else:
                check("WARN", f"ollama model {want} missing",
                      f"{why} falls back (have: {flat or 'none'}); "
                      f"ollama pull {want}")
        # a LISTED model can still be cold (unloaded after 5 min idle):
        # generate one token so the demo's first answer is not a stall
        try:
            import urllib.request as _ur
            req = _ur.Request(
                "http://127.0.0.1:11434/api/chat",
                data=json.dumps({"model": chat_model, "stream": False,
                                 "keep_alive": "4h",
                                 "options": {"num_predict": 1},
                                 "messages": [{"role": "user",
                                               "content": "hi"}]}
                                ).encode(),
                headers={"Content-Type": "application/json"})
            import time as _t
            t0 = _t.time()
            with _ur.urlopen(req, timeout=90) as r:
                r.read()
            check("PASS", f"ollama warm + resident for 4h "
                          f"(first token in {_t.time() - t0:.1f}s)")
        except Exception:
            check("WARN", "ollama listed but did not answer a "
                          "test generation", "chat falls back to "
                          "cloud keys or templates")
    except Exception:
        check("WARN", "ollama not running",
              "chat falls back to cloud keys or plain templates; "
              "RAG falls back to keyword search")

    # 4. Gemini keys - ask llm_providers itself (same parser the server
    # uses, incl. the SIH_GEMINI_KEYS env var), so this check can never
    # disagree with the runtime
    try:
        import llm_providers
        st = llm_providers.status()
        n = st.get("gemini_keys", 0)
        extra = st.get("compat_providers") or []
        if n:
            check("PASS", f"cloud chain: {n} Gemini key(s)" +
                          (f" + {', '.join(extra)}" if extra else ""))
        elif st.get("keys_file") == "missing":
            check("WARN", "keys.json missing",
                  "copy keys.json.example -> keys.json and paste the keys")
        else:
            check("WARN", "keys.json present but no usable Gemini keys",
                  "cloud chain disabled - local chain still works")
    except Exception as e:
        check("WARN", "could not read the key pool", str(e))

    # 5. Offline voice: resolve the cache EXACTLY like the runtime does
    # (voice.py defaults HF_HOME to D:\hf-cache when that folder exists;
    # then huggingface_hub: HF_HUB_CACHE > HUGGINGFACE_HUB_CACHE >
    # HF_HOME/hub > ~/.cache/huggingface/hub)
    stt_model = os.environ.get("SIH_STT_MODEL", "small")
    model_dir = f"models--Systran--faster-whisper-{stt_model}"
    if "HF_HOME" not in os.environ and Path(r"D:\hf-cache").exists():
        os.environ["HF_HOME"] = r"D:\hf-cache"      # same as voice.py
    if os.environ.get("HF_HUB_CACHE"):
        hub = Path(os.environ["HF_HUB_CACHE"])
    elif os.environ.get("HUGGINGFACE_HUB_CACHE"):
        hub = Path(os.environ["HUGGINGFACE_HUB_CACHE"])
    elif os.environ.get("HF_HOME"):
        hub = Path(os.environ["HF_HOME"]) / "hub"
    else:
        hub = Path.home() / ".cache" / "huggingface" / "hub"
    if (hub / model_dir).exists():
        check("PASS", f"faster-whisper '{stt_model}' cached at {hub} "
                      "(offline STT ready)")
    else:
        check("WARN", f"whisper '{stt_model}' not in the runtime cache",
              f"looked in {hub} - first voice use would try to download")

    # 6. Neural voice + cloud LLM need REAL internet - a captive portal
    # accepts TCP connects, so require the genuine 204 response
    try:
        with urllib.request.urlopen("https://www.gstatic.com/generate_204",
                                    timeout=4) as r:
            real_net = r.status == 204
        if real_net:
            check("PASS", "internet reachable (edge-tts + cloud chain)")
        else:
            check("WARN", "network answers but looks like a captive "
                          "portal", "open a browser and complete the "
                  "portal login; until then voice uses the offline "
                  "Windows voice and chat uses local ollama")
    except OSError:
        check("WARN", "no internet",
              "voice falls back to the offline Windows voice, chat to "
              "local ollama - the demo still works")

    # 7. Demo story + day-drift hazards
    if db is not None:
        n_animals = db.animals.count_documents({})
        if n_animals == 20:
            check("PASS", "animals collection seeded (20)")
        else:
            check("FAIL", f"animals collection has {n_animals}, expected 20",
                  "run: venv\\Scripts\\python demo_seed.py")

        star3 = db.sessions.find_one({"session_id": "demo-star-3"})
        if star3 is None:
            check("FAIL", "demo story not seeded",
                  "run: venv\\Scripts\\python demo_seed.py")
        else:
            check("PASS", "demo story present (star history found)")

            import rules
            import vkg
            for aid, role in [(STAR, "star (live re-scan)"),
                              (LIVE_TRIGGER, "live outbreak trigger")]:
                a = db.animals.find_one({"_id": aid})
                ok, reason = rules.check_eligibility(a) if a else (False,
                                                                   "missing")
                if ok:
                    check("PASS", f"{role} eligible TODAY", reason)
                else:
                    check("FAIL", f"{role} NOT eligible today: {reason}",
                          "re-run demo_seed.py (dates are relative to "
                          "seed day and have drifted)")

            n = vkg.herd_symptom_count(db, "Anand", "skin_nodules",
                                       exclude_animal=LIVE_TRIGGER)
            if n >= 2:
                check("PASS", f"outbreak cluster armed ({n} flagged; "
                              f"scoring the trigger makes {n + 1})")
            else:
                check("FAIL", f"outbreak cluster stale ({n} within 14 days)",
                      "re-run demo_seed.py - seeded sessions aged out "
                      "of the 14-day window")

            newest = star3.get("date", "")
            if newest and (date.today() -
                           datetime.strptime(newest, "%Y-%m-%d").date()
                           ).days > 10:
                check("WARN", "story was seeded a while ago",
                      "re-run demo_seed.py for fresh dates")

        alerts = db.vet_alerts.count_documents({})
        herd = db.vet_alerts.count_documents({"herd_alerts.0":
                                              {"$exists": True}})
        if alerts >= 4 and herd >= 1:
            check("PASS", f"officer feed ready ({alerts} alerts, "
                          f"{herd} with outbreak signal)")
        else:
            check("FAIL", f"officer feed incomplete ({alerts} alerts, "
                          f"{herd} outbreak)", "re-run demo_seed.py")

        stray = db.sessions.count_documents({
            "animal_id": STAR,
            "session_id": {"$not": {"$regex": "^demo-"}}})
        if stray:
            check("WARN", f"star has {stray} non-demo session(s)",
                  "test/rehearsal rows will show in the hero history "
                  "tab - re-run demo_seed.py to reset")

    # 8. Demo assets + overlay cache warm
    for name in ("side.jpg", "rear.jpg"):
        if (HERE / "demo_assets" / name).exists():
            check("PASS", f"demo photo bundled: {name}")
        else:
            check("FAIL", f"demo_assets/{name} missing",
                  "overlays will fall back to a grey placeholder")
    warm = len(list((HERE / "overlays_cache").glob("demo-*/*.jpg"))) \
        if (HERE / "overlays_cache").exists() else 0
    if warm >= 50:
        check("PASS", f"overlay cache pre-rendered ({warm} images)")
    else:
        check("WARN", f"overlay cache cold ({warm} images)",
              "re-run demo_seed.py to pre-render (first taps render live)")

    # 9. Scoring engine + LIVE-TRIGGER dress rehearsal. The loader
    # hot-swaps to ml/pipeline.py the moment it imports cleanly, and a
    # contract-valid ML engine is NOT guaranteed to detect the nodules
    # the outbreak moment depends on - so rehearse the exact call here
    # and FAIL loudly rather than find out on stage.
    try:
        sys.path.append(str(HERE.parent / "contract"))
        import vkg
        from scoring_loader import engine_status, score_animal
        st = engine_status()
        importable = st.get("real_pipeline_importable")
        check("PASS", "scoring engine: " +
              ("ml-pipeline importable" if importable else
               "baseline (ml/ not importable - expected until "
               "Person 2's pipeline lands)"))
        trig = db.animals.find_one({"_id": LIVE_TRIGGER}) \
            if db is not None else None
        if trig is not None:
            res = score_animal("demo_side.jpg", "demo_rear.jpg", None, trig)
            symptoms = {s.get("symptom")
                        for s in res.get("symptom_vector", [])}
            n = vkg.herd_symptom_count(db, trig["village"], "skin_nodules",
                                       exclude_animal=LIVE_TRIGGER)
            if "skin_nodules" in symptoms and \
                    n + 1 >= vkg.OUTBREAK_MIN_ANIMALS:
                check("PASS", "LIVE TRIGGER rehearsed",
                      f"engine '{res.get('engine')}' detects nodules; "
                      f"outbreak banner would show {n + 1} animals")
            else:
                check("FAIL", "LIVE TRIGGER would NOT fire on stage",
                      f"engine '{res.get('engine')}' returned "
                      f"{sorted(symptoms) or 'no symptoms'}, prior "
                      f"flagged animals={n}. If ml/ landed recently, "
                      "move it aside for the demo; else re-run "
                      "demo_seed.py")
            star = db.animals.find_one({"_id": STAR})
            if star is not None:
                sres = score_animal("demo_side.jpg", "demo_rear.jpg",
                                    None, star)
                if sres.get("symptom_vector"):
                    check("WARN", "star re-scan is no longer "
                          "symptom-free", str([s.get("symptom") for s
                                               in sres["symptom_vector"]]))
    except Exception as e:
        check("FAIL", "scoring loader broken", str(e))

    # 10. Disk space
    free_gb = shutil.disk_usage(str(HERE)).free / 1e9
    if free_gb > 2:
        check("PASS", f"disk space ok ({free_gb:.1f} GB free)")
    else:
        check("FAIL", f"low disk space ({free_gb:.1f} GB free)")

    # 11. Where the phone app should point
    ips = lan_ips()
    if ips:
        urls = "  ".join(f"http://{ip}:8000" for ip in ips)
        check("PASS", "server address for the app", urls)
        print("       (if the phone can't connect: allow python through "
              "the Windows firewall, same Wi-Fi)")
    else:
        check("WARN", "no LAN IP found", "connect the laptop to Wi-Fi/hotspot")

    # verdict
    fails = [r for r in _results if r[0] == "FAIL"]
    warns = [r for r in _results if r[0] == "WARN"]
    print("=" * 46)
    if fails:
        print(f"NO-GO: {len(fails)} blocking problem(s), "
              f"{len(warns)} warning(s). Fix the [FAIL] lines above.")
    else:
        print(f"GO: all critical checks passed ({len(warns)} warning(s), "
              "fallbacks cover them).")
        if server_up:
            print("Console: http://127.0.0.1:8000/demo   "
                  "Assistant: http://127.0.0.1:8000/chat-ui")
    print()
    print("RUNBOOK - the live moments:")
    print(f"  1. History/trend + chatbot: star Gir {STAR}")
    print(f"  2. LIVE OUTBREAK: score {LIVE_TRIGGER} (Mehsana buffalo, "
          "Anand) in the console")
    print("  3. Officer feed tab: outbreak cluster + lameness case")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
