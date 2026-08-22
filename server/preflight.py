"""Stage-day GO / NO-GO checker. Run this BEFORE walking on stage:

  venv\\Scripts\\python preflight.py

Checks every dependency the demo needs and prints one verdict at the
end. FAIL = the demo will visibly break; WARN = a fallback will kick
in (voice/LLM quality drops but nothing dies). Works whether the
server is already running or not.
"""
import importlib
import json
import os
import shutil
import socket
import sys
import urllib.request
from datetime import date, datetime
from pathlib import Path

from pymongo import MongoClient
from pymongo.errors import PyMongoError


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
    """The routed address FIRST - that is the one the phone can reach.
    Virtual switches (Hyper-V/WSL) and APIPA are filtered/deprioritised
    so nobody types a 172.x vEthernet address into the app."""
    routed, others = None, []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))          # no packet sent; picks a route
        routed = s.getsockname()[0]
        s.close()
    except OSError:
        pass
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if ip.startswith(("127.", "169.254.")) or ip == routed:
                continue
            others.append(ip)
    except OSError:
        pass
    return routed, sorted(others)


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
    chat_model = os.environ.get("SIH_CHAT_MODEL", "gemma2:9b")
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

    # 7. Demo story + day-drift hazards. Guarded: if mongod dies while
    # this runs (it can - the ollama warm check above may hold the
    # process for a while), the run must keep going and still print a
    # verdict instead of dying mid-report.
    try:
        if db is not None:
            n_animals = db.animals.count_documents({})
            if n_animals == 20:
                check("PASS", "animals collection seeded (20)")
            else:
                check("FAIL",
                      f"animals collection has {n_animals}, expected 20",
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
                    ok, reason = rules.check_eligibility(a) if a else \
                        (False, "missing")
                    if ok:
                        check("PASS", f"{role} eligible TODAY", reason)
                    else:
                        check("FAIL",
                              f"{role} NOT eligible today: {reason}",
                              "re-run demo_seed.py (dates are relative to "
                              "seed day and have drifted)")

                n = vkg.herd_symptom_count(db, "Anand", "skin_nodules",
                                           exclude_animal=LIVE_TRIGGER)
                if n >= 2:
                    check("PASS", f"outbreak cluster armed ({n} flagged; "
                                  f"scoring the trigger makes {n + 1})")
                else:
                    check("FAIL",
                          f"outbreak cluster stale ({n} within 14 days)",
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
    except PyMongoError as e:
        check("FAIL", "MongoDB went away mid-check", str(e))

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
        import time as _time

        import vkg
        from scoring_loader import RETRY_SECONDS, engine_status, score_animal

        # the loader imports ml/ on a BACKGROUND thread, so asking once
        # would report "not importable" for a pipeline that is seconds
        # from being adopted. If ml/pipeline.py exists on disk, wait for
        # the import to settle before judging anything.
        ml_present = (HERE.parent / "ml" / "pipeline.py").exists()
        st = engine_status()
        if ml_present and not st.get("real_pipeline_importable"):
            print("       ml/pipeline.py found - waiting for the import "
                  "to finish (up to 90s)...")
            deadline = _time.monotonic() + 90
            while _time.monotonic() < deadline:
                _time.sleep(2)
                st = engine_status()
                if st.get("real_pipeline_importable"):
                    break
                if _time.monotonic() > deadline - RETRY_SECONDS:
                    break
        importable = st.get("real_pipeline_importable")
        if ml_present and not importable:
            check("FAIL", "ml/pipeline.py is present but NOT importable",
                  "the server will silently keep the baseline engine - "
                  "run: venv\\Scripts\\python ..\\contract\\"
                  "check_pipeline.py to see the import error")
        else:
            check("PASS", "scoring engine: " +
                  ("ml-pipeline imported and will be used" if importable
                   else "baseline (no ml/ yet - expected until Person 2's "
                        "pipeline lands)"))

        # rehearse with the SAME real files the console uploads - passing
        # nonexistent paths would crash a real CV pipeline, the loader
        # would fall back, and this check would pass on the baseline
        side = str(HERE / "demo_assets" / "side.jpg")
        rear = str(HERE / "demo_assets" / "rear.jpg")
        trig = db.animals.find_one({"_id": LIVE_TRIGGER}) \
            if db is not None else None
        if trig is not None:
            res = score_animal(side, rear, None, trig)
            engine = res.get("engine")
            if importable and engine != "ml-pipeline":
                check("FAIL", "the ml pipeline was REJECTED on a real call",
                      f"it imported but this scoring returned "
                      f"engine='{engine}' - it crashed or violated the "
                      "contract. Run ..\\contract\\check_pipeline.py for "
                      "the exact reason")
            symptoms = {s.get("symptom")
                        for s in res.get("symptom_vector", []) or []}
            n = vkg.herd_symptom_count(db, trig["village"], "skin_nodules",
                                       exclude_animal=LIVE_TRIGGER)
            if "skin_nodules" in symptoms and \
                    n + 1 >= vkg.OUTBREAK_MIN_ANIMALS:
                check("PASS", "LIVE TRIGGER rehearsed",
                      f"engine '{engine}' detects nodules; outbreak "
                      f"banner would show {n + 1} animals")
            else:
                check("FAIL", "LIVE TRIGGER would NOT fire on stage",
                      f"engine '{engine}' returned "
                      f"{sorted(symptoms) or 'no symptoms'}, prior "
                      f"flagged animals={n}. If ml/ landed recently, "
                      "move it aside for the demo; else re-run "
                      "demo_seed.py")
            star = db.animals.find_one({"_id": STAR})
            if star is not None:
                sres = score_animal(side, rear, None, star)
                if sres.get("symptom_vector"):
                    check("WARN", "star re-scan is no longer "
                          "symptom-free", str([s.get("symptom") for s
                                               in sres["symptom_vector"]]))
    except PyMongoError as e:
        check("FAIL", "MongoDB went away during the engine check", str(e))
    except Exception as e:
        check("FAIL", "scoring loader broken", str(e))

    # 9b. ML library versions, but ONLY once ml/ is on disk. The loader
    # imports ml.pipeline into THIS process, so the pipeline's libraries
    # become the server's libraries - and server/requirements.txt does not
    # list them, which means a fresh venv can import the pipeline and then
    # fail inside it.
    #
    # The version pin is not housekeeping. RT-DETRv2 stores its per-layer
    # prediction heads tied; transformers 5.15.1 expects them untied and
    # fills the gap with RANDOM weights. The ear_tag head reads decoder
    # layer 1 - one of the layers it randomises. The result is not a crash:
    # it is a confident-looking 0.62 detection with a box outside the image,
    # and it changes on every process start. 5.0.0 loads them correctly.
    # That cost two days to find, so it gets asserted here.
    if (HERE.parent / "ml" / "pipeline.py").exists():
        for mod, pin in (("torch", None), ("timm", None),
                         ("transformers", "5.0.0")):
            try:
                m = importlib.import_module(mod)
                got = getattr(m, "__version__", "?")
                if pin and got != pin:
                    check("FAIL", f"{mod} {got} - MUST be {pin}",
                          "5.15.1 randomises RT-DETRv2's tied prediction "
                          "heads: ear_tag boxes look plausible and are "
                          f"garbage. pip install {mod}=={pin}")
                else:
                    check("PASS", f"{mod} {got}"
                          + (f" (pinned {pin})" if pin else ""))
            except ImportError:
                check("FAIL", f"{mod} not installed in the SERVER env",
                      "the loader imports ml.pipeline into this process, so "
                      "the pipeline's deps must be installed here too - they "
                      "are not in server/requirements.txt")
    else:
        check("PASS", "ML library check skipped (no ml/pipeline.py yet)",
              "once Person 2's pipeline lands this asserts "
              "transformers==5.0.0")

    # 10. Disk space
    free_gb = shutil.disk_usage(str(HERE)).free / 1e9
    if free_gb > 2:
        check("PASS", f"disk space ok ({free_gb:.1f} GB free)")
    else:
        check("FAIL", f"low disk space ({free_gb:.1f} GB free)")

    # 11. Where the phone app should point
    routed, others = lan_ips()
    if routed:
        check("PASS", "GIVE THE APP THIS ADDRESS", f"http://{routed}:8000")
        if others:
            print("       (other interfaces, usually virtual switches - "
                  "ignore unless the hotspot is one of them: " +
                  ", ".join(others) + ")")
        print("       (if the phone can't connect: allow python through "
              "the Windows firewall, and be on the SAME hotspot)")
    elif others:
        check("WARN", "no routed address - not on a network?",
              "candidates: " + ", ".join(f"http://{i}:8000" for i in others))
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
