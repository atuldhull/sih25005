"""End-to-end voice tests. Needs MongoDB running. Forces the OFFLINE
chains (SAPI TTS + local whisper) so results don't depend on internet
or pasted keys; the neural/cloud tiers are exercised manually.
First run downloads the whisper model (~75 MB) into the HF cache.

Run:  venv\\Scripts\\python test_voice.py
"""
from pathlib import Path

import chat
import voice
from fastapi.testclient import TestClient
from main import VOICE_DIR, app

client = TestClient(app)

ELIGIBLE = "356279812345"  # Gir, has a session with weight on record


def main():
    voice.USE_EDGE_TTS = False   # deterministic: local SAPI + whisper only
    voice.USE_CLOUD_STT = False
    Path(VOICE_DIR).mkdir(parents=True, exist_ok=True)

    qbase = str(Path(VOICE_DIR) / "test.question")
    written = voice.synthesize("What is the weight of my cow?", "en", qbase)
    if not written:
        print("SKIP  no Windows TTS voice available - cannot build a spoken "
              "question; voice path untested on this machine")
        return
    print(f"PASS  spoken question synthesized "
          f"({Path(written).stat().st_size} bytes {Path(written).suffix})")

    heard, lang = voice.transcribe(written)
    assert "weight" in heard.lower(), f"whisper heard: {heard!r}"
    print(f"PASS  offline whisper transcribed: {heard!r} (lang {lang})")

    with open(written, "rb") as f:
        r = client.post("/transcribe",
                        files={"audio": ("q.wav", f, "audio/wav")})
    assert r.status_code == 200, r.text
    assert "weight" in r.json()["heard"].lower()
    print(f"PASS  /transcribe fills the chat box: {r.json()['heard']!r}")

    real_url = chat.OLLAMA_URL
    real_cloud = chat.llm_providers.try_cloud
    chat.OLLAMA_URL = "http://127.0.0.1:9"
    chat.llm_providers.try_cloud = lambda s, u: (None, None)
    try:
        r2 = client.post("/chat", json={"animal_id": ELIGIBLE,
                                        "message": "what is her weight?",
                                        "speak": True})
        body = r2.json()
        assert r2.status_code == 200 and "kg" in body["answer"], body
        if body.get("audio_url"):
            r3 = client.get(body["audio_url"])
            assert r3.status_code == 200 and r3.content[:4] == b"RIFF"
            print(f"PASS  /chat speak=true -> spoken reply "
                  f"({len(r3.content)} bytes WAV)")
        else:
            print("WARN  audio_url null - no local voice for the reply "
                  "(text-only; acceptable offline)")

        with open(written, "rb") as f:
            r4 = client.post("/chat/voice",
                             data={"animal_id": ELIGIBLE},
                             files={"audio": ("q.wav", f, "audio/wav")})
        assert r4.status_code == 200 and "kg" in r4.json()["answer"], r4.text
        print(f"PASS  /chat/voice one-shot still works "
              f"(heard {r4.json()['heard']!r})")

        r5 = client.get("/voice-audio/..%2f..%2fmain.py")
        assert r5.status_code == 404
        print("PASS  voice-audio path traversal rejected")
    finally:
        chat.OLLAMA_URL = real_url
        chat.llm_providers.try_cloud = real_cloud
        voice.USE_EDGE_TTS = True
        voice.USE_CLOUD_STT = True


if __name__ == "__main__":
    main()
