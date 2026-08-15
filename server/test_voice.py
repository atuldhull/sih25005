"""End-to-end voice test: synthesize a spoken question with the local
Windows voice, feed it through /chat/voice, verify the transcription,
the grounded answer, and the reply audio. Needs MongoDB running.
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
    Path(VOICE_DIR).mkdir(parents=True, exist_ok=True)
    question_wav = Path(VOICE_DIR) / "test.question.wav"
    ok = voice.synthesize("What is the weight of my cow?", "en",
                          str(question_wav))
    if not ok:
        print("SKIP  no Windows TTS voice available - cannot build a spoken "
              "question; voice path untested on this machine")
        return
    print(f"PASS  spoken question synthesized ({question_wav.stat().st_size} bytes)")

    heard, lang = voice.transcribe(str(question_wav))
    assert "weight" in heard.lower(), f"whisper heard: {heard!r}"
    print(f"PASS  whisper transcribed it: {heard!r} (lang {lang})")

    # deterministic answers for the assertion: force the template path
    real_url = chat.OLLAMA_URL
    chat.OLLAMA_URL = "http://127.0.0.1:9"
    try:
        with question_wav.open("rb") as f:
            r = client.post("/chat/voice",
                            data={"animal_id": ELIGIBLE},
                            files={"audio": ("q.wav", f, "audio/wav")})
        assert r.status_code == 200, r.text
        body = r.json()
        assert "weight" in body["heard"].lower()
        assert "kg" in body["answer"]
        print(f"PASS  /chat/voice grounded answer: {body['answer'][:60]}...")

        if body["audio_url"]:
            r2 = client.get(body["audio_url"])
            assert r2.status_code == 200
            assert r2.content[:4] == b"RIFF", "not a WAV file"
            print(f"PASS  spoken reply served ({len(r2.content)} bytes WAV)")
        else:
            print("WARN  audio_url null - no TTS voice for the reply "
                  "(app shows text; acceptable)")

        r3 = client.get("/voice-audio/..%2f..%2fmain.py")
        assert r3.status_code == 404
        print("PASS  voice-audio path traversal rejected")
    finally:
        chat.OLLAMA_URL = real_url


if __name__ == "__main__":
    main()
