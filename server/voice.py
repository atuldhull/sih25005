"""Voice layer for the farmer chatbot - free and fully local.

STT: faster-whisper (CPU int8). First call downloads the model into
the HF cache (HF_HOME, on D:) and keeps it loaded. Auto-detects the
spoken language, so Hindi speech comes out as Devanagari text and the
chat layer answers in Hindi automatically.

TTS: best-effort via Windows SAPI (pyttsx3). If a matching voice
exists it returns a WAV; if not, the app simply shows text - the
voice reply is a bonus, never a dependency.
"""
import os
import threading

STT_MODEL = os.environ.get("SIH_STT_MODEL", "base")

_whisper = None
_whisper_lock = threading.Lock()
# SAPI is not thread-safe; serialize TTS calls
_tts_lock = threading.Lock()


class EngineUnavailable(Exception):
    """STT engine could not load (missing model/download failure) -
    distinct from 'this particular clip was not understandable'."""


def _get_whisper():
    global _whisper
    with _whisper_lock:
        if _whisper is None:
            from faster_whisper import WhisperModel
            _whisper = WhisperModel(STT_MODEL, device="cpu", compute_type="int8")
    return _whisper


def transcribe(audio_path: str) -> tuple[str, str]:
    """Returns (text, detected_language_2letter). Empty text means THIS
    clip was not understandable; a broken engine raises EngineUnavailable
    so the server can answer 503 instead of a misleading 'record again'."""
    try:
        model = _get_whisper()
    except Exception as e:
        raise EngineUnavailable(str(e)[:200])
    try:
        segments, info = model.transcribe(audio_path, beam_size=1, vad_filter=True)
        text = " ".join(s.text.strip() for s in segments).strip()
        return text, (info.language or "en")
    except Exception:
        return "", "en"


def synthesize(text: str, lang: str, out_path: str) -> bool:
    """Write a WAV reply. True on success, False if no usable voice."""
    with _tts_lock:
        try:
            # SAPI runs over COM, which must be initialized per-thread -
            # FastAPI sync endpoints run in a threadpool, so do it here
            # (double-init is harmless, ole32 just returns S_FALSE)
            import ctypes
            try:
                ctypes.windll.ole32.CoInitialize(None)
            except Exception:
                pass
            import pyttsx3
            engine = pyttsx3.init()
            if lang == "hi":
                hindi = [v for v in engine.getProperty("voices")
                         if "hindi" in (v.name or "").lower()
                         or "hi-in" in (v.id or "").lower()
                         or "kalpana" in (v.name or "").lower()]
                if hindi:
                    engine.setProperty("voice", hindi[0].id)
                else:
                    engine.stop()
                    return False  # no Hindi voice - text-only is honest
            engine.setProperty("rate", 165)
            engine.save_to_file(text, out_path)
            engine.runAndWait()
            engine.stop()
            return os.path.exists(out_path) and os.path.getsize(out_path) > 0
        except Exception:
            return False
