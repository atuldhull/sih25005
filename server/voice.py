"""Voice layer for the farmer chatbot.

STT chain: Gemini via the same rotating free-tier keys (excellent
Hindi/Kannada, native script out) -> local faster-whisper (offline
fallback; weaker on Indian languages, which is exactly why the UI now
shows the transcription for review before sending).

TTS chain: Microsoft Edge neural voices (free, no key, natural Indian
voices for en/hi/kn - needs internet) -> Windows SAPI (offline,
robotic but functional) -> text-only. synthesize() returns the actual
file written (.mp3 for neural, .wav for SAPI) or None.

All incoming audio (phones record webm/opus) is normalized to 16 kHz
mono WAV locally with PyAV - bundled with faster-whisper, no ffmpeg
install needed.
"""
import os
import re
import threading
from pathlib import Path

# team convention: the HF model cache lives on D: so every account /
# restored laptop finds the pre-downloaded whisper weights without the
# user-scoped HF_HOME env var having to exist (preflight.py mirrors this)
if "HF_HOME" not in os.environ and Path(r"D:\hf-cache").exists():
    os.environ["HF_HOME"] = r"D:\hf-cache"

# "small" is noticeably better than "base" at Hindi/Kannada and still
# runs fine on CPU (~250 MB one-time download into the HF cache on D:)
STT_MODEL = os.environ.get("SIH_STT_MODEL", "small")

# tests flip these off for deterministic offline behavior
USE_CLOUD_STT = True
USE_EDGE_TTS = True

EDGE_VOICES = {"en": "en-IN-NeerjaNeural",
               "hi": "hi-IN-SwaraNeural",
               "kn": "kn-IN-SapnaNeural"}

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_KANNADA = re.compile(r"[ಀ-೿]")

_whisper = None
_whisper_lock = threading.Lock()
# SAPI is not thread-safe; serialize TTS calls
_tts_lock = threading.Lock()


class EngineUnavailable(Exception):
    """STT engine could not load (missing model/download failure) -
    distinct from 'this particular clip was not understandable'."""


def _script_lang(text: str) -> str:
    if _KANNADA.search(text):
        return "kn"
    if _DEVANAGARI.search(text):
        return "hi"
    return "en"


def _get_whisper():
    global _whisper
    with _whisper_lock:
        if _whisper is None:
            from faster_whisper import WhisperModel
            _whisper = WhisperModel(STT_MODEL, device="cpu", compute_type="int8")
    return _whisper


def normalize_to_wav(src_path: str, dst_path: str) -> bool:
    """Any container/codec in (webm/opus, mp3, wav...) -> 16 kHz mono
    s16 WAV. True on success."""
    try:
        import av
        from av.audio.resampler import AudioResampler
        with av.open(src_path) as in_c:
            in_stream = in_c.streams.audio[0]
            resampler = AudioResampler(format="s16", layout="mono", rate=16000)
            with av.open(dst_path, "w", format="wav") as out_c:
                out_stream = out_c.add_stream("pcm_s16le", rate=16000,
                                              layout="mono")
                for frame in in_c.decode(in_stream):
                    for rf in resampler.resample(frame):
                        for pkt in out_stream.encode(rf):
                            out_c.mux(pkt)
                for pkt in out_stream.encode(None):
                    out_c.mux(pkt)
        return Path(dst_path).stat().st_size > 44
    except Exception:
        return False


def transcribe(audio_path: str, lang_hint: str | None = None) -> tuple[str, str]:
    """Returns (text, detected_language: en/hi/kn). lang_hint (from the
    UI's language selector) steers both engines - it stops whisper from
    mis-detecting Hindi as English and transliterating garbage. Empty
    text means THIS clip was not understandable; a broken engine raises
    EngineUnavailable so the server can answer 503."""
    hint = lang_hint if lang_hint in ("en", "hi", "kn") else None
    wav_path = audio_path + ".norm.wav"
    have_wav = normalize_to_wav(audio_path, wav_path)
    try:
        if USE_CLOUD_STT and have_wav:
            import llm_providers
            text = llm_providers.transcribe_cloud(
                Path(wav_path).read_bytes(), mime="audio/wav", lang_hint=hint)
            if text:
                return text, _script_lang(text)

        try:
            model = _get_whisper()
        except Exception as e:
            raise EngineUnavailable(str(e)[:200])
        try:
            segments, info = model.transcribe(
                wav_path if have_wav else audio_path,
                beam_size=1, vad_filter=True,
                language=hint, task="transcribe")
            text = " ".join(s.text.strip() for s in segments).strip()
            lang = _script_lang(text) if text else (info.language or "en")
            return text, lang
        except Exception:
            return "", "en"
    finally:
        Path(wav_path).unlink(missing_ok=True)


def _edge_tts(text: str, lang: str, out_path: str) -> bool:
    try:
        import asyncio

        import edge_tts

        async def run():
            comm = edge_tts.Communicate(text, EDGE_VOICES.get(lang,
                                                              EDGE_VOICES["en"]))
            await asyncio.wait_for(comm.save(out_path), timeout=10.0)

        asyncio.run(run())
        return os.path.exists(out_path) and os.path.getsize(out_path) > 0
    except Exception:
        return False


def _sapi_tts(text: str, lang: str, out_path: str) -> bool:
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
            if lang in ("hi", "kn"):
                match = [v for v in engine.getProperty("voices")
                         if lang == "hi" and ("hindi" in (v.name or "").lower()
                                              or "kalpana" in (v.name or "").lower())]
                if not match:
                    engine.stop()
                    return False  # no local voice for this language - honest
                engine.setProperty("voice", match[0].id)
            engine.setProperty("rate", 165)
            engine.save_to_file(text, out_path)
            engine.runAndWait()
            engine.stop()
            return os.path.exists(out_path) and os.path.getsize(out_path) > 0
        except Exception:
            return False


def synthesize(text: str, lang: str, out_base: str) -> str | None:
    """Write a spoken reply. out_base is a path WITHOUT extension.
    Returns the actual file path written (.mp3 neural / .wav SAPI),
    or None when no voice is available - the app then shows text only."""
    if USE_EDGE_TTS and _edge_tts(text, lang, out_base + ".mp3"):
        return out_base + ".mp3"
    if _sapi_tts(text, lang, out_base + ".wav"):
        return out_base + ".wav"
    return None
