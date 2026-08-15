"""Cloud + free-tier LLM provider chain for the chatbot.

Order: Gemini (rotating free-tier keys - configure 2-3, when one hits
its per-minute quota the next takes over) -> any OpenAI-compatible
free providers (Groq, OpenRouter, ...) -> caller falls back to local
Ollama, then deterministic templates. The chain never raises: every
failure rotates forward, logging one operator-readable line (key
identified by last 4 characters ONLY - full keys never leave
keys.json).

KEYS ARE NEVER COMMITTED. They live in server/keys.json (gitignored)
or the SIH_GEMINI_KEYS env var (comma-separated). keys.json is
hot-reloaded on change - paste keys while the server runs, no restart.
"""
import json
import os
import sys
import threading
import time
from pathlib import Path

import httpx

TIMEOUT = httpx.Timeout(12.0, connect=3.0)
_KEYS_FILE = Path(__file__).parent / "keys.json"

QUOTA_COOLDOWN = 65.0      # free tiers meter per-minute: sit out one minute
BAD_KEY_COOLDOWN = 3600.0  # invalid/revoked key: stop wasting calls on it


def _log(msg: str):
    print(f"[llm] {msg}", file=sys.stderr)


class KeyPool:
    """Rotates keys; a key on cooldown is skipped until its time expires."""

    def __init__(self, keys: list[str]):
        self._lock = threading.Lock()
        self._cooldown_until = {k: 0.0 for k in keys if k and "PASTE" not in k}

    def usable_keys(self) -> list[str]:
        now = time.monotonic()
        with self._lock:
            return [k for k, t in self._cooldown_until.items() if t <= now]

    def cooldown(self, key: str, seconds: float):
        with self._lock:
            if key in self._cooldown_until:
                self._cooldown_until[key] = time.monotonic() + seconds

    def size(self) -> int:
        return len(self._cooldown_until)


_lock = threading.Lock()
_state = {"mtime": None, "config": {}, "gemini": KeyPool([]),
          "compat": {}, "file_status": "missing"}


def _tail(key: str) -> str:
    return "..." + key[-4:] if len(key) >= 4 else "****"


def _rebuild(config: dict):
    _state["config"] = config
    _state["gemini"] = KeyPool(config.get("gemini", {}).get("keys", []))
    compat = {}
    for p in list(config.get("openai_compatible", [])):
        if not all(p.get(k) for k in ("name", "base_url", "model")):
            _log(f"WARNING: openai_compatible entry {p.get('name', '?')!r} is "
                 "missing name/base_url/model - entry skipped")
            config["openai_compatible"].remove(p)
            continue
        compat[p["name"]] = KeyPool(p.get("keys", []))
    _state["compat"] = compat


def _reload_if_changed():
    """keys.json is tiny - re-stat it each call so pasted keys take
    effect within seconds, without a server restart."""
    with _lock:
        mtime = None
        if _KEYS_FILE.exists():
            try:
                mtime = _KEYS_FILE.stat().st_mtime
            except OSError:
                pass
        if mtime == _state["mtime"] and _state["config"]:
            return
        cfg = {}
        status = "missing"
        if mtime is not None:
            try:
                cfg = json.loads(_KEYS_FILE.read_text(encoding="utf-8"))
                status = "loaded"
            except Exception as e:
                _log(f"WARNING: {_KEYS_FILE.name} exists but is not valid "
                     f"JSON ({e}) - cloud chain disabled until fixed")
                status = "invalid"
        env_keys = [k.strip() for k in
                    os.environ.get("SIH_GEMINI_KEYS", "").split(",") if k.strip()]
        if env_keys:
            gem = cfg.setdefault("gemini", {})
            gem["keys"] = list(dict.fromkeys(env_keys + gem.get("keys", [])))
        changed = mtime != _state["mtime"]
        _state["mtime"] = mtime
        _state["file_status"] = status
        _rebuild(cfg)
        if changed and status == "loaded":
            _log(f"keys.json loaded: {_state['gemini'].size()} gemini key(s), "
                 f"{len(_state['compat'])} compat provider(s)")


def _gemini_model() -> str:
    return os.environ.get("SIH_GEMINI_MODEL",
                          _state["config"].get("gemini", {})
                          .get("model", "gemini-2.5-flash"))


class QuotaExhausted(Exception):
    pass


class BadKey(Exception):
    pass


def _gemini_once(key: str, system: str, user: str) -> str | None:
    # key travels in a HEADER, never in the URL - exception messages and
    # logs that include the URL can therefore never contain the key
    r = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{_gemini_model()}:generateContent",
        headers={"x-goog-api-key": key}, timeout=TIMEOUT,
        json={
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 1000,
                # 2.5-class models spend the budget on hidden "thinking"
                # and can return EMPTY text at low limits - disable it,
                # a 100-word grounded answer needs no deliberation
                "thinkingConfig": {"thinkingBudget": 0},
            },
        })
    if r.status_code == 429:
        raise QuotaExhausted()
    if r.status_code in (400, 401, 403):
        raise BadKey()
    r.raise_for_status()
    cand = (r.json().get("candidates") or [{}])[0]
    parts = ((cand.get("content") or {}).get("parts") or [])
    text = " ".join(p.get("text", "") for p in parts).strip()
    if not text and cand.get("finishReason") == "MAX_TOKENS":
        _log(f"gemini returned empty text with finishReason=MAX_TOKENS "
             f"(model {_gemini_model()}) - check thinking budget/model")
    return text or None


def _try_gemini(system: str, user: str) -> tuple[str, str] | None:
    pool = _state["gemini"]
    for key in pool.usable_keys():
        try:
            text = _gemini_once(key, system, user)
            if text:
                return text, f"gemini:{_gemini_model()}"
        except QuotaExhausted:
            _log(f"gemini key {_tail(key)} hit quota (429) - "
                 f"cooling {int(QUOTA_COOLDOWN)}s, rotating")
            pool.cooldown(key, QUOTA_COOLDOWN)
        except BadKey:
            _log(f"gemini key {_tail(key)} rejected (400/401/403) - "
                 "cooling 1h; check the key in keys.json")
            pool.cooldown(key, BAD_KEY_COOLDOWN)
        except Exception as e:
            _log(f"gemini key {_tail(key)}: {type(e).__name__} - cooling "
                 f"{int(QUOTA_COOLDOWN)}s")
            pool.cooldown(key, QUOTA_COOLDOWN)
    return None


def _try_compat(system: str, user: str) -> tuple[str, str] | None:
    for p in _state["config"].get("openai_compatible", []):
        pool = _state["compat"].get(p["name"])
        if pool is None:
            continue
        for key in pool.usable_keys():
            try:
                r = httpx.post(
                    f"{p['base_url'].rstrip('/')}/chat/completions",
                    timeout=TIMEOUT,
                    headers={"Authorization": f"Bearer {key}"},
                    json={"model": p["model"], "temperature": 0.3,
                          "max_tokens": 400,
                          "messages": [{"role": "system", "content": system},
                                       {"role": "user", "content": user}]})
                if r.status_code == 429:
                    _log(f"{p['name']} key {_tail(key)} hit quota - rotating")
                    pool.cooldown(key, QUOTA_COOLDOWN)
                    continue
                if r.status_code in (400, 401, 403):
                    _log(f"{p['name']} key {_tail(key)} rejected - cooling 1h")
                    pool.cooldown(key, BAD_KEY_COOLDOWN)
                    continue
                r.raise_for_status()
                text = (r.json()["choices"][0]["message"]["content"] or "").strip()
                if text:
                    return text, f"{p['name']}:{p['model']}"
            except Exception as e:
                _log(f"{p['name']} key {_tail(key)}: {type(e).__name__} - "
                     "rotating")
                pool.cooldown(key, QUOTA_COOLDOWN)
    return None


def try_cloud(system: str, user: str) -> tuple[str | None, str | None]:
    """Best cloud answer available, or (None, None). Never raises."""
    _reload_if_changed()
    for attempt in (_try_gemini, _try_compat):
        try:
            got = attempt(system, user)
            if got:
                return got
        except Exception:
            pass
    return None, None


def status() -> dict:
    _reload_if_changed()
    return {
        "keys_file": _state["file_status"],
        "gemini_keys": _state["gemini"].size(),
        "gemini_keys_usable_now": len(_state["gemini"].usable_keys()),
        "gemini_model": _gemini_model(),
        "compat_providers": [p["name"] for p in
                             _state["config"].get("openai_compatible", [])],
    }
