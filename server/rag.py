"""Local RAG over OUR OWN reference corpus (server/knowledge/*.md).

Strictly-our-data by construction: the corpus is authored in this repo,
embeddings are computed locally (Ollama nomic-embed-text), stored in
our MongoDB, and searched with plain cosine similarity. No internet,
no external knowledge base.

Degrades gracefully: if the embedding model is unavailable, search
falls back to keyword overlap - worse ranking, same corpus, still
strictly our data.
"""
import hashlib
import math
import os
import re
import threading
from pathlib import Path

import httpx

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
OLLAMA_URL = "http://127.0.0.1:11434"
EMBED_MODEL = os.environ.get("SIH_EMBED_MODEL", "nomic-embed-text")

_index_lock = threading.Lock()
_indexed = False


def _load_chunks() -> list[dict]:
    """Every '## heading' section in every knowledge file is one chunk."""
    chunks = []
    for md in sorted(KNOWLEDGE_DIR.glob("*.md")):
        text = md.read_text(encoding="utf-8")
        for m in re.finditer(r"^## +(.+?)\n(.*?)(?=^## |\Z)", text,
                             re.MULTILINE | re.DOTALL):
            title, body = m.group(1).strip(), m.group(2).strip()
            if not body:
                continue
            digest = hashlib.sha256(f"{title}\n{body}".encode()).hexdigest()[:16]
            chunks.append({"_id": digest, "source": md.name,
                           "title": title, "text": body})
    return chunks


def _embed(text: str) -> list[float] | None:
    try:
        r = httpx.post(f"{OLLAMA_URL}/api/embeddings",
                       timeout=httpx.Timeout(20.0, connect=3.0),
                       json={"model": EMBED_MODEL, "prompt": text})
        r.raise_for_status()
        emb = r.json().get("embedding")
        return emb if emb else None
    except Exception:
        return None


def ensure_index(db) -> dict:
    """Idempotent: (re)index only new/changed chunks. Chunks whose
    embedding failed are stored anyway - keyword fallback covers them."""
    global _indexed
    with _index_lock:
        chunks = _load_chunks()
        current_ids = {c["_id"] for c in chunks}
        db.knowledge.delete_many({"_id": {"$nin": list(current_ids)}})
        existing = {d["_id"]: d for d in
                    db.knowledge.find({}, {"_id": 1, "embedding": 1})}
        added = embedded = 0
        for c in chunks:
            old = existing.get(c["_id"])
            if old is not None and old.get("embedding"):
                continue
            emb = _embed(f"{c['title']}\n{c['text']}")
            db.knowledge.replace_one(
                {"_id": c["_id"]}, {**c, "embedding": emb}, upsert=True)
            added += 1
            embedded += 1 if emb else 0
        _indexed = True
        return {"chunks": len(chunks), "updated": added, "embedded": embedded}


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


_WORD = re.compile(r"[a-z]{3,}")


def _keyword_score(query: str, doc: dict) -> float:
    qwords = set(_WORD.findall(query.lower()))
    if not qwords:
        return 0.0
    hay = f"{doc['title']} {doc['text']}".lower()
    title = doc["title"].lower()
    hits = sum(1 for w in qwords if w in hay)
    title_hits = sum(1 for w in qwords if w in title)
    return (hits + 2 * title_hits) / (3 * len(qwords))


def search(db, query: str, k: int = 2) -> list[dict]:
    """Top-k chunks as {title, text, source, score, method}."""
    if not _indexed:
        ensure_index(db)
    docs = list(db.knowledge.find({}))
    if not docs:
        return []

    qemb = _embed(query)
    results = []
    if qemb is not None:
        for d in docs:
            if d.get("embedding"):
                results.append({**d, "score": _cosine(qemb, d["embedding"]),
                                "method": "embedding"})
    if not results:  # no query embedding or no embedded docs
        for d in docs:
            results.append({**d, "score": _keyword_score(query, d),
                            "method": "keyword"})

    results.sort(key=lambda d: -d["score"])
    return [{"title": d["title"], "text": d["text"], "source": d["source"],
             "score": round(d["score"], 3), "method": d["method"]}
            for d in results[:k]]


def is_strong(hit: dict) -> bool:
    return hit["score"] >= (0.45 if hit["method"] == "embedding" else 0.15)
