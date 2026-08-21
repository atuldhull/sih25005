"""Local RAG over OUR OWN reference corpus (server/knowledge/*.md + vkg.json).

Strictly-our-data by construction: the corpus is authored in this repo,
embeddings are computed locally (Ollama nomic-embed-text), stored in
our MongoDB, and searched with plain cosine similarity. No internet,
no external knowledge base.

Degrades gracefully: if the embedding model is unavailable, search
falls back to keyword overlap - worse ranking, same corpus, still
strictly our data.
"""
import hashlib
import json
import math
import os
import re
import threading
from pathlib import Path

import httpx

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
VKG_PATH = Path(__file__).parent / "vkg.json"
OLLAMA_URL = "http://127.0.0.1:11434"
EMBED_MODEL = os.environ.get("SIH_EMBED_MODEL", "nomic-embed-text")

_index_lock = threading.Lock()
_indexed = False


def _load_vkg_chunks() -> list[dict]:
    """The veterinary knowledge graph's farmer advice, made retrievable.

    server/vkg.json already carries authored care advice for twelve conditions
    - mastitis, lameness, LSD, FMD and the rest - in English and Hindi. Until
    now the only route to any of it was for the LAST scoring session to have
    flagged that exact condition, so a farmer who simply described what they
    could see got nothing.

    The measured consequence: asked "her udder is swollen and hard", the
    assistant steered to Lumpy Skin Disease, because LSD was that animal's
    flagged risk - while the correct mastitis advice sat in this file, in this
    repo, unreachable. Retrieval had nothing better to offer, because the .md
    corpus contains the word "mastitis" zero times.

    Nothing here is generated. This indexes text a person already wrote. It
    does not invent husbandry advice and must never be used to.
    """
    try:
        data = json.loads(VKG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []      # a missing or broken vkg.json costs retrieval, not the server

    chunks = []
    for key, cond in (data.get("conditions") or {}).items():
        advice = (cond.get("advice_farmer") or "").strip()
        if not advice:
            continue

        # The symptom names ride along in the text so that a farmer describing
        # what they can see ("swollen udder", "limping") matches the condition,
        # even though the advice paragraph itself never names the symptom.
        body = advice
        symptoms = ", ".join(sorted(cond.get("symptoms", {}))).replace("_", " ")
        if symptoms:
            body += f"\n\nSigns associated with this: {symptoms}."
        hindi = (cond.get("advice_farmer_hi") or "").strip()
        if hindi:
            body += f"\n\n{hindi}"

        title = cond.get("label") or key.replace("_", " ").title()
        digest = hashlib.sha256(f"vkg:{key}\n{body}".encode()).hexdigest()[:16]
        chunks.append({"_id": digest, "source": "vkg.json",
                       "title": title, "text": body})
    return chunks


def _load_chunks() -> list[dict]:
    """Every '## heading' section in every knowledge file is one chunk, plus
    one chunk per authored condition in the veterinary knowledge graph."""
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
    chunks.extend(_load_vkg_chunks())
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


# Measured, not chosen. 0.45 sat BELOW this embedder's noise floor: literal
# gibberish ("asdfgh qwerty zxcvb") scored 0.474, an income-tax question 0.541
# and a train-timetable question 0.516 - all "strong" - so essentially every
# question came back with citations, and the citations were decoration. A
# deworming answer was sourced to "capture-guide.md - How to take the rear
# photo", and a feeding answer to "eligibility.md - Why the window exists".
#
# Re-measured after vkg.json was indexed, 8 in-corpus queries against 6
# out-of-corpus ones:
#     relevant  0.617 - 0.818   (min: "she is very thin and not eating")
#     junk      0.404 - 0.516   (max: "what time is the train to Delhi?")
# 0.57 sits in that gap: all 8 relevant pass, none of the 6 junk do.
#
# Two honest caveats. The sample is 14 queries, so treat this as a floor that
# separates the cases we tested rather than a calibrated operating point - if
# a real question you would want answered scores below it, re-measure rather
# than nudging the number. And a score above the bar means the chunk is ON
# TOPIC, never that the answer built from it is correct.
EMBEDDING_STRONG = 0.57
KEYWORD_STRONG = 0.15


def is_strong(hit: dict) -> bool:
    if hit["method"] == "embedding":
        return hit["score"] >= EMBEDDING_STRONG
    return hit["score"] >= KEYWORD_STRONG
