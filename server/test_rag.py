"""Tests for the knowledge-RAG layer. Needs MongoDB running.
Passes with or without Ollama (keyword fallback covers the rest).

Run:  venv\\Scripts\\python test_rag.py
"""
import chat
import rag
from fastapi.testclient import TestClient
from main import app, db

client = TestClient(app)

ELIGIBLE = "356279812345"


def main():
    stats = rag.ensure_index(db)
    assert stats["chunks"] >= 15, stats
    print(f"PASS  knowledge index: {stats['chunks']} chunks "
          f"({stats['embedded']} newly embedded this run)")

    hits = rag.search(db, "what does rump angle mean", k=2)
    assert hits and "Rump Angle" in hits[0]["title"], hits[:1]
    print(f"PASS  semantic search top hit: {hits[0]['title']} "
          f"({hits[0]['method']}, score {hits[0]['score']})")

    # keyword fallback when the embedding endpoint is dead
    real = rag.OLLAMA_URL
    rag.OLLAMA_URL = "http://127.0.0.1:9"
    try:
        hits = rag.search(db, "rump angle meaning", k=2)
        assert hits and "Rump Angle" in hits[0]["title"], hits[:1]
        assert hits[0]["method"] == "keyword"
        print("PASS  keyword fallback finds the same chunk with Ollama dead")
    finally:
        rag.OLLAMA_URL = real

    # deterministic knowledge answer through /chat (template path);
    # stub the cloud chain too in case this machine has real keys
    real_chat = chat.OLLAMA_URL
    real_cloud = chat.llm_providers.try_cloud
    chat.OLLAMA_URL = "http://127.0.0.1:9"
    chat.llm_providers.try_cloud = lambda s, u: (None, None)
    try:
        r = client.post("/chat", json={"animal_id": ELIGIBLE,
                                       "message": "what does rump angle mean?"})
        body = r.json()
        assert "hook" in body["answer"].lower() or "pin" in body["answer"].lower() \
            or "slope" in body["answer"].lower(), body["answer"]
        assert body["sources"], "sources should cite the corpus"
        print(f"PASS  /chat knowledge answer w/ sources: {body['sources']}")

        # animal questions are NOT hijacked by the knowledge corpus
        r2 = client.post("/chat", json={"animal_id": ELIGIBLE,
                                        "message": "what is her weight?"})
        b2 = r2.json()
        # The point here is that the corpus does NOT hijack a question about
        # this animal - not that a number comes back. A session produced by the
        # demonstration engine correctly answers with a placeholder disclosure
        # rather than quoting an invented weight, and that is still a record
        # answer rather than a corpus one.
        assert "session" in b2["answer"], b2["answer"]
        assert b2["sources"] == [], "record answer must not cite the corpus"
        assert "hook" not in b2["answer"].lower(), (
            "the knowledge corpus answered a question about the animal")
        print("PASS  record questions still answered from the record, no hijack")
    finally:
        chat.OLLAMA_URL = real_chat
        chat.llm_providers.try_cloud = real_cloud


if __name__ == "__main__":
    main()
