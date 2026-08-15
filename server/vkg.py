"""Veterinary Knowledge Graph - deterministic risk estimation.

Walks the symptom->condition graph in vkg.json. No ML here on
purpose: the same symptom vector ALWAYS produces the same ranked
risks, and every risk lists exactly which symptoms drove it. That
traceability IS the explainability of the screening layer.
"""
import json
from datetime import date, timedelta
from pathlib import Path

_VKG = json.loads((Path(__file__).parent / "vkg.json").read_text(encoding="utf-8"))

CONDITIONS = _VKG["conditions"]
KNOWN_SYMPTOMS = sorted({s for c in CONDITIONS.values() for s in c["symptoms"]})

OUTBREAK_MIN_ANIMALS = 3


def estimate_risks(symptom_vector: list[dict]) -> list[dict]:
    """symptom_vector: [{symptom, confidence, region, source}, ...]
    Returns ranked risks, strongest first. Unknown symptom names are
    ignored (forward compatibility while Person 2 iterates)."""
    fired = {s["symptom"]: s["confidence"] for s in symptom_vector}

    risks = []
    for cid, cond in CONDITIONS.items():
        matched = {s: w for s, w in cond["symptoms"].items() if s in fired}
        if not matched:
            continue
        evidence = sum(w * fired[s] for s, w in matched.items())
        coverage = round(evidence / sum(cond["symptoms"].values()), 2)
        level = "high" if coverage >= 0.55 else "medium" if coverage >= 0.30 else "low"

        if cond["urgency"] == "high" and level != "low":
            action = "refer to vet urgently"
        elif level in ("high", "medium"):
            action = "refer to vet"
        else:
            action = "monitor and recheck in a few days"

        risks.append({
            "condition": cid,
            "label": cond["label"],
            "urgency": cond["urgency"],
            "risk": level,
            "score": coverage,
            "because_of": sorted(matched, key=lambda s: -matched[s]),
            "action": action,
        })

    return sorted(risks, key=lambda r: -r["score"])


def needs_escalation(risks: list[dict]) -> bool:
    return any(r["risk"] == "high" or
               (r["urgency"] == "high" and r["risk"] != "low") for r in risks)


def herd_symptom_count(db, village: str, symptom: str, days: int = 14) -> int:
    """How many DISTINCT animals in this village showed this symptom
    in recent sessions. >= OUTBREAK_MIN_ANIMALS (incl. the current
    one) reads as an outbreak signal worth escalating."""
    village_ids = [a["_id"] for a in db.animals.find({"village": village}, {"_id": 1})]
    since = (date.today() - timedelta(days=days)).isoformat()
    return len(db.sessions.distinct("animal_id", {
        "animal_id": {"$in": village_ids},
        "date": {"$gte": since},
        "result.symptom_vector.symptom": symptom,
    }))
