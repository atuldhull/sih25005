"""Plain-language screening reports - one for the farmer, one for
the vet. Template-based and deterministic: the text only REPHRASES
what the knowledge graph concluded, it never adds medical content.

Optional polish: set env var SIH_OLLAMA_MODEL (e.g. 'llama3.2') and
a local Ollama will reword the farmer report more naturally. If
Ollama is absent or slow, the template text is used unchanged - the
demo never depends on an LLM being up.
"""
import os

import httpx

DISCLAIMER = ("This is a screening result from photos and video, not a "
              "diagnosis. A veterinarian makes the final call.")


def _short(animal_id: str) -> str:
    return "..." + animal_id[-4:]


def build_reports(animal: dict, risks: list[dict], symptom_vector: list[dict],
                  herd_alerts: list[dict], screened: bool = False) -> dict:
    """screened says whether a veterinary screening actually RAN.

    It matters more than it looks. This build has no trained symptom detector,
    so the screening never runs and symptom_vector is always empty - and an
    empty list was being reported to the farmer as "No health problems were
    flagged" and to the vet officer as "Detected signs: none". Both read as a
    clinical finding. A farmer might skip a visit on the strength of it.

    Not screened is not the same as clear, and the difference belongs in the
    words, not in a field nobody reads.
    """
    # Detected symptoms are themselves proof that a screening ran, so the flag
    # is a floor rather than an override. Without this a caller that reports
    # real findings but forgets the flag would have them announced under a
    # heading saying nothing was examined.
    screened = bool(screened) or bool(symptom_vector) or bool(risks)

    farmer = [f"Health check for your {animal['breed']} ({_short(animal['_id'])}):"]
    if not screened:
        farmer.append("This check measured your animal's body only. It did "
                      "NOT check for illness - no health screening was done "
                      "today, so this says nothing about whether your animal "
                      "is well. If she seems unwell, please see your "
                      "veterinary officer.")
    elif not risks:
        farmer.append("No health problems were flagged from today's photos "
                      "and video. Keep up the regular care.")
    else:
        for r in risks[:3]:
            farmer.append(f"- {r['label']}: {r['risk']} risk. "
                          f"{_advice_farmer(r['condition'])} ({r['action']})")
    for h in herd_alerts:
        farmer.append(f"- Note: {h['animals_affected_14d']} animals in "
                      f"{h['village']} showed similar signs recently. "
                      "Please inform your veterinary officer.")
    farmer.append(DISCLAIMER)

    vet = [f"AI pre-screening summary - {animal['_id']} "
           f"({animal['breed']} {animal['species']}, village {animal['village']}):"]
    if symptom_vector:
        signs = ", ".join(f"{s['symptom']} (conf {s['confidence']:.2f}, "
                          f"{s['source']})" for s in symptom_vector)
        vet.append(f"Detected signs: {signs}.")
    elif screened:
        vet.append("Detected signs: none.")
    else:
        vet.append("NOT SCREENED - no symptom detector ran on this session. "
                   "This is not a negative finding; nothing was examined.")
    for i, r in enumerate(risks, 1):
        vet.append(f"{i}. {r['label']} - score {r['score']:.2f} ({r['risk']} risk, "
                   f"urgency {r['urgency']}), driven by: "
                   f"{', '.join(r['because_of'])}. {_advice_vet(r['condition'])}")
    for h in herd_alerts:
        vet.append(f"Herd signal: {h['animals_affected_14d']} distinct animals in "
                   f"{h['village']} with '{h['symptom']}' in the last 14 days.")
    vet.append(DISCLAIMER)

    return {"farmer": _maybe_polish("\n".join(farmer)), "vet": "\n".join(vet)}


def _advice_farmer(condition_id: str) -> str:
    from vkg import CONDITIONS
    return CONDITIONS[condition_id]["advice_farmer"]


def _advice_vet(condition_id: str) -> str:
    from vkg import CONDITIONS
    return CONDITIONS[condition_id]["advice_vet"]


def _maybe_polish(text: str) -> str:
    model = os.environ.get("SIH_OLLAMA_MODEL")
    if not model:
        return text
    try:
        r = httpx.post("http://127.0.0.1:11434/api/generate", timeout=10.0, json={
            "model": model, "stream": False,
            "prompt": ("Reword this livestock health note in simple, warm "
                       "language a farmer understands. Keep EVERY fact, number "
                       "and the final disclaimer. Do not add any new medical "
                       "information.\n\n" + text),
        })
        r.raise_for_status()
        polished = r.json().get("response", "").strip()
        return polished if polished else text
    except Exception:
        return text
