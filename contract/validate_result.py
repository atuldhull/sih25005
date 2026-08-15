"""Contract compliance checker for scoring_result.json v1.

Anyone can run this against a result dict to get an exact list of
contract violations - no arguing about shapes on integration day.

  from validate_result import validate
  problems = validate(result_dict, mode="pipeline")   # or "full"

mode="pipeline": what ml/score_animal() must return. Server-injected
keys (session_id, risk_report, herd_alerts, reports, escalated,
captured_at, synced) are not required; eligible/eligible_reason are
optional because the server recomputes them.
mode="full": the complete POST /session response shape.

Run directly to self-check the contract example and the server's
fake scoring engine:  py contract/validate_result.py
"""

TRAITS_20 = {
    "Stature": "Dairy Strength", "Heart Girth": "Dairy Strength",
    "Body Length": "Dairy Strength", "Body Depth": "Dairy Strength",
    "Angularity": "Dairy Strength",
    "Rump Angle": "Rump", "Rump Width": "Rump",
    "Rear Legs Set": "Feet & Legs", "Rear Legs Rear View": "Feet & Legs",
    "Foot Angle": "Feet & Legs",
    "Fore Udder Attachment": "Udder", "Rear Udder Height": "Udder",
    "Central Ligament": "Udder", "Udder Depth": "Udder",
    "Front Teat Placement": "Udder", "Teat Length": "Udder",
    "Rear Teat Placement": "Udder", "Rear Udder Width": "Udder",
    "Teat Thickness": "Udder",
    "Body Condition Score": "General",
}

PIPELINE_KEYS = ["animal_id", "species", "breed_registered", "breed_verified",
                 "breed_verify_confidence", "captured", "traits", "weight_kg",
                 "symptom_vector", "health_flags"]
SERVER_KEYS = ["session_id", "eligible", "eligible_reason", "risk_report",
               "herd_alerts", "reports", "escalated", "captured_at", "synced"]
TRAIT_KEYS = ["name", "category", "score", "confidence", "measured_value",
              "ci", "measure_class", "view", "overlay_points", "explanation"]


def validate(result: dict, mode: str = "pipeline") -> list[str]:
    p = []
    required = PIPELINE_KEYS + (SERVER_KEYS if mode == "full" else [])
    for k in required:
        if k not in result:
            p.append(f"missing top-level key: '{k}'")

    traits = result.get("traits")
    if not isinstance(traits, list):
        p.append("'traits' must be a list")
        return p
    if len(traits) != 20:
        p.append(f"traits must have exactly 20 entries, got {len(traits)}")

    seen = {}
    for i, t in enumerate(traits):
        where = f"traits[{i}]"
        if not isinstance(t, dict):
            p.append(f"{where}: not an object")
            continue
        name = t.get("name")
        where = f"traits[{i}] ({name})"
        for k in TRAIT_KEYS:
            if k not in t:
                p.append(f"{where}: missing key '{k}'")
        if name not in TRAITS_20:
            p.append(f"{where}: '{name}' is not one of the 20 NDDB trait names")
        else:
            seen[name] = True
            if t.get("category") != TRAITS_20[name]:
                p.append(f"{where}: category must be '{TRAITS_20[name]}', "
                         f"got '{t.get('category')}'")
        score = t.get("score")
        if score is None:
            if "not_scored_reason" not in t:
                p.append(f"{where}: score is null but 'not_scored_reason' is "
                         "missing - the app MUST show why")
        elif not (isinstance(score, int) and 1 <= score <= 9):
            p.append(f"{where}: score must be int 1-9 or null, got {score!r}")
        conf = t.get("confidence")
        if not (isinstance(conf, (int, float)) and 0 <= conf <= 1):
            p.append(f"{where}: confidence must be 0-1, got {conf!r}")
        if t.get("measure_class") not in ("A", "B", "C", "SMAL"):
            p.append(f"{where}: measure_class must be A/B/C/SMAL, "
                     f"got {t.get('measure_class')!r}")
        if t.get("view") not in ("side", "rear", "video"):
            p.append(f"{where}: view must be side/rear/video, got {t.get('view')!r}")
        if not isinstance(t.get("overlay_points"), list):
            p.append(f"{where}: overlay_points must be a list")
    for name in TRAITS_20:
        if name not in seen:
            p.append(f"traits: missing NDDB trait '{name}'")

    w = result.get("weight_kg")
    if not isinstance(w, dict):
        p.append("'weight_kg' must be an object {low, high, method, ...}")
    else:
        for k in ("low", "high", "method"):
            if k not in w:
                p.append(f"weight_kg: missing '{k}'")
        if isinstance(w.get("low"), (int, float)) and \
           isinstance(w.get("high"), (int, float)) and w["low"] > w["high"]:
            p.append("weight_kg: low > high")

    cap = result.get("captured")
    if not isinstance(cap, dict) or \
       set(cap or {}) != {"side_photo", "rear_photo", "gait_video"}:
        p.append("'captured' must be {side_photo, rear_photo, gait_video} booleans")

    for i, s in enumerate(result.get("symptom_vector") or []):
        for k in ("symptom", "confidence", "region", "source"):
            if k not in s:
                p.append(f"symptom_vector[{i}]: missing '{k}'")

    return p


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    here = Path(__file__).parent
    example = json.loads((here / "scoring_result.json").read_text(encoding="utf-8"))
    probs = validate(example, mode="full")
    print(f"contract example (full mode): "
          f"{'OK' if not probs else f'{len(probs)} problems'}")
    for x in probs:
        print("  -", x)

    sys.path.insert(0, str(here.parent / "server"))
    try:
        from scoring import score_animal
        fake = score_animal("s.jpg", "r.jpg", "g.mp4",
                            {"_id": "356279812345", "species": "cattle",
                             "breed": "Gir"})
        probs = validate(fake, mode="pipeline")
        print(f"server fake engine (pipeline mode): "
              f"{'OK' if not probs else f'{len(probs)} problems'}")
        for x in probs:
            print("  -", x)
    except ImportError as e:
        print(f"server fake engine: skipped ({e})")
