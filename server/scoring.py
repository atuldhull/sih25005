"""Fake scoring engine.

score_animal() returns the exact contract/scoring_result.json shape
with plausible fake values. Person 2's real ML pipeline replaces the
BODY of this function later - same inputs, same output shape - and
nothing else in the server (or the app) has to change.

Fake values are seeded from the animal id, so the same animal always
gets the same scorecard (looks consistent in demos), while different
animals get different ones.
"""
import random
from datetime import datetime

# name, category, measure_class, view, (low, high, unit) or None if
# the trait is subjective (measured relative, no single number shown)
TRAITS = [
    ("Stature",               "Dairy Strength", "C",    "side",  (125, 145, "cm")),
    ("Heart Girth",           "Dairy Strength", "SMAL", "side",  (170, 195, "cm")),
    ("Body Length",           "Dairy Strength", "C",    "side",  (140, 160, "cm")),
    ("Body Depth",            "Dairy Strength", "C",    "side",  (70, 85, "cm")),
    ("Angularity",            "Dairy Strength", "A",    "side",  (15, 30, "deg")),
    ("Rump Angle",            "Rump",           "A",    "side",  (0, 8, "deg slope")),
    ("Rump Width",            "Rump",           "C",    "rear",  (15, 22, "cm")),
    ("Rear Legs Set",         "Feet & Legs",    "A",    "side",  (140, 165, "deg")),
    ("Rear Legs Rear View",   "Feet & Legs",    "B",    "video", None),
    ("Foot Angle",            "Feet & Legs",    "A",    "side",  (40, 55, "deg")),
    ("Fore Udder Attachment", "Udder",          "B",    "side",  None),
    ("Rear Udder Height",     "Udder",          "C",    "rear",  (18, 28, "cm")),
    ("Central Ligament",      "Udder",          "B",    "rear",  None),
    ("Udder Depth",           "Udder",          "C",    "rear",  (2, 10, "cm above hock")),
    ("Front Teat Placement",  "Udder",          "B",    "rear",  None),
    ("Teat Length",           "Udder",          "C",    "rear",  (4, 7, "cm")),
    ("Rear Teat Placement",   "Udder",          "B",    "rear",  None),
    ("Rear Udder Width",      "Udder",          "C",    "rear",  (10, 17, "cm")),
    ("Teat Thickness",        "Udder",          "C",    "rear",  (2, 4, "cm")),
    ("Body Condition Score",  "General",        "SMAL", "side",  None),
]

CONFIDENCE_BY_CLASS = {"A": (0.80, 0.92), "B": (0.60, 0.75),
                       "C": (0.72, 0.88), "SMAL": (0.65, 0.80)}


def score_animal(side_photo_path, rear_photo_path, gait_video_path, animal_record):
    rng = random.Random(animal_record["_id"])

    traits = []
    abstain_index = rng.randrange(len(TRAITS)) if rng.random() < 0.25 else -1
    for i, (name, category, mclass, view, meas) in enumerate(TRAITS):
        points = [[rng.randint(200, 800), rng.randint(100, 650)]
                  for _ in range(rng.randint(2, 3))]

        if i == abstain_index:
            traits.append({
                "name": name, "category": category, "score": None,
                "confidence": round(rng.uniform(0.30, 0.49), 2),
                "measured_value": None, "ci": None,
                "measure_class": mclass, "view": view, "overlay_points": [],
                "explanation": None,
                "not_scored_reason": f"{name} not clearly visible - confidence "
                                     "too low. Retake the photo to score this trait.",
            })
            continue

        lo, hi = CONFIDENCE_BY_CLASS[mclass]
        measured = ci = None
        if meas:
            vlo, vhi, unit = meas
            val = round(rng.uniform(vlo, vhi), 1)
            measured = f"{val} {unit}"
            if mclass in ("C", "SMAL"):
                spread = round((vhi - vlo) * 0.08, 1)
                ci = f"{round(val - spread, 1)}-{round(val + spread, 1)} {unit}"

        traits.append({
            "name": name, "category": category,
            "score": rng.randint(3, 8),
            "confidence": round(rng.uniform(lo, hi), 2),
            "measured_value": measured, "ci": ci,
            "measure_class": mclass, "view": view,
            "overlay_points": points,
            "explanation": f"{name} assessed from the {view} capture.",
        })

    weight_mid = rng.randint(360, 430)
    # deterministic symptoms on an explicit allowlist so NO un-scripted
    # animal can fire a surprise escalation if a judge asks us to score
    # a random one. 346 = seeded outbreak member, 351 = the live
    # on-stage trigger, 347 = the story's lameness buffalo (needs video)
    tag = str(animal_record["_id"])
    nodules = tag in ("356279812346", "356279812351")
    limping = tag == "356279812347" and gait_video_path is not None

    return {
        "session_id": None,  # filled by the server
        "animal_id": animal_record["_id"],
        "species": animal_record["species"],
        "breed_registered": animal_record["breed"],
        "breed_verified": True,
        "breed_verify_confidence": round(rng.uniform(0.85, 0.95), 2),
        "eligible": True,           # server re-fills from its own check
        "eligible_reason": None,    # server re-fills
        "captured": {
            "side_photo": side_photo_path is not None,
            "rear_photo": rear_photo_path is not None,
            "gait_video": gait_video_path is not None,
        },
        "traits": traits,
        "weight_kg": {"low": weight_mid - 14, "high": weight_mid + 14,
                      "method": "girth-length-regression",
                      "cross_check": f"smal-volume: {weight_mid + rng.randint(-8, 8)}"},
        "symptom_vector": (
            [{"symptom": "skin_nodules", "confidence": 0.82,
              "region": "skin", "source": "photo"}] if nodules else
            [{"symptom": "gait_asymmetry",
              "confidence": round(rng.uniform(0.6, 0.85), 2),
              "region": "legs", "source": "video"}] if limping else []),
        "risk_report": [],  # the server's knowledge graph fills this
        "health_flags": (["visible_abnormality"] if nodules else
                         ["locomotion_abnormal"] if limping else []),
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "synced": True,
    }
