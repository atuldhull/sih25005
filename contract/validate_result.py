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

# Breed/group verification, added by ml/pose_features/embedding_extractor.py.
#
# OPTIONAL on purpose. The baseline engine does not produce them, and the app
# can ignore unknown keys safely, so adding them needed no contract break -
# which is why they are here rather than in PIPELINE_KEYS. Validated for TYPE
# when present, never for presence.
#
# Why they exist at all: exact-breed verification does not work on data we can
# legally use (38.1% source-held-out, and its confidence carries no
# information), so breed_verified stays null. The coarser GROUP call does work
# - 80.2% against a 60.7% background control - and group_consistent against
# the BPA record is the actually useful signal.
VERIFICATION_KEYS = {
    "predicted_species": (str,),
    "species_confidence": (int, float),
    "species_consistent": (bool,),
    "predicted_group": (str,),
    "group_confidence": (int, float),
    "group_consistent": (bool,),
    "group_reliable": (bool,),
    "breed_verify_note": (str,),
}
VALID_GROUPS = {"red_zebu", "grey_draught", "dwarf_cattle", "exotic_dairy",
                "buffalo"}


def validate(result: dict, mode: str = "pipeline") -> list[str]:
    p = []
    required = PIPELINE_KEYS + (SERVER_KEYS if mode == "full" else [])
    for k in required:
        if k not in result:
            p.append(f"missing top-level key: '{k}'")

    # verification fields: null is always valid (it means "did not / could
    # not assess"), but a present value must have the right type. A string
    # where the app expects a bool is a decode failure on the phone.
    for k, types in VERIFICATION_KEYS.items():
        if k in result and result[k] is not None:
            if not isinstance(result[k], types):
                p.append(f"{k}: must be "
                         f"{' or '.join(t.__name__ for t in types)} or null, "
                         f"got {type(result[k]).__name__}")
    g = result.get("predicted_group")
    if g is not None and g not in VALID_GROUPS:
        p.append(f"predicted_group: '{g}' is not one of {sorted(VALID_GROUPS)}")
    for k in ("species_confidence", "group_confidence"):
        v = result.get(k)
        if isinstance(v, (int, float)) and not 0.0 <= v <= 1.0:
            p.append(f"{k}: must be between 0 and 1, got {v}")
    # A verdict without a prediction is incoherent - it would tell the app to
    # flag a record while naming nothing to flag it against.
    if result.get("group_consistent") is not None and             result.get("predicted_group") is None:
        p.append("group_consistent is set but predicted_group is null - a "
                 "verdict with nothing behind it")

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
        pts = t.get("overlay_points")
        if not isinstance(pts, list):
            p.append(f"{where}: overlay_points must be a list")
        else:
            # each point must be an [x, y] pair of numbers - anything
            # else crashes the overlay renderer at draw time
            for j, pt in enumerate(pts):
                if not (isinstance(pt, (list, tuple)) and len(pt) == 2
                        and all(isinstance(v, (int, float))
                                and not isinstance(v, bool)
                                for v in pt)):
                    p.append(f"{where}: overlay_points[{j}] must be "
                             f"[x, y] numbers, got {pt!r}")
                    break
    for name in TRAITS_20:
        if name not in seen:
            p.append(f"traits: missing NDDB trait '{name}'")

    # THE ADOPTION GATE USED TO LIVE HERE, AND HAS BEEN REMOVED ON PURPOSE.
    #
    # It rejected any pipeline result in which every trait scored null, on the
    # reasoning that a fully unscored result must not displace a working
    # baseline. That reasoning treated "measured nothing" as a malformed
    # result. It is not malformed - it is a finding, and it is often the only
    # true thing the system can say.
    #
    # Because a rejected result fell through to the baseline engine, this gate
    # did the exact opposite of what its name suggests: it converted honest
    # refusals into fabrications. Measured on this build, every one of these
    # returned HTTP 200 with twenty confident scores and a weight near 350 kg:
    #     a drawing of a chair        (pipeline: 0/20, no_animal_detected)
    #     pure random RGB noise       (byte-identical output)
    #     a 44-byte ASCII text file   (byte-identical output)
    # along with 12 of 16 real photographs of Indian cattle and buffalo taken
    # without an ear-tag close-up. One of those fabrications wrote a Lumpy Skin
    # Disease row into a veterinary officer's alert feed.
    #
    # A validator's job is to check SHAPE. Whether a shape-valid result is
    # worth adopting is a policy question, and it now sits in one place -
    # server/scoring_loader.py - where it is answered "yes, always": a
    # contract-valid result is the answer, and each trait carries its own
    # not_scored_reason for the app to show.
    #
    # If you are considering restoring this, note that the two rules it was
    # protecting are still enforced below and unchanged: weight_kg low and
    # high must be null together or numeric together (never one of each), and
    # every one of the twenty NDDB traits must be present.

    w = result.get("weight_kg")
    if not isinstance(w, dict):
        p.append("'weight_kg' must be an object {low, high, method, ...}")
    elif w.get("low") is None and w.get("high") is None:
        pass  # honest "could not measure" (no-fake-data rule) - the app
        # shows "not measured" instead of an invented number
    else:
        if "method" not in w:
            p.append("weight_kg: missing 'method'")
        # low/high may be None together - the honest "not measured" state
        # (e.g. Heart Girth requires a 3D model that doesn't exist yet, see
        # ml/weight/estimator.py). A fabricated number is worse than an
        # honest null, so null is valid here - the adoption gate above is
        # what prevents an all-null *result* from being adopted, not this
        # field in isolation. What's not valid is exactly one of the two
        # being null, or either being present but not a number.
        low, high = w.get("low"), w.get("high")
        if low is None and high is None:
            pass  # honest not-measured state
        elif low is None or high is None:
            p.append("weight_kg: 'low' and 'high' must both be null "
                     "(not measured) or both be numbers, not one of each")
        elif not (isinstance(low, (int, float)) and isinstance(high, (int, float))):
            for k, v in (("low", low), ("high", high)):
                if not isinstance(v, (int, float)):
                    p.append(f"weight_kg: '{k}' must be a number or null, "
                             f"got {type(v).__name__}")
        if isinstance(w.get("low"), (int, float)) and \
           isinstance(w.get("high"), (int, float)):
            if w["low"] > w["high"]:
                p.append("weight_kg: low > high")
            # shape-valid but physically absurd weights (unit/scale bugs)
            # must not pass the gate - adult cattle/buffalo territory only
            if not (30 <= w["low"] <= 1500 and 30 <= w["high"] <= 1500):
                p.append(f"weight_kg: implausible range {w['low']}-{w['high']} kg "
                         "- check units/scale calibration")

    cap = result.get("captured")
    if not isinstance(cap, dict) or \
       set(cap or {}) != {"side_photo", "rear_photo", "gait_video"}:
        p.append("'captured' must be {side_photo, rear_photo, gait_video} booleans")

    # None where a list belongs crashes downstream iteration - reject it
    for key in ("symptom_vector", "health_flags"):
        if key in result and not isinstance(result.get(key), list):
            p.append(f"'{key}' must be a list, got {type(result.get(key)).__name__}")

    sv = result.get("symptom_vector")
    for i, s in enumerate(sv if isinstance(sv, list) else []):
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
