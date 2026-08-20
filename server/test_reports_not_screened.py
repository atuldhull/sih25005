"""A report must not tell a farmer their animal is well when nothing looked.

This build has no trained symptom detector. The screener says so plainly - its
own note reads "An empty symptom list here means NOT SCREENED, not HEALTHY" -
but that note went into the pipeline's warnings and the reports never saw it.
So symptom_vector was always empty, and empty was being rendered as:

    farmer   "No health problems were flagged from today's photos and video.
              Keep up the regular care."
    vet      "Detected signs: none."

Both read as a clinical finding. A farmer might skip a visit on the strength
of the first, and the second is written to a veterinary officer's feed.

    venv/Scripts/python test_reports_not_screened.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import reports  # noqa: E402

ANIMAL = {"_id": "356279812345", "breed": "Gir", "species": "cattle",
          "village": "Anand"}
SYMPTOM = [{"symptom": "udder_swelling", "confidence": 0.75, "region": "udder",
            "source": "image"}]
RISK = [{"condition": "mastitis", "label": "Mastitis", "urgency": "high",
         "risk": "high", "score": 0.8, "because_of": ["udder_swelling"],
         "action": "refer to vet"}]

failures = 0


def check(ok, name, detail=""):
    global failures
    if ok:
        print("PASS  " + name)
    else:
        failures += 1
        print("FAIL  " + name + (("\n      " + detail) if detail else ""))


# --- nothing screened -----------------------------------------------------
r = reports.build_reports(ANIMAL, [], [], [], screened=False)
low = r["farmer"].lower()
check("did not" in low or "not check" in low,
      "the farmer is told the animal was NOT checked for illness",
      r["farmer"][:200])
check("keep up the regular care" not in low,
      "and is not reassured about health that was never assessed")
check("veterinary" in low or "vet" in low,
      "and is pointed at a vet if the animal seems unwell")
check("NOT SCREENED" in r["vet"],
      "the vet officer's summary says NOT SCREENED", r["vet"][:200])
check("Detected signs: none" not in r["vet"],
      "and never reports a negative finding for an examination that did not "
      "happen")

# --- screened and clear ---------------------------------------------------
r2 = reports.build_reports(ANIMAL, [], [], [], screened=True)
check("no health problems were flagged" in r2["farmer"].lower(),
      "a real clear result still reads as a clear result")
check("Detected signs: none." in r2["vet"],
      "and the vet summary says so plainly")

# --- findings imply a screening happened ---------------------------------
r3 = reports.build_reports(ANIMAL, RISK, SYMPTOM, [])
check("Mastitis" in r3["farmer"] and "Mastitis" in r3["vet"],
      "findings are reported even when the caller forgot the flag")
check("NOT SCREENED" not in r3["vet"],
      "detected symptoms are themselves proof a screening ran - findings must "
      "never be announced under a heading saying nothing was examined")

# --- the disclaimer survives every path ----------------------------------
for label, rep in (("not screened", r), ("clear", r2), ("findings", r3)):
    check(reports.DISCLAIMER in rep["farmer"] and reports.DISCLAIMER in rep["vet"],
          f"the disclaimer is present on the {label} path")

print("\n" + ("ALL CHECKS PASSED" if not failures else f"{failures} FAILED"))
sys.exit(1 if failures else 0)
