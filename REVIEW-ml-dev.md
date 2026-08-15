# Review of ml-dev commit 0e4c336 (Person 2's ML pipeline)

Reviewed 2026-08-15 against the frozen contract + team plan. Verdict up front:
**genuinely strong engineering skeleton — right interface, right discipline,
26/26 tests pass — but it was written without the frozen contract (branch
forked before the freeze), so the output shape must be remapped, plus one
licensing fix and a few geometry bugs before integration.**

`main` has already been merged into `ml-dev` for you — `git pull` on ml-dev
and you'll have `contract/scoring_result.json` (the source of truth) and a new
tool: **`py contract/validate_result.py`** self-checks reference outputs, and
`validate(your_dict, mode="pipeline")` gives you an exact list of remaining
contract violations. Work until that list is empty — no guessing needed.

---

## What is excellent — keep exactly as is

- `score_animal(side_img, rear_img, video_path, animal_record)` — signature
  matches the agreement character-for-character.
- **Never-crash, never-lie discipline**: with zero ML backends installed the
  pipeline returns a truthful NOT_SCORED result instead of a stack trace or a
  fake number. That is our "refusing to score is a feature" pillar, in code.
- Test suite: 26/26 pass in a 3-package venv (no torch needed) because models
  are faked at the seam. Proper dependency injection under hackathon pressure.
- Weight formula: metric Schaeffer `girth² × length / 10838.4` — derivation
  documented and arithmetically exact (verified).
- The 24-point keypoint schema + `normalize_keypoints()` seam (missing joint →
  confidence 0 → dependent traits not_measurable) is exactly the design that
  prevents a week-3 integration disaster.
- Commit hygiene: no weights/binaries committed, honest TODO stubs.

---

## Blockers (must fix before ml-dev merges to main)

### B1. Output shape ≠ frozen contract
`scoring_result_to_dict()` emits your internal dataclass shape. The contract
needs different keys almost everywhere:
- `eligibility{passed, reasons[]}` → `eligible` (bool) + `eligible_reason` (str)
- `weight{estimate_kg, range_kg, confidence}` → `weight_kg{low, high, method, cross_check}`
- missing: `captured{side_photo, rear_photo, gait_video}`, `breed_registered`,
  `breed_verified`, `breed_verify_confidence`
- `symptom_vector` items `{category, present, confidence}` →
  `{symptom, confidence, region, source}` (symptom names MUST come from the
  vocabulary in `server/vkg.json`)

**Fix:** keep your internal dataclasses — add ONE adapter function
`to_contract_dict()` in `result_builder.py` that maps to the contract keys.
Never let the internal shape leak past it.

### B2. Per-trait shape shares only `confidence` with the contract
`{trait_id, score_1_9, confidence}` → contract wants
`{name, category, score, confidence, measured_value, ci, measure_class, view,
overlay_points, explanation}` + `not_scored_reason` when score is null.
The good news: your explainer already computes explanations AND overlay
points — they're just being misfiled into `warnings` and dropped
(`pipeline.py` passes only `text_summary` through). Route them into the trait
objects.

### B3. Trait registry is a different 20 traits
Only 5 names match the NDDB scorecard. Rename the near-misses
(`Height at Withers`→`Stature`, `Chest Depth`→`Body Depth`,
`Pastern (Foot) Angle`→`Foot Angle`, `Rear Leg Set`→`Rear Legs Set`), add the
11 missing (incl. **8 of the 9 Udder traits** — the app's biggest section —
plus Angularity, Rear Legs Rear View, Body Condition Score), add `category` +
`view` fields, add `"SMAL"` to the trait_class Literal. Keep your extra
angle/ratio traits as an internal feature layer — just don't emit them in
`traits[]`. Buffalo hooks: rump-width landmarks differ; teat traits measure
the left REAR teat (cattle: left front). Udder traits need teat/udder
keypoints — extend the 24-point schema (your own docs say the annotation
schema must be final before dataset work, so this is urgent).

### B4. Ultralytics (AGPL) is the preferred detection backend
`load_rt_detr()` tries `from ultralytics import YOLO` first and its error
message says "install ultralytics". The license-safe story is one of our pitch
differentiators — a judge grepping the repo would find the exact thing the
plan bans in caps. **Fix:** delete the ultralytics path; use HuggingFace
`transformers` → `RTDetrV2ForObjectDetection` (Apache-2.0). Your
`_parse_predictions()` is already backend-agnostic, so it plugs almost
straight in.

---

## Important fixes

1. **Angle normalization** — `_compute_angle()` returns raw `atan2`: the same
   rump slope reads 5.7° facing left but 174.3° facing right → score flips
   6 → 1. Fold line angles into (−90, 90°]. Add a mirrored-keypoints test.
2. **Leg-set geometry is degenerate** — a severely cow-hocked animal scores a
   perfect 5 @ 0.95 confidence (keypoint pairing merges the wrong joints).
   Compute per-leg deviation from vertical instead, average left/right.
3. **Chest depth masquerading as Heart Girth** — a ~75 cm depth squared in
   Schaeffer's formula gives ~78 kg instead of ~448 kg. Either convert
   explicitly (girth ≈ 2.3–2.5 × depth, label `method: "depth-proxy"`) or mark
   Heart Girth not_measurable until SMAL/real girth exists.
4. **Out-of-range values clamp to extreme scores** — a garbage measurement
   becomes a confident-looking 1 or 9. Contract rule #1: refuse with
   `not_scored_reason` instead. Refusal is a feature we pitch.
5. **`ml/` is not importable** — no `ml/__init__.py` and imports are rooted at
   ml/ itself, so `from ml.pipeline import score_animal` crashes (and
   `scoring`/`rules` module names collide with the server's). Add
   `ml/__init__.py` + package-qualified imports (`from ml.common.schemas
   import ...`), update the 3 test files, delete the stale
   `sys.path.insert(0, "ml_pipeline")` lines.
6. **requirements.txt lists phantom packages** — `rtdetr-pytorch`, `rtmpose`,
   `dino-v2` don't exist on PyPI (verified). Real routes: `transformers`
   (RT-DETRv2 + DINOv2), `rtmlib` (RTMPose inference), and `paddleocr`
   additionally needs `paddlepaddle`. Also add `numpy` (directly imported).
7. **Tag spec numbers** — v1 uses a 5.0×3.5 cm panel as the ruler; the plan
   says the panel is vendor-variable (55–69 mm) and must NOT be the scale
   source. The trusted printed sizes: button 27 mm ±2, barcode line 10 mm,
   digit line 18 mm. Panel corners stay useful for solvePnP POSE only.

## Minor

- `check_eligibility()` in ml/ is a quality gate, not NDDB eligibility —
  rename to `scoreability` to avoid colliding with the contract's `eligible`
  (the server owns the NDDB rule).
- `rump_angle` `reverse=True` inverts the ICAR direction convention — drop it.
- Mock tag IDs aren't 12-digit Pashu-Aadhaar-shaped — reuse the server's
  seeded animals (`server/seed.py`).

---

## Division of labor (agreed shape)

- **Person 2 emits (pipeline mode):** animal_id, species, breed_*, captured,
  traits[20], weight_kg, symptom_vector, health_flags.
- **Server injects:** session_id, eligible/eligible_reason (recomputed),
  risk_report, herd_alerts, reports, escalated, captured_at, synced.
- Check yourself anytime with:
  `venv python -c "from validate_result import validate; ..."` or just run
  `py contract/validate_result.py` for the reference self-checks.

Suggested order: B1+B2+B3 together (one evening of mapping work — the data
mostly exists already), then B4 + fix 5 (unblocks integration), then the
geometry fixes 1–4, then the rest.

---

# Verification round 1 — commits 68c986e, 0b6fb13, 9bc3cb6 (2026-08-19)

Independently verified on a clean checkout. Excellent progress:

**Confirmed fixed ✅**
- B3 registry: 20 contract traits, categories, SMAL class — scratch test
  emits all 20 and validates clean.
- B4 licensing: zero ultralytics imports remain (only comments explaining
  the removal); transformers RT-DETRv2 backend in place.
- Fix 5: `from ml.pipeline import score_animal` imports cleanly; all 25
  tests pass in a minimal venv.
- Fix 6: requirements.txt is all real, installable packages.

**One wiring gap left — the only thing between you and the merge 🔴**
`score_animal()` still returns the OLD internal shape (29 contract
violations at the entry point). Your `to_contract_dict()` works — but
nothing calls it. Since it needs the internal intermediates
(ScoringResult, measurements, keypoints), the call has to happen inside
`score_animal()` where they're in scope:
build result → `return to_contract_dict(result, measurements, keypoints,
breed_registered=animal_record.get("breed"), captured={...from the three
input args...})`. Roughly 10 lines. Re-run
`py contract/validate_result.py`-style check on `score_animal()` output
(not the adapter directly) and you're green.

Also: promote `ml/test_contract_output.py` from scratch to a real test in
`ml/tests/` asserting `validate(score_animal(...), mode="pipeline") == []`
— that's the exact merge gate, in your suite.

**Heads-up: the validator now checks plausibility too.** Your scratch run
produced weight 0.41 kg — shape-valid, physically absurd (synthetic-unit
artifact). `validate_result.py` now rejects weight outside 30–1500 kg, so
make sure the degraded path returns weight honestly (contract allows
scoring refusal; it doesn't allow a 0.4 kg cow).

**Still open from the important list**: angle folding for facing-direction
(imp. 1), clamp-vs-refuse on out-of-range values (imp. 4), and verify the
leg-set deviation-from-vertical rewrite landed in code, not just comments
(imp. 2). Heart girth to not_measurable-until-SMAL (imp. 3) looks
addressed via the registry — nice.
