# Measured or invented

An audit of what this system reports, what it refuses, and what it was making
up. Every number here came from a command that can be re-run against this repo.
Where something was not executed, it says so.

Companion to [STATUS-measured.md](STATUS-measured.md). Fixes shipped in commit
`e3df4e1`.

---

## 1. The finding that outranked the questions we asked

We set out to answer four questions about accuracy. Before reaching any of
them, the audit found that the server would score anything at all — and that
when the real pipeline honestly refused, the server substituted invented
numbers in its place.

| input | what the pipeline said | what the server returned |
|---|---|---|
| a drawing of a chair | 0/20, `no_animal_detected` | 20/20, Stature 127.9 cm, 348–376 kg, **LSD alert in the vet feed** |
| pure random RGB noise | 0/20, `no_animal_detected` | byte-identical numbers |
| a 44-byte text file | 0/20, `unreadable_image` | 20/20, `skin_nodules` @ 0.82, source "photo" |
| 16 real cattle/buffalo photos, no tag close-up | **12 of 16 scored zero** | all 12 fabricated |

The invented figures come from `random.Random(animal_record["_id"])`
(`server/scoring.py:45`), so they are stable per animal and reproduce exactly —
which makes them look like a measurement.

### The cause: a safety gate working backwards

A pipeline result with all twenty traits null was treated as a **contract
violation**, so a half-built pipeline could not displace the baseline engine.
The intent was sound. But `scoring_loader` answers a rejected result by
*calling* the baseline engine, which invents all twenty scores. The gate never
suppressed the refusal — it swapped it for a fabrication.

It lived in **two** places, which is why fixing one changed nothing:

- `contract/validate_result.py` — a validator's job is shape, not policy
- `server/scoring_loader.py` — where the policy now sits alone

### Fixed

Refusing is an answer. It now ships as `engine: ml-pipeline` with every trait's
own `not_scored_reason`, a null weight, and an empty symptom vector — so
nothing invented reaches a farmer, a report, or an officer's feed.

    chair  ->  engine ml-pipeline, 0 of 20, no_animal_detected,
               weight null, 0 symptoms, no alert
    real Sahiwal  ->  unchanged: 3 of 20, red_zebu, 172.7-274.4 kg

Pinned by `server/test_refuses_rather_than_invents.py`, which drives the loader
with a chair, noise and a text file. The two tests that asserted the old gate
now assert the new rule.

---

## 2. Can the weight be made more accurate?

Roughly **2× more accurate** — but not by tuning anything. The whole miss is one
number: the centimetre scale.

The pipeline used **0.069 cm/px**. Three independent anatomical anchors measured
off the silhouette put the truth near **0.090**:

| anchor | pixels | reference | implied cm/px |
|---|---|---|---|
| wither crest → hoof line | 1418 | 120–135 cm | 0.0846–0.0952 |
| barrel span | 1591 | 140–155 cm | 0.0880–0.0974 |
| chest depth at heart girth | 797 | 68–75 cm | 0.0853–0.0941 |
| **what the pipeline used** | — | synthetic tag | **0.0691** |

Weight goes as `cm_per_px ** 3` (`volume_3d.py:365`). Changing **only** the
scale moves the estimate from **172.7–193.8 kg to 382.1–428.8 kg**.

**The falsification that needs no ruler:** at the pipeline's own scale this cow
is 97.9 cm at the withers and 109.9 cm long — heifer dimensions. At 0.090 she is
127.6 and 143.2 cm. Adult Sahiwal.

### Three things spoil the obvious fix

**Her real ear tag was cropped and tested — the pipeline refused it.**
*"Ear tag found but its round button could not be measured"*, and the digit
fallback: *"found 1 ink band(s), need at least 2"*. She wears a large farm
management tag (S-102), not an NDDB tag. Worse: had the ruler accepted it and
applied the NDDB panel spec (5.5–6.9 cm), the answer would be **93–205 kg** —
worse than the current wrong value. To reach the anatomical scale, that 97 px
blob would have to be 8.25–9.21 cm tall, physically bigger than the panel the
ruler assumes.

**Fixing the scale breaks the one output that currently looks right.** Heart
Girth reads 164.5 cm against a 165–185 reference. At the corrected scale it
becomes **214.4 cm** — impossible. The ellipse-perimeter girth model is
independently ~15% high and is currently cancelling part of the scale error.
Two defects, partly offsetting. They have to move together, or fixing the scale
looks like a regression.

**The cross-check was never independent.** Schaeffer came out at exactly
**1.4972 × the volume midpoint at every scale** in the sweep, because girth
carries the scale squared and length carries it once — so both routes carry it
cubed and the ratio is a constant. It can catch a shape error and never a scale
error, and scale is the error. Now labelled *same-scale shape check* in the
output.

### Separately, affecting 2 of the 3 traits that score

Rump Width 36.2 cm and Udder Depth 6.5 cm are **rear-photo pixel distances
multiplied by the side photo's cm/px** (36.2 = 523.1 rear px × 0.06907). The two
frames differ by roughly 1.7–2×. `volume_3d.py:13-24` sets out exactly why this
is invalid and the volume path obeys it; the trait path does not.

### Honest verdict

Tag-as-ruler cannot carry this photograph. Real accuracy needs a genuine
close-up of the tag the animal actually wears **with its make and physical size
stated**, or a known-size object placed in frame. That is a person with a
camera, not a commit.

---

## 3. What the assistant can actually answer

It was serving pre-fed records and it was not estimating — partly a safety rule
doing its job, partly three real defects behind it. All four below are fixed and
verified live.

| asked | before | now |
|---|---|---|
| what is her weight? | "223–**362** kg" — 362 was a seeded random number | "between 173 and 274 kg… two methods disagree, a weighbridge is recommended before selling or dosing" |
| her udder is swollen and hard | steered to Lumpy Skin Disease | cites `vkg.json — Mastitis` |
| how do I file my income tax return? | scored 0.541 "strong", answered with citations | `sources: []`, clean refusal |
| वज़न कितना है? | truncated "किलोग्र", invented non-words | fluent Devanagari |

### Why each was wrong

**The trend had no per-session filter.** One line below the careful check that
withholds a demonstration weight (`chat.py:131`), the trend list took every
session's weight regardless of engine. Measured before the fix, the context read
`Weight trend: 223.0 -> 223.0 -> 362 -> 362 -> 223.0 kg`. Now filtered per
session.

**An animal with no session produced no context line, and the model filled the
silence.** Animal `356279812348` was told *"we can estimate Tharparkar cattle's
weight using the girth-length method"* — describing a procedure that never ran.
It now states plainly that no session exists.

**12 conditions of real advice were unreachable.** `server/vkg.json` already
held authored care advice — 1,407 characters of English and 1,279 of Hindi
across mastitis, lameness, LSD, FMD and the rest — but only reachable if the
last session had flagged that exact condition. Now indexed alongside
`knowledge/*.md`. **No new veterinary text was written, and none should be.**

**The relevance gate sat below the embedder's noise floor.** At 0.45, gibberish
`asdfgh qwerty zxcvb` scored 0.474 and passed, while *"how do I treat
mastitis?"* scored 0.493. Every answer got citations and the citations were
decoration — a deworming answer cited *"capture-guide.md — How to take the rear
photo"*.

Re-measured after indexing vkg, 8 in-corpus queries against 6 out-of-corpus:

    relevant  0.617 - 0.818   (min: "she is very thin and not eating")
    junk      0.404 - 0.516   (max: "what time is the train to Delhi?")

Set to **0.57**, in the gap. Sample is 14 queries — treat it as a floor that
separates the cases we tested, and re-measure rather than nudging it.

**The model could not spell.** qwen2.5:7b truncates Hindi "kilogram" to
किलोग्र and emits Korean and Chinese glyphs for Kannada, carrying a fabricated
weight range — which `_reply_ok` waved through because it only checked that
Kannada codepoints were *present*. Switched to **gemma2:9b** (already pulled
locally), timeout 20s → 45s, and the gate now rejects CJK in an Indic reply.
**These three had to move together**: raising the timeout alone would have let
the glyph soup reach the farmer instead of failing safely.

---

## 4. Testing on a real phone

Closer than expected. The server already binds `0.0.0.0`, and Windows Firewall
is **not** blocking it — an existing allow rule already covers the python binary
that owns the socket. Do not add a firewall rule; you do not need admin.

**Over USB:**

    adb reverse tcp:8000 tcp:8000

then set the app's server address (Settings → Server address) to
`http://127.0.0.1:8000`. No admin, no firewall change, and immune to the one
real fragility — the existing allow rule is Public-profile only, so a Wi-Fi
reclassification to Private would break LAN access. Re-run after every unplug.

Over Wi-Fi instead, the laptop is `192.168.29.51`.

### Three blockers

1. **Resolution.** All four capture screens use `ResolutionPreset.medium` =
   720×480. The ear-tag panel measures 191 px native but **48 px** after
   downscaling, against `MIN_PANEL_PX = 40`. A 20% margin. `high` (1280×720)
   makes it 80 px.
2. **Demo mode is a compile-time constant** (`DemoCameraConfig.enabled`, a
   `static const`). A recompile and full restart — and shipping the wrong
   constant means the app photographs a bundled cow while pointed at a real one.
3. **A latent dead screen.** `video_capture_screen.dart:66` requests the
   microphone, but every controller sets `enableAudio: false` so the CameraX
   plugin never needs it. Declining it sets `_isCameraReady = false` and renders
   a black screen with an infinite spinner — no error, no retry, and
   `openAppSettings` appears nowhere in the repo. Harmless today only because
   demo mode returns before that line. It goes live the moment you flip the flag.

**No physical phone was ever attached.** Every claim here is from source, the
merged APK manifest, and plugin sources. Expect the numbers to differ from the
demo baseline.

---

## 5. Will it figure out a random cattle photo?

No — by design in one place and a gap in another.

`main.py:298` looks the tag up in the records and returns 404 before anything is
saved; tag `999999999999` creates no session directory. **No HTTP route can
create an animal record.** And eligibility needs `lactation_no` and
`last_calving_date` — registry history, not appearance. `grep -rln lactation`
over `ml/` returns zero files. The camera does not even supply the lookup key:
`tag_reader.py:19-21` says OCR is not implemented and hardcodes
`identity: None`, so a human types the twelve digits.

### The one thing it gets right was being thrown away

The species/group classifier ran and was correct on all 13 images where it made
a call — it labelled an HD Murrah buffalo "buffalo" at 0.999 against a Sahiwal
record. Because zero traits scored, that correct verdict was discarded along
with everything else and replaced by the fake result, which reports
`predicted_species: null`. Now that refusals survive, so does the
classification.

Present it as a **check on the record, never a correction**. Honest
source-held-out accuracy is 88.5% species, 80.2% group against a 60.7%
background-only control, and 1 of 16 was a false accusation. Those 16 images
come from a source the classifier trained on, so 13/13 is optimistic.

---

## 6. Shipped in `e3df4e1`

- Honest refusal replaces the invented scorecard — `server/scoring_loader.py`,
  `contract/validate_result.py`
- Baseline weights filtered out of the chat trend, and a no-session animal says
  so — `server/chat.py`
- gemma2:9b, 45s timeout, CJK rejected in Indic replies — `server/chat.py`,
  `server/preflight.py`
- `vkg.json` indexed, threshold recalibrated to 0.57 — `server/rag.py`
- The weight range and its disagreement reach the farmer — `server/chat.py`
- "Nothing could be measured" state in the app — `lib/widgets/result_cards.dart`,
  `lib/screens/capture/session_saved_screen.dart`
- The cross-check says what it can and cannot catch — `ml/weight/volume_3d.py`

411 ML tests, 11 server suites, the demo console suite, `flutter analyze` and
the Flutter tests all pass.

---

## 7. Next, ranked by accuracy per hour

1. **Refuse the tag scale transfer when the two blobs are not the same shape.**
   Aspects here are 1.196 vs 1.702 — a 42% disagreement `PANEL_ASPECT_RANGE`
   swallows whole. Turns a confidently wrong number into an honest "no scale, no
   weight". *hours*
2. **Give the rear photo its own scale, or refuse rear-view class-C traits.**
   Refusing is the honest short-term answer since the rear photo carries no tag.
   It cuts the demo from 3 scored traits to 1. *days*
3. **Answer deterministic facts from the template, not the model.** Two of three
   runs told a day-88 cow inside the 30–90 window to "wait until day 91", which
   would end her eligibility — while `rules.check_eligibility` returns the right
   answer 100% of the time. *hours*
4. **Fix the two video-screen defects before flipping demo mode off.**
   `ensure(forVideo: true)` → `false`, and give the permission refusal a message
   instead of an infinite spinner. *minutes*
5. **Raise capture resolution to `high` on side, rear and tag.** Takes the tag
   panel from a 20% margin to 100%. It narrows the gap to the press photo the
   baseline came from; it does not close it. *minutes to change, hours to
   validate*
6. **Check the typed tag against the records before capture.** An unknown tag
   currently costs the farmer four captures, then a false "the server could not
   be reached" and a queue entry that retries forever. Only 404 and 422 are
   terminal; keep 5xx and timeouts on the retry path. *hours*
7. **Add a scale plausibility WARNING — report-only.** Nothing currently gates
   the scale, only the dimensionless shape ratio. It must stay a warning:
   `volume_3d.py:346-350` already argues that gating the output weight against
   an expected range makes the estimator confirm what it was told. *hours*
8. **Recalibrate or drop the elliptical heart girth.** Prerequisite for the
   scale fix not looking like a regression. Any shape factor chosen without
   taped animals is fitted to a reference table, not measured, and must be
   labelled as an assumption. *days*

---

## 8. Do not do these

Each would make a number look better without being better. They are listed
because they are the obvious next move, and someone will suggest them.

**Tune `DENSITY_KG_PER_M3`, or add a breed correction, so this Sahiwal reads
350 kg.** Reaching 400 kg by density alone needs about 2000 kg/m³ — twice water,
physically impossible for a live animal, and obvious to any judge who reads the
constant. The 2.2× scale error survives underneath, cancelled only for animals
photographed at this exact distance with this exact tag mismatch.

**Calibrate cm/px by assuming this cow's body length is 147.5 cm.** Produces a
comfortable ~380–430 kg and is circular: the answer becomes a restatement of the
breed table with a photograph attached. Legitimate only as a labelled
diagnostic — which is how it was used in this audit — never as a reported
measurement.

**Widen per-trait plausibility ranges to push past 3/20.** More traits would
score; none would be more correct. The refusals are the system catching its own
bad landmarks and are the most trustworthy thing in the output.

**Have a model write feeding, disease and breeding sections into
`knowledge/*.md` so the RAG has something to cite.** The single most dangerous
item here and the easiest to do. The sources would be model-generated text in a
government app advising farmers on medication timing and breeding,
indistinguishable in the UI from the NDDB-derived definitions beside it. The
corpus must be authored or licensed by a veterinarian.

**Treat baseline weights as measured so every animal shows a number.** Every
animal would answer 366 / 380 / 418 kg and the demo would look complete. Those
figures describe no photograph of any animal.

**Raise the chat timeout without swapping the model.** It turns a visible
failure into an invisible one. At a 300s timeout, qwen's Kannada output was
Korean and Chinese glyphs carrying a fabricated weight range — and `_reply_ok`
returned true on it.

---

## 9. Needs a person, not a commit

1. **A real tag photograph with its size stated.** Harder than it sounds. The
   demo close-up `3-tag.png` is a synthetic NDDB rectangle, and
   `assets/cow_demo/cow_tag.jpg` is md5-identical to it, so the Android demo
   path has the same defect. But photographing the cow's *actual* tag does not
   fix it either — the pipeline refuses that tag, and the NDDB spec applied to
   it gives 93–205 kg. Someone has to state the make and physical size of the
   tag this animal wears, photograph an animal wearing a genuine NDDB tag, or
   place a known-size object in frame.
2. **A weighbridge reading and a girth tape** on the actual animals. Nothing in
   this pipeline is calibrated — the density range, the elliptical
   cross-section, the ellipse-perimeter girth and the Schaeffer divisor are all
   uncalibrated against its own outputs, and `volume_3d.py:51-54` says so. The
   300–400 kg and 165–185 cm figures used throughout are a breed reference
   table, not a measurement of this cow.
3. **A veterinarian to author or license the husbandry corpus.**
   `server/knowledge/*.md` is 2,000 words with zero coverage of feeding,
   disease, treatment, vaccination, breeding, deworming, housing or minerals.
   The twelve vkg conditions are authored and now wired; anything beyond them
   must come from a vet or an official ICAR/NDDB publication cleared for
   redistribution.
4. **A native Kannada speaker, and a Hindi reviewer,** to write the
   deterministic template strings. The model swap fixes the LLM path, but the
   template path is what fires during emergencies and whenever Ollama is down,
   and it has no Kannada branch at all. Do not let a model write them.
5. **An annotation session for udder and teat keypoints.** Ten of the twenty
   traits return confidence 0.0 because those landmarks were never annotated —
   only 22 of 41 canonical joints are trained (`ml/pipeline.py:262-266`). Half
   the rubric is unreachable until someone annotates.
6. **A source-diverse, properly labelled breed dataset.** The checkpoint
   disables its own breed head at 38.1% source-held-out, and tightening
   confidence from 100% to 30% coverage moves accuracy only +5.6 points — the
   confidence carries no information. Bovine breed models score ~98% off the
   backdrop; always test source-held-out.
7. **A physical Android phone.** None was ever attached, so not one step of the
   on-phone procedure has been executed.
8. **A decision from the sponsor** on whether this app may give any
   treatment-adjacent advice at all. The current line is "general care allowed,
   medicines and doses vet-only", which is a reasonable guess, but the measured
   replies sit right on it. The topic gate cannot be written correctly until
   someone says where the line is.

---

## 10. What this audit got wrong

Eight claims went through an adversarial pass whose job was to refute them.
Three did not survive. Recorded here so they are not quietly repeated.

**"Windows Firewall will block the phone."** It will not. The process holding
`0.0.0.0:8000` is the system python, not the venv stub, and Windows matches app
rules against the socket-owning image — an existing allow rule already permits
it. *Do not run* `New-NetFirewallRule`. The one genuine risk it missed: that
rule is Public-profile only, so a Wi-Fi reclassification to Private would break
LAN access.

**"All seeded sessions were baseline, so the weight refusal applied to every
animal."** It applied to five. The other fifteen have no session at all, so the
withhold branch never ran and the model drifted instead — a measurably
different failure, and the one now fixed.

**"For general questions the retrieval has nothing to retrieve."** Retrieval
always retrieved: `rag.search` returns top-k unconditionally and the threshold
sat below the noise floor, so irrelevant chunks were injected under a header
reading "REFERENCE INFORMATION (official guideline notes)" and echoed back as
citations. A false-citation bug, materially worse than having nothing. Two
further corrections: calling the corpus "app documentation, not husbandry
knowledge" was overstated — `traits.md` is 20 of the 37 chunks of ICAR/NDDB
linear type-trait definitions — and `vkg.json` did contain real husbandry
advice all along.

**"Fix the scale and the weight becomes right."** Necessary but not sufficient.
A sweep of forced tag heights shows no tag size simultaneously reconciles
weight, heart girth and stature with the reference — Stature reads 54.90 cm at
the current scale and only 98.32 cm even at an absurd 12 cm tag, against a
120–135 cm reference. There is at least one further scale-independent landmark
or pose error that a correct tag photograph will not fix.
