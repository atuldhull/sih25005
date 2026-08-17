# PASHU MITRA — Pitch Script (Team ASTRAL)

Target: 4 min 15 s total (safe inside a 5-minute cap). ~140 spoken
words/minute. Italic lines are stage directions — never read aloud.
Rehearse the two handoffs until they need zero thought.

Slide mapping: Atul = slides 1–2 · Tharun = slides 3–4 · Ritika = slides 5–6.

---

## ATUL — Opening & Overview (~85 seconds, ~205 words)

*(Slide 1. Stand centre. Calm pace — you own the room's first impression.)*

Good morning, judges. We are **Team ASTRAL**, presenting for problem
statement **SIH260486** — *Image-based Animal Type Classification for
cattle and buffaloes* — from the Ministry of Fisheries, Animal Husbandry
and Dairying.

Before our solution, one thing needs clearing up — because it is the
single most common mistake on this problem statement. *(small pause)*
"Type classification" does **not** mean telling cow from buffalo, or
identifying the breed. Type classification is a formal scoring exercise:
**NDDB's official scorecard of twenty body-structure traits** — stature,
rump angle, udder depth — each scored one to nine. These scores decide
which animals India breeds from under the Rashtriya Gokul Mission.

So where exactly is the problem? *(gesture to the peach card)* India has
**34.22 crore** tagged animals. The number that have ever been
type-classified? **Twenty-seven thousand.** Not crore. Twenty-seven
thousand records — total. Because scoring requires a trained classifier
to physically visit every farm, and those classifiers barely exist.

*(Slide 2)*

Our answer is **Pashu Mitra**. Two photos and an eight-second walking
video, from any field worker's phone, produce the complete official
scorecard. *(sweep a hand across the phone-screen strip)* This is the
whole flow in the worker's hand — scan the tag, two guided photos, a
short walk, saved offline. Three things separate us from every other team today: the
animal's own **ear tag becomes our measuring ruler** — no published
system anywhere does this; our system **refuses to score when it
shouldn't**, instead of guessing; and this is **not a concept — it runs
today**.

**Handoff:** "Tharun will take you inside the pipeline." *(step left,
stay visible)*

### Atul's interruption survival kit
- **"Isn't this just breed classification?"** → "No — breed is a field
  the government database already stores; we verify it from the photo as
  a fraud check. The twenty conformation traits are what nobody can
  capture at scale today; that's the entire problem statement."
- **"Where's the BPA API?"** → "There is no public one — we say so
  openly. We run a faithful mock and a thin adapter, so post-selection
  integration is a swap, not a rebuild."
- **"Is the prototype real?"** → "Fully — backend, scorecard flow,
  offline sync, knowledge-graph screening and the multilingual voice
  assistant are running, with forty-plus automated tests. We can demo
  live right now."
- **"Data privacy?"** → "Everything runs on our own server; voice clips
  are deleted after transcription; facts in every answer come only from
  the animal's own record."

---

## THARUN — The Solution, Deep (~115 seconds, ~270 words)

*(Slide 3. Point at the pipeline column, walk it top to bottom.)*

Thank you, Atul. Let me walk you through what happens after the field
worker points the phone.

The app guides the capture: side photo, rear photo, short walking video.
Before we score anything, **two gates** run. A quality gate rejects blur
or a half-visible animal. Then NDDB's own eligibility rule — first
lactation, day thirty to ninety after calving. If the animal is not
eligible, we refuse and tell the worker exactly why. That refusal rule
is straight from NDDB's guideline.

Now the part I enjoy most. Every registered animal wears a government
ear tag, and its printed features have **specified physical sizes** —
the round button is exactly twenty-seven millimetres. When that tag
appears in a photo, we solve its 3D pose, and suddenly we know real
centimetres for the whole image. **The government already placed a ruler
on every animal in India — we are the first to use it.** Zero extra
hardware, forever.

From there: keypoint detection finds the body landmarks; geometry gives
us angles, ratios and centimetre measurements; and ordinal scoring heads
produce the twenty scores, one to nine — each with a confidence value,
and each with its **evidence drawn on the photo**. Tap "rump angle,
score seven" — you see the measured angle on the animal. No black box.

Our whole stack — Flutter, FastAPI, RT-DETR, RTMPose — is free and
**licence-safe for government deployment**. Zero AGPL anywhere.

*(Slide 4)*

Is it realistic? Every step is published: body-condition scoring from
plain 2D video already matches expert vets at **84.6 percent**; weight
from one side photo reaches **6.2 percent error on Sahiwal and Red
Sindhi**. And our risks are on the slide with working mitigations — we
would rather show you the risk table ourselves than have you find it.

**Handoff:** "Ritika will show you what this changes on the ground."

### Tharun's interruption survival kit
- **"Why ordinal regression, not classification?"** → "Because a seven
  is closer to a six than to a two — plain classification can't see
  that. The one commercial system that matched human experts used
  ordinal regression too."
- **"What if the tag is at a bad angle or dirty?"** → "solvePnP on a
  flat plane is mathematically exact for pose; the 27-mm button is our
  cross-check; every centimetre value carries a confidence interval.
  And the angle-based traits need no ruler at all — they ship first."
- **"Training data?"** → "AP-10K gives us pre-trained animal pose across
  54 species; we fine-tune with a few hundred annotated Indian images;
  and CowDatabase — 103 cattle with tape-measured ground truth — lets us
  publish a real accuracy table with zero fieldwork."
- **"Why not full 3D reconstruction?"** → "Published data shows even a
  three-camera stereo rig measures heart girth worst of all traits.
  Single-photo metric 3D doesn't exist in the literature. It's on our
  roadmap, not our critical path — and we can defend that choice with
  citations."
- **"What accuracy will you claim?"** → "Within ±1 point on the 1–9
  scale, validated leave-animal-out. For context: trained human scorers
  agree exactly only 30 to 58 percent of the time. Consistency is the
  machine's advantage."

---

## RITIKA — Impact & Close (~75 seconds, ~180 words)

*(Slide 5. Warmer register — this is the human part.)*

Thank you, Tharun. So what actually changes on the ground?

For the **farmer**: an animal's weight without a weighbridge — with
photo-proof they can see and trust. Early disease screening, with
village-level outbreak alerts — remember, the 2022 lumpy skin outbreak
killed over **ninety-seven thousand cattle** in three months. And a
voice assistant that answers in Hindi or Kannada, from **their own
animal's record** — never generic internet advice. All of it fully
offline, because village networks are what they are — it syncs whenever
the network appears.

For the **nation**: indigenous-breed improvement finally gets scorecard
data at population scale, with machine consistency — repeatability of
0.99 against 0.94 for human scorers — at **zero hardware cost**, on
phones already in workers' pockets.

*(Slide 6)*

And we don't ask you to take our word for anything — every claim we made
today traces to NDDB's own guidelines or a peer-reviewed paper, all on
this slide.

*(pause — look up, slow down)*

Judges — the government built the module. The tags are already on the
animals. The only missing piece was the classifier at the farm gate.
**Pashu Mitra makes every field phone that classifier.** We are Team
ASTRAL — and our live demo is ready whenever you are. Thank you.

### Ritika's interruption survival kit
- **"How does offline actually work?"** → "Every session saves locally
  first with a device-generated ID; a background listener uploads when
  network appears; retries are duplicate-safe by design, so nothing is
  ever lost or double-counted."
- **"Will farmers really use this?"** → "They never type — the app
  guides three captures with an on-screen outline, and they can simply
  speak to it in their language. The photo-proof overlays build trust:
  they can see why an animal scored what it scored."
- **"Low-end phones?"** → "Capture and queueing are lightweight
  on-device; heavy models run on the server. Published systems run
  comparable models on hardware weaker than a budget phone."
- **"Which languages?"** → "Hindi, English and Kannada today — spoken
  and written, auto-detected — and the reply language can be forced from
  a selector. The architecture adds a language, not a rewrite."

---

## Q&A ownership (decide once, never glance around)
- ML / models / accuracy / training → **Tharun**
- App / offline / UX / farmers / languages → **Ritika**
- NDDB rules / architecture / government integration / citations /
  "how are you different" / anything unclaimed → **Atul**

## Demo roles
Atul drives (his laptop, his server) · Ritika narrates the screen ·
Tharun explains under-the-hood on request.

## Timing cuts
- Told "3 minutes"? Tharun drops the licence-safe line + one evidence
  number; Ritika drops the κ sentence. Everything else stays.
- Running long on stage? The references slide narration is one sentence
  — never skip the final two lines of the close.
