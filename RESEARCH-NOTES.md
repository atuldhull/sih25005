# Research notes — published work relevant to SIH25005

Compiled 2026-08-19 from a literature sweep across six areas: trait/BCS
scoring, cattle keypoints & measurement, weight estimation, 3D
reconstruction, gait & disease detection, and animal identity.
~50 papers, links inline. Read this before writing the deck — every
claim we make on stage should lean on a number from here.

---

## The five headline validations (our design choices are published science)

1. **Ordinal scoring works at human level from plain 2D video.**
   CattleEye (J. Dairy Science 2024, 34,150 training scores): agreed with
   an expert within ±0.25 BCS on **84.6%** of 9,657 test scores — better
   than one of the two human expert pairs (75.3%) — using **ordinal
   regression**, exactly our CORAL/CORN plan. Machine repeatability
   kappa 0.99 vs 0.94 for the human re-scoring the same cows.
   https://www.journalofdairyscience.org/article/S0022-0302(23)00804-4/fulltext

2. **Our pipeline shape (segmentation → keypoints → measurement rules)
   is peer-reviewed.** Smart Agricultural Technology 2024: same
   architecture, >90% mIoU, **<15% MAPE** on linear type traits (RGB-D
   rig). RTMPose itself is validated on cattle: **82.9% AP at 39 FPS**
   (Agriculture 2023). Keypoint pipelines reach **MAE 1.52 cm (1.28%)**
   on body height (Animals 2024, 95 cattle).

3. **Ear-tag-as-ruler is confirmed white space — and scale is provably
   THE bottleneck.** No published cattle system uses the mandatory ear
   tag as its scale reference (every accurate system uses depth cameras
   or known distance). When a single photo relies on AI monocular depth
   for scale, weight error roughly **doubles: MAPE 10–11.6% vs ~6% for
   scale-controlled methods**. A physical scale anchor in every photo is
   the cheap, correct fix — that's our tag.

4. **"Comparator, not scorer" is what the inter-rater data demands.**
   Trained humans agree on the exact BCS only **30–58%** of the time;
   on 7-point locomotion scores, experts agree exactly only **45%**
   (kappa 0.23–0.60); among 18 type-trait classifiers, same-trait
   correlations differ from unity. Consistency is the machine's edge.

5. **Refusing to score is what the best systems do.** The commercial
   DeLaval 3D camera underestimates **92% of over-conditioned cows** —
   documented blind spots at the extremes. Confidence-gating (auto-accept
   only above ~65–70% confidence; Animals 2023) is published best
   practice, not our excuse.

---

## Area digests

### Trait scoring & BCS
- Research BCS systems cluster at 70–85% within 0.25 / 90–97% within
  0.5 units. Our analogous promise: "within ±1 point on the 1–9 scale".
- 3-class bins (thin/target/fat) reach near-perfect agreement where
  12-class doesn't (Animals 2023) → surface coarse bins + fine score.
- BCS CNNs run in ~535 ms on a Jetson Nano (Sensors 2023) — weaker than
  a mid-range phone → offline on-device inference is demonstrably real.
- ICAR-NDRI is already publishing on hump area / hoof angle from 200
  Sahiwal images (Indian J. Dairy Science 2026) — with NO accuracy
  numbers reported. Publishing a real MAE table beats them.
- 1 BCS unit ≈ 50 kg body weight → free cross-check between our BCS and
  weight modules.

### Keypoints & measurement
- Pretrain on **AP-10K** (NeurIPS 2021: 10,015 images, 54 species, 17
  keypoints, bovids included) → fine-tune on a few hundred Indian
  photos, not thousands.
- Benchmark cm-errors to target: body height 1.5–3 cm, body length
  ~2.3–6.5%, chest girth 4.4–5.2% (multiple 2024–2025 papers).
- **CowDatabase** (github.com/ruchaya/CowDatabase): 103 cattle, RGB +
  depth + 9 tape-measured ground truths — we can validate our
  measurement rules TODAY with zero fieldwork and publish our own MAE
  table. Best single evidence artifact available to us.
- Bos indicus proof: a smartphone + YOLOv11 system for Brahman cattle
  (12,660 field images, Smart Ag Tech 2025) matched manual measurement
  with MAPE <0.4% — phone-photo conformation on humped breeds works.

### Weight estimation
- The accuracy ladder: girth tape MAE 2.7 kg (restrained calves; but 32%
  of variance is who holds the tape) → 2D DL **6.22% MAPE / MAE 18 kg**
  on Sahiwal/Red Sindhi/nondescript (CattleNet-XAI, PLOS One 2025, 513
  cattle — OUR breeds) → monocular-depth-scale 10–11.6% → 3D ceiling
  2–3.2%. Our ear-tag-scaled target of 5–8% MAPE is credible, not hype.
- Girth-only regressions degrade by breed (R² 0.58–0.75 in adult dairy)
  → multi-measurement + breed feature, exactly our keypoint approach.
- Murrah buffalo: gradient boosting on 8 measurements = R² 0.82 / MAE
  5.3 kg from only 130 animals (Animals 2024) — a pilot a student team
  can collect.
- Always validate **leave-animal-out** (error jumps 2.03%→4.70% on
  unseen cows) — bake into any accuracy claim.
- Report weight as a range (point ± literature MAPE) — we already do.

### 3D reconstruction (why SMAL stays demoted)
- Single-photo metric 3D of cattle does not exist in the literature;
  SMAL fits are qualitatively right, not tape-measure right. A species
  template matters hugely (hSMAL halved horse error: 14→7 cm).
- Even a 3-stereo-camera rig measures **chest girth worst of all
  traits: 5.22% / 9.71 cm** (vs 2.32% body height) — the chest underside
  occludes. This kills "why not full 3D?" in Q&A.
- Hackathon path for girth without 3D: **elliptical estimate**
  (Ramanujan circumference) from chest depth (side view) + chest width
  (rear view), both ear-tag-scaled.
- Cheap upgrade: zero-shot monocular depth (Depth Anything) features
  lifted weight R² 0.90→0.95 on plain images (J. Animal Science 2026).
- Wow-slide option: fit SMAL to one good photo offline (smal-fitter /
  BM-GCN GUI) and show the rotating mesh — labelled "roadmap", never
  claimed as the measurement engine.

### Gait & disease
- Lameness recipe = ours: keypoints → 3 interpretable traits (**back
  arch #1**, head bob, tracking distance) → ~80%; best published 94%
  4-grade / 98% binary (Sci Reports 2023, 250 cows). **1 second of
  walking video suffices for 85%** (2025 preprint) — short phone clips
  are enough.
- Target binary lame/not-lame with confidence, NOT fine grades (humans
  only 45% exact themselves).
- **LSD**: MobileNetV2 hits **95%** from only 793 images (PLOS One
  2024); public data ≈ 3–4k images (Mendeley 1,024 + Roboflow 2,223 +
  Kaggle). TFLite int8 ≈ 3–5 MB → offline phone screening.
- The caution that justifies our VKG: LSD accuracy falls **96% → 85%**
  when look-alike diseases (ringworm, FMD) enter the test set
  (Veterinary Sciences 2024). A classifier alone over-claims; a
  differential layer (our knowledge graph + follow-up questions via the
  chatbot) is the published-correct design.
- Mastitis from RGB photos is NOT credible (all serious work uses
  thermal cameras: 96.3% sensitivity with FLIR-class hardware) — keep
  mastitis as a VKG symptom checklist, roadmap the thermal clip-on.
- India's 2022 LSD outbreak killed **97,000+ cattle in ~3 months**; no
  India-collected labeled LSD image dataset has been published since —
  our field data drive would be a first.

### Identity & breeds
- **Muzzle biometrics is the standout new idea**: muzzle prints are
  unique and lifelong ("cattle fingerprints"); four independent studies
  hit **96.5–99.5%** identification. The killer citations: an Indian
  ICAR-system study — 264 Vrindavani cattle, **97.22%, model under
  4 MB** (offline budget-phone capable, J. Dairy Research 2025) — and a
  Bangladesh deployment built explicitly because **insurers would not
  trust tamper-prone ear tags** (826 cattle, 96.49%, FPR 0.098%).
  Survives image compression to 25% size (99.5%).
- Ear-tag OCR alone is fragile: full-tag single-image read ≈ **65%**;
  multi-frame voting → ~93% (Frontiers 2022; Sensors 2024: 92.1%).
  OCR assists identity; muzzle + record anchor it.
- Breed classification honesty: even ICAR-NDRI reaches only **84–86%**
  separating Sahiwal from Red Sindhi. Validates our
  verify-don't-predict stance; for suggestions use top-3 + an open-set
  "nondescript — refer to expert" threshold.
- Coat-pattern re-ID (96–98%) needs patterned coats — inapplicable to
  solid-colored indigenous breeds and Murrah. Say so before judges do.

---

## Ranked idea shortlist

**Adopt now (hackathon-doable)**
1. CORN ordinal head for trait scores (`pip install coral-pytorch`,
   ~30 lines) — the loss the human-level commercial system used. [P2]
2. CowDatabase validation run → our own published-style MAE table with
   zero fieldwork. [P2, P3 assists]
3. Muzzle-print enroll/verify as tag-fraud backup (YOLO crop →
   embedding → cosine gallery). New feature, huge story. [P2 model,
   P3 gallery + endpoint]
4. LSD MobileNetV2 fine-tune on the ~4k public images → TFLite. [P2]
5. VKG differential questions when the skin classifier fires (the
   96→85% confusion stat is the justification) — extends the chatbot
   we already have. [P3]
6. Elliptical heart girth from side+rear keypoints (no 3D). [P2]
7. Confidence gating thresholds (~65–70%) on trait auto-accept — we
   have the plumbing; tune and cite. [P2/P3]
8. Gait v1: back-arch + head-bob + tracking distance → binary flag,
   with live back-arch overlay in capture UI. [P2, P1 overlay]
9. Breed suggestions as top-3 + open-set nondescript. [P2]
10. AP-10K pretraining for the keypoint model. [P2]

**Deck-only (citations, no code)**
- Accuracy-ladder slide (tape → 2D → 3D) with our position marked.
- Human inter-rater numbers vs machine repeatability.
- "Even 3 stereo cameras miss girth by 10 cm" (defends architecture).
- Multi-modal identity table (tag OCR 65% / muzzle 97% / coat N/A).
- 97,000 dead cattle in the 2022 LSD outbreak; no Indian dataset since.

**Roadmap (say, don't build)**
- Three-photo premium weight mode (MAPE 2.22% published, Jan 2026).
- Depth-capable phone tier (ARCore/LiDAR → ~3% MAPE).
- IndicCattle/Murrah parametric 3D template (hSMAL logic).
- Thermal clip-on mastitis screening.
- Field data drives: India's first phone-photo weight + LSD datasets.

---

## Datasets to grab

| Dataset | What | Where |
|---|---|---|
| CowDatabase | 103 cattle, RGB-D + 9 tape measurements | github.com/ruchaya/CowDatabase |
| AP-10K | 10k images, 54 species, 17 keypoints | NeurIPS 2021, public |
| Mendeley LSD | 1,024 skin images | data.mendeley.com/datasets/w36hpf86j2/1 |
| Roboflow LSD | 2,223 images (1,520 healthy) | via Veterinary Sciences 2024 paper |
| Kaggle Indian Bovine Breeds | 32+ breeds, 36–439 imgs/class (skewed) | Kaggle |
| Muzzle sets | 268-cattle (public) + 300-cattle few-shot | Animals 2022 / Agronomy 2021 |
| MultiCamCows2024 | 90 cows, 101k images, re-ID code | arXiv 2410.12695, CC BY 4.0 |
| coral-pytorch | CORAL/CORN losses, MIT license | pip |

## Do-not-overclaim list (judges with vet/CS backgrounds will know)

- Never claim fine-grained lameness scores — binary + confidence only.
- Never claim mastitis from RGB photos.
- Never claim metric 3D from one photo; SMAL demo = "roadmap".
- Breed classifier accuracy claims above ~90% on Indian breeds will not
  survive cross-examination; ours verifies, not predicts.
- Always quote leave-animal-out numbers, never same-animal validation.
