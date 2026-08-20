# Where this actually stands, measured

Every number here comes from a script in the repo that can be re-run. Nothing
is an estimate.

    python -m ml.eval_coverage <folder> 60             what fills, and why not
    python -m ml.eval_landmark_placement <folder> 30   do landmarks land right
    python -m ml.eval_confidence_gate <folder> 60      where the gate belongs
    python -m ml.eval_geometry <folder> 60             is a trait's geometry right

---

## The twenty contract traits

| | count | blocked by |
|---|---|---|
| measurable now | 4 | — `rump_angle`, `rear_legs_set`, `rear_legs_rear_view`, `foot_angle` |
| need a centimetre scale | 5 | an NDDB-spec tag in the frame |
| unlocked by the two-view cross-section | 1 | `heart_girth`, previously classed SMAL |
| need landmarks nobody annotated | 9 | udder, teat and rib landmarks |
| need a fat-cover model | 1 | `body_condition_score` |

So **10 of 20 are reachable** with a conformant tag close-up, and nine of the
remaining ten are an annotation job, not a modelling one.

## What the demo shows

The adoption gate keeps the baseline engine unless the ML pipeline scores at
least one contract trait. Measured over 46 photographs with no tag close-up,
the pipeline scores at least one on **16 of them — 35%**. On the rest the
baseline answers, and the baseline invents all twenty scores and a weight.

That is disclosed: `engine` is in every response, and the console now says
**DEMONSTRATION DATA — these scores were NOT measured** outright, rather than
relying on a small "Baseline" label beside twenty confident-looking numbers.

**A tag close-up flips it.** The same session posted twice:

| | engine | weight |
|---|---|---|
| without `tag_photo` | `baseline` | invented |
| with `tag_photo` | `ml-pipeline` | measured |

## Why coverage is what it is

The pose model has learned the rear half of the animal and not the front.
Audited against where a cow's anatomy has to be, over 30 photographs:

    lands correctly   hip_bone_left 82%, tail_head 71%, hock 68%,
                      pastern 67%, chest_bottom 76%*, chest_front 59%*
    does not          pin_right 0%, hook_right 9%, knee_left 11%,
                      shoulder 22-29%, back_mid 23%
                                                    (* derived, not predicted)

`chest_front` was the worst — 0 of 18, predicted behind the middle of the
animal — and is now derived from the silhouette instead, which moved
`body_length_to_height_ratio` from 0.55 to 0.93 against its 0.9–1.4 band.
The full report, with three directions worth investigating, is in
[FINDINGS-pose-landmark-placement.md](FINDINGS-pose-landmark-placement.md).

This is why the only traits that score are leg angles: the leg landmarks are
the ones that land correctly.

## Things that were silently not working

Each of these produced no error, which is why they lasted:

- **SAM2 was never installed in the server's environment**, and was absent from
  every requirements file. `segment_animal` fell back to the detection box as a
  rectangle on every image, disabling `chest_bottom`, the udder floor, and the
  weight estimator. Now pinned.
- **The digit-row scale method was dead code** — written, validated against the
  published NDDB dimensions, covered by tests, never called. Per the spec the
  button is on the REAR of the ear while the digits are printed on the FRONT,
  so a head-on photograph of a tag has no button in it.
- **`/session` had no field for the tag close-up**, so the photograph the app
  captures never left the phone and nothing measured in centimetres could work.
- **A close-up's scale was applied to side-photo pixels.** Different shots,
  different distances; this multiplies every class-C trait by their ratio, and
  produces plausible-looking centimetres rather than an error. Now carried
  across properly, using the tag as a bridge.
- **Blur was fatal below a threshold that had never been checked against what
  blur costs.** The score saturates exactly where usable and hopeless separate.
  This repo's own demo `side.jpg` scores 8.9, yields 8 usable joints, and was
  being refused before anything looked at it.

## What the system refuses, and why that is the point

- a weight, without a scale — the number a farmer might sell an animal on
- an angle whose error bar covers its whole scoring band (`foot_angle` was
  being scored on a ±25° quantity in a 25°-wide band)
- a reconstruction that is not of a side-on animal — a bovine torso is roughly
  twice as long as it is deep, and a head-on shot produces a beautifully formed
  volume that means nothing
- a breed. The exact-breed head measured 38.1% source-held-out and disables
  itself; the group head (80.2%, against a 60.7% background-only control) ships
  instead. `breed_verified: false` means NOT CHECKED, never contradicted
- a tag that is not NDDB-spec. The handwritten management tag on a real
  photograph was refused, correctly, with an actionable reason

## Tests

    258   ml/tests
      9   server suites (test_app_contract, test_session, test_demo,
          test_concurrency, test_vkg, test_rag, test_chat, test_providers,
          test_voice)
     38   server/test_demo_ui.js — the console's wording and styling

## What is needed from people, not from code

1. **One real animal photographed properly** — side, rear, and a close-up of an
   NDDB-spec tag. The current demo assets are roughly 200 px of real detail
   stretched to 800; nothing can measure from them, and no deblurring can undo
   it. This single trio would put the demo on real data end to end.
2. **Udder and teat annotation** — nine of the twenty traits, and the largest
   single block of missing coverage.
3. **The front-half landmarks**, for Person 2 — see the findings report.
4. **One line in the app**, for Person 1 — upload `tag_photo`. See
   [HANDOVER-app-breed-identity.md](HANDOVER-app-breed-identity.md).
