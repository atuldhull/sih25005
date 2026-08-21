# Review: Section 5.4, skin lesion detection

Reply to Tharun's progress report. Read the whole thing carefully — the short
version is that the pipeline work is solid, and there is one structural
question about the source-held-out experiment that needs answering before any
of the numbers can be interpreted, including the good ones.

Nothing here is a request to redo the training. Most of it is a request for
information I could not get from the report alone.

---

## What is clearly right

Four things in this that are easy to skip and were not skipped:

- **Cluster-based splitting** rather than a random split. Near-duplicate frames
  cannot leak from train into test, which is the most common way a livestock
  dataset flatters itself.
- **The masked BCE was verified, not asserted.** `loss = 0.3132`,
  `gradient = [-0.2689, 0, 0, 0]` demonstrates that a `-1` label contributes
  zero gradient. That is a real check and it is the right one.
- **The backbone is frozen** and the architecture was preserved, so the head is
  the only thing that moved.
- **A source-held-out run was done at all**, and the number that made the result
  look worse was reported rather than left out. That matters more than the
  score.

---

## 1. The blocking question: what does the source-held-out run actually train on?

This one has to be settled first, because it decides whether the 0.390 is a
generalization number or an artefact.

From the dataset description in the report:

| source | images | `skin_nodules` | `circular_skin_patches` |
|---|---|---|---|
| Mendeley LSD | 940 | labelled (289 pos / 651 neg) | masked `-1` |
| Roboflow cattle-smwai | 163 | masked `-1` | labelled (99 pos / 64 neg) |

The confusion matrices confirm the two heads never overlap on the test set:

    skin nodules      41 + 4 + 96 + 7  = 148 images
    circular patches   9 + 0 + 12 + 0  =  21 images
                                        ----
                                         169  = the whole test set, no overlap

So each head is only ever evaluated on its own source.

**Now apply the masking rule to the held-out runs.** If Mendeley is held out of
training, the `skin_nodules` head trains only on Roboflow — where every
`skin_nodules` label is `-1` and contributes zero gradient, by the design that
was verified above. That head would then be evaluated at its initialisation.
An F1 of 0.390 with precision 0.471, against a base rate of 48/148 = 0.324,
is close to what that would look like.

But the same reasoning makes the other result impossible. If Roboflow is held
out, the `circular_skin_patches` head gets no gradient either — and an
untrained head that simply fires on everything would score precision 9/21 =
0.43, recall 1.0, **F1 0.60**, not 0.889.

Both numbers cannot come from the protocol I am inferring. So one of these is
true and I cannot tell which from the report:

1. The held-out runs still train on the held-out source's *other* labels, and
   I have the protocol wrong.
2. "Held out" means held out of the test set rather than out of training.
3. One of the two figures is measuring something other than generalization.

**Please send the actual training/eval script for the source-held-out runs, or
just describe what goes into the training set in each case.** Everything below
depends on the answer.

If it turns out that holding out a source removes all supervision for that
head, then a true source-held-out experiment is **not possible on this dataset
as built** — each symptom's labels live entirely inside one source. That is not
a coding mistake, it is a data-collection consequence, and the fix would be
skin-nodule images from a second source rather than any change to the model.

---

## 2. The background-only control did not complete

The report says this plainly and it is the right call. The concern is the table
immediately above the note:

| symptom | normal recall | background recall | gap |
|---|---|---|---|
| skin nodules | 0.854 | 0.000 | 0.854 |
| circular skin patches | 1.000 | 0.000 | 1.000 |

That gap column should be blank, not 0.854 and 1.000.

A run that errored and produced no predictions yields **exactly the same
0.000 recall** as a model that correctly refused every background image. The
two are indistinguishable in that number, so it cannot be read as the control
passing. As written the table looks like the strongest evidence in the report,
and it is currently the weakest.

This is the experiment that matters most. It is what caught an earlier bovine
model scoring 97.9% by learning the farm rather than the animal, and it is the
single check that a lesion detector most needs, because lesion datasets tend to
be photographed differently from healthy ones.

**Needed:** the RT-DETR checkpoint warning resolved and the control re-run, and
the gap column left empty until it produces real metrics.

---

## 3. F1 1.000 is nine positive examples

Wilson 95% intervals on the reported counts:

    skin nodules      recall     41/48 = 0.854    CI  0.728 - 0.928
                      precision  41/45 = 0.911    CI  0.793 - 0.965

    circular patches  recall      9/9  = 1.000    CI  0.701 - 1.000
                      precision   9/9  = 1.000    CI  0.701 - 1.000
    source-held-out   recall      8/9  = 0.889    CI  ~0.57 - 0.98

A perfect score whose lower bound is 0.70 is not a perfect detector. The skin
nodules intervals are respectable; the circular patches ones are too wide to
act on.

**Suggestion:** report every metric with its n and its interval in
`evaluation_results.json`. It costs nothing and it stops "1.000" being quoted
at a judge, which will not survive the follow-up question.

---

## 4. What the numbers mean for wiring this into the app

We already have a rule for this in the project, and it settles the question
without anyone having to argue about it:

> The exact-breed head measured **38.1%** source-held-out and **disables
> itself**. The group head, at **80.2%** against a 60.7% background-only
> control, ships with a `group_reliable` flag so the app can show it as a hint
> rather than a finding.

Applying that same standard:

| head | source-held-out | comparable to | implication |
|---|---|---|---|
| `skin_nodules` | 0.390 | exact-breed head (0.381) | should disable itself |
| `circular_skin_patches` | 0.889 | group head (0.802) | could ship as a hint — but n = 9 |

Both of those are provisional until question 1 is answered.

### Why this is not a formality

`symptom_vector` is not a display field. It flows into
`vkg.estimate_risks` → `needs_escalation` → the veterinary officer's alert
feed. An entry there is a request for a person to drive to a farm.

We spent this week removing **fabricated** `skin_nodules` findings at
confidence 0.82 that were writing Lumpy Skin Disease alerts into that feed
from photographs of nothing at all. Wiring in a detector whose out-of-source
precision is 0.471 would put a real model behind the same path and send a vet
to the wrong farm roughly half the times it fires.

A detector that disables itself and says so is worth more to this project than
one that answers and is wrong. That is already how the breed head behaves, and
it is the thing judges have responded to.

---

## 5. What to send back

Roughly in order of how much it unblocks:

1. **The source-held-out training/eval script**, or a plain description of what
   is in the training set for each held-out run. This is the blocker.
2. **The background control re-run** with the RT-DETR checkpoint issue fixed —
   and the gap column blank until then.
3. **`evaluation_results.json` with n and confidence intervals** on every
   metric, and the source-held-out figures presented as the headline rather
   than the appendix.
4. **Push the work.** Nothing is on `ml-dev`, `main` or `integration`, and
   there are no `skin_lesion` files on any branch. Needed to integrate:
   - `checkpoints/skin_lesion/best_model.pth`
   - `checkpoints/skin_lesion/split_info.json`
   - `eval_results/evaluation_results.json`
   - `labels.csv` and the dataset-preparation script
   - the inference entry point, and the exact output shape it produces
5. **The per-symptom decision threshold** and how it was chosen. Nothing in the
   report says where the operating point came from, and precision/recall move a
   long way with it. If it was left at 0.5, say so — that is a defensible
   default, but it should be a stated choice rather than an unexamined one.
6. **Confirmation of what `ticks_visible` and `wounds` do at inference.** They
   are masked in training, so the head has never received a gradient for them.
   If those two logits are still emitted, they have to be dropped downstream —
   an untrained logit is not a low-confidence prediction, it is noise, and the
   app currently has no way to tell the difference.

---

## 6. What is being built on the integration side meanwhile

So that this drops straight in whenever the numbers land:

- A **`symptom_reliable` flag per symptom**, mirroring `group_reliable`
  exactly. A symptom below its measured bar is carried as a hint and is
  **excluded from `needs_escalation`**, so it never reaches the vet feed on its
  own.
- The scorecard's health card wording changed from "NOT SCREENED" to naming
  what actually screened and how well it was measured — for example *"screened
  for skin nodules by a model measured at F1 0.39 on photographs from a source
  it had never seen"*. Whatever the final figures are, they get shown, not
  summarised away.
- The inference call itself, behind the same import-and-validate gate the pose
  pipeline uses, so a missing or broken checkpoint degrades to "not screened"
  rather than to an exception or, worse, to a fabricated finding.

None of that needs the model to exist. It needs the output shape, which is
item 4 above.

---

## In short

The training pipeline is in good shape and the data hygiene is better than
most of what gets submitted at this level. Two things stand between it and
being integrable, and neither is about the model:

- the source-held-out protocol needs clarifying, because as described it may
  be measuring an untrained head
- the background control needs to actually run

Once those land, the honest path is almost certainly to ship
`circular_skin_patches` as a hint, have `skin_nodules` disable itself the way
the breed head does, and say so on the scorecard in plain words.
