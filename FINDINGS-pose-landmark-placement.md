# The pose model has learned the rear half of the animal, not the front

For Person 2. Nothing has been retrained or changed — `best.pt`, the dataset
and the training configuration are untouched. This is a report with a
reproduction.

It explains something that had no other explanation: why the only traits that
ever score are leg traits, and why every measurement involving the front of
the animal comes out roughly half the size it should be.

---

## What you can see directly

Render the predictions over any clear side view:

```python
from ml.detection.detector import detect_animal
from ml.pose_features.pose_extractor import extract_keypoints
a = detect_animal(F); k = extract_keypoints(F, a.bbox)
# draw k over the image
```

On a 2560×1700 press photograph of a Sahiwal cow — square-on, unoccluded, sharp
— `tail_head` lands on the tail head, and the hocks, pasterns and hooves land
on the legs. But `chest_front`, `shoulder_left`, `shoulder_right`, `hook_right`,
`chest_width_left` and `pin_left` are all clustered together in the middle of
the barrel, nowhere near the anatomy they are named for. The brisket point sits
behind the middle of the animal.

## Measured over 30 photographs

`ml/eval_landmark_placement.py` checks something PCK cannot. PCK measures
distance from an annotated point; it cannot tell you the predictions are
landing in the wrong PLACE on the animal. This instead asks whether each
landmark falls in the region of the silhouette where that part of a cow has to
be — which needs no ground truth, because it follows from the animal's own
outline.

Coordinates are silhouette-relative and oriented: **x = 0 at the head, 1 at the
tail; y = 0 on the topline, 1 on the ground.**

```
python -m ml.eval_landmark_placement <folder> 30
```

| landmark | in place | median x | expected x |
|---|---|---|---|
| hip_bone_left | 82% | 0.84 | 0.65–0.95 |
| tail_head | 71% | 0.91 | 0.80–1.00 |
| hock_left / hock_right | 68% | 0.89 / 0.86 | 0.60–0.95 |
| pastern_left / right | 67% / 60% | 0.86 | anywhere, low |
| hip_bone_right | 60% | 0.87 | 0.65–0.95 |
| **withers** | 35% | 0.57 | 0.20–0.55 |
| **shoulder_right / left** | 29% / 22% | 0.47 / 0.49 | 0.10–0.40 |
| **knee_right / left** | 27% / 11% | 0.43 / 0.49 | 0.05–0.40 |
| **hook_right** | 9% | 0.49 | 0.65–0.95 |
| **back_mid** | 0% | 0.78 | 0.35–0.70 |
| **pin_right** | 0% | 0.45 | 0.78–1.00 |
| **chest_front** | 0% | 0.60 | 0.00–0.30 |

The pattern is one-directional and hard to read any other way: **landmarks on
the front of the animal are pulled backwards toward its centre.** The brisket,
the shoulders and the carpus all sit near x ≈ 0.5. The rear landmarks — tail
head, hip bones, hocks, pasterns — are mostly where they belong.

A caveat on reading this: the expected boxes are my judgement of where a cow's
anatomy is, drawn deliberately loose. Treat the marginal rows (withers,
shoulders, hooves) as suggestive. `chest_front` at 0% with a median of 0.60
against a bound of 0.30 is not marginal — the brisket is being predicted behind
the middle of the animal, on 18 of 18 photographs where it was confident enough
to use at all.

## What it costs downstream

- **`chest_front` → `pin_left` spans 28% of the animal's own detection box.**
  It should span roughly 70%. This holds on the press photograph as well as on
  scraped ones, so it is not an image-quality effect.
- `body_length_to_height_ratio` therefore reads **0.55** against a band of
  0.9–1.4, with an uncertainty of only ±0.03. A cow cannot be half as long as
  it is tall. Substituting a silhouette-derived brisket changed it by 0.01, so
  `pin_left` is displaced too — both landmarks sit mid-body.
- With a scale supplied, `stature` comes out at **82 cm** and `body_length` at
  **73 cm** on an animal whose barrel alone is 150 cm.
- `pin_left` and `knee_left` are predicted at the SAME pixel on **41 of 50**
  photographs — never through low confidence, always by collapsing together.
  That is consistent with both being drawn to the same mid-body region.
- Of the twenty contract traits, the ones that score are the leg angles. That
  is not a coincidence; the leg landmarks are the ones that land correctly.

## What might be worth checking

Offered as directions, not conclusions — this is your model and you have the
training data.

1. **Whether the held-out split resembles a field photograph.** The reported
   PCK@0.02 is 61.7% overall, and `chest_front` is quoted at 0.429. A joint
   landing at x ≈ 0.6 when it belongs at x ≤ 0.3 is out by far more than 2% of
   the box side, so either the split is unrepresentative or something between
   training and inference differs.
2. **Annotation consistency on the front half.** The brisket, the point of
   shoulder and the carpus are harder to place by eye than a tail head or a
   hock. If different annotators put them in different places, the model would
   learn their average — which is roughly the middle of the animal, and is
   exactly what the numbers show.
3. **Whether these joints were annotated at all in some images.** A landmark
   marked absent in training rather than skipped would pull predictions toward
   the mean position.

The interim handling on this side: measurement now carries an uncertainty
derived from the geometry, scoring refuses when that uncertainty covers the
trait's whole band, and a value that is precise but anatomically impossible is
reported as landmarks-probably-wrong rather than as an unusual animal. So none
of this produces a wrong number for a farmer — it produces fewer numbers.
