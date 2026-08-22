# Two findings about ear tags on a real photograph

Tested against `Sahiwal-cow-ear-tagging-scaled.jpg` — a 2560×1700 press-quality
side view of a Sahiwal cow at a show, with a yellow ear tag plainly legible in
the near ear.

Nothing here has been changed. The detector belongs to Person 2 and is not
being retrained or edited; this is a report.

---

## 1. The detector does not see a tag that is plainly there

`detect_ear_tag()` returns `None` on this image. So does the raw detector at
decoder layer 1 with the threshold dropped to **0.02** — one detection comes
back, and it is the animal. There is no low-confidence ear-tag box being
filtered out; the class is not firing at all.

The tag is not hard to see. Cropped from the full-resolution frame it is
roughly 80×110 px of saturated yellow against brown hide, unoccluded, in
focus, and near the centre of the frame.

Reported measurement for this class was AP@0.5 = 0.954 on the held-out test
split, which is why layer 1 is used at all. That number and this result are
both true, which suggests the test split and a press photograph of a show
animal do not look alike to the model. Worth a look at what the training tags
have in common before trusting the class in the field.

Reproduce:

```python
from ml.detection.detector import detect_animal, detect_ear_tag
a = detect_animal(F)
print(detect_ear_tag(F, a.bbox))          # -> None
```

## 2. Real farm tags are often not NDDB tags, and the ruler is right to refuse

Locating the tag by hand and handing its box straight to `measure_scale()`
gives a refusal, and the refusal is correct:

```
TagScaleRefused: Ear tag found but its round button could not be measured -
retake the photo with the tag facing the camera
```

The tag on this animal is a plain management tag: a yellow panel with **S-102**
written on it by hand. No barcode, no 10 mm and 18 mm digit rows, no round
button visible from the front — the NDDB button sits on the REAR of the ear.
Both scale methods in `tag_ruler.py` key off features this tag does not have,
so both correctly decline rather than inventing a centimetres-per-pixel from a
panel whose true size is unknown and supplier-dependent.

### What follows from that

Scale is what gates the eleven class-C traits and the entire weight estimate.
If field photographs commonly carry non-conformant tags, then a single side
photograph will commonly yield no scale, and those traits will be honestly
unavailable rather than wrong — but unavailable all the same.

The app already has the right answer to this in
`lib/screens/capture/scan_tag_screen.dart`: a dedicated close-up of the tag,
which the pipeline accepts as `tag_img` and uses in preference to the side
photograph. Two things follow for the demo and for the field:

- the close-up must show an **NDDB-spec** tag — barcode row, two digit rows,
  or the round button — because that is what the ruler measures;
- when it does not, the result says `no_scale` and the class-C traits carry
  `not_scored_reason`, which is the designed behaviour and should be shown as
  such, not read as a failure.

A third option, if non-conformant tags turn out to be the norm, is to let the
farmer photograph the tag beside any object of known size. That is a product
decision rather than a modelling one, and nothing has been built for it.
