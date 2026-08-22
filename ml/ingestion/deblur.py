"""Sharpness assessment and honest blur recovery.

WHY THIS IS NOT A DEBLURRING MODEL
A learned deblurrer (DeblurGAN, NAFNet, Restormer) does not recover detail -
it SYNTHESISES plausible detail. On a photograph of a face that is a cosmetic
improvement. Here the output is fed to a pose model that places anatomical
landmarks, and then to a measurement layer that turns those landmarks into
centimetres a farmer is told to act on. Invented edges become invented joints
become invented measurements, and nothing downstream can tell the difference,
because a hallucinated edge is sharper and more confident-looking than a real
soft one.

So this module amplifies edges that are ALREADY THERE and refuses when there
is nothing to amplify. Unsharp masking is

    out = img + amount * (img - blur(img))

a linear high-frequency boost. It cannot add structure the sensor did not
record: where the image is flat the subtraction yields zero and nothing is
added. That property is the entire reason it was chosen over a model.

THE DISTINCTION THAT MATTERS
Two very different failures both read as "blurry" on a Laplacian score:

  recoverable   real detail is present but the contrast carrying it is low -
                slight defocus, haze, a soft lens, mild camera shake.
                Boosting helps, and it helps the pose model measurably.

  no_detail     the high frequencies are absent, not attenuated - an upscaled
                thumbnail, a heavy JPEG, a hard motion smear. Nothing to
                amplify. Sharpening then produces contrast without
                information, which is worse than the blur because it looks
                fixed.

Telling them apart is a resolution question, not a contrast question, so it is
measured by round-tripping the image through a downscale: if throwing away
15/16 of the pixels changes almost nothing, then almost nothing was being
carried by those pixels in the first place. Unlike a contrast measurement it
cannot be fooled by simply raising contrast.

Measured on this repo's own demo assets:

    side.jpg  836x820    4x round-trip RMSE 1.74   -> ~209 px of real detail
    rear.jpg  812x648    4x round-trip RMSE 1.33   -> ~203 px of real detail
    a real phone photo   4x round-trip RMSE 4.70   -> detail throughout

The demo images are roughly 200 px of information stretched to 800. No model
can undo that, and this module says so instead of pretending.
"""
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from ml.detection.detector import DetectionBackendError


def _get_cv2():
    try:
        import cv2
        return cv2
    except ImportError as exc:
        raise DetectionBackendError(
            "'opencv-python' (cv2) is not installed. Install it with "
            "`pip install opencv-python` to assess image sharpness."
        ) from exc


# Laplacian variance is compared at a fixed size - see quality_validation for
# the 12x measurement swing that motivated it. Same constant, same reason.
WORK_LONG_SIDE = 1024

# At or above this the image is left alone.
SHARP_ENOUGH = 100.0

# A 4x downscale round trip on an image with genuine detail changes it
# substantially. Below this RMSE the pixels carry nothing, the nominal
# resolution is a lie, and sharpening would be invention rather than recovery.
DETAIL_ROUND_TRIP_SCALE = 4
MIN_ROUND_TRIP_RMSE = 3.0

# The radius is deliberately small. A wide radius produces the halo that reads
# as "sharpened" to a human but shifts where an edge appears to be, and an
# edge that moves is a joint that moves.
UNSHARP_SIGMA = 1.2
UNSHARP_AMOUNT = 1.4

# Refuse to claim a recovery that did not happen. If boosting does not raise
# the score by this factor the image goes forward unmodified and still soft -
# better an honestly soft image than a sharpened one that gained nothing.
MIN_USEFUL_GAIN = 1.25

# Sharpening is a real intervention and downstream confidence must reflect it.
RECOVERED_CONFIDENCE_PENALTY = 0.85


@dataclass
class SharpnessReport:
    """What we know about an image's sharpness, and what can be done.

    verdict is one of:
        "sharp"        usable as-is
        "recoverable"  soft, but real detail is present under the softness
        "no_detail"    the resolution is nominal only; refuse
    """
    score: float
    verdict: str
    detail_rmse: float
    effective_long_side: Optional[int] = None
    region: str = "frame"

    @property
    def usable(self) -> bool:
        return self.verdict in ("sharp", "recoverable")


def _to_work_size(image):
    cv2 = _get_cv2()
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= WORK_LONG_SIDE:
        return image
    s = WORK_LONG_SIDE / float(longest)
    return cv2.resize(image, (max(1, int(w * s)), max(1, int(h * s))),
                      interpolation=cv2.INTER_AREA)


def laplacian_variance(image) -> float:
    """Blur score at the fixed working size. Lower means softer."""
    cv2 = _get_cv2()
    work = _to_work_size(image)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def detail_round_trip_rmse(image, scale: int = DETAIL_ROUND_TRIP_SCALE) -> float:
    """How much the image loses when downscaled and put back.

    This separates "soft" from "empty". An image carrying real detail is
    changed a lot by discarding 15/16 of its pixels; an upscaled thumbnail is
    barely changed at all, because the information was never there.
    """
    cv2 = _get_cv2()
    import numpy as np
    work = _to_work_size(image)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    small = cv2.resize(gray, (max(1, w // scale), max(1, h // scale)),
                       interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return float(np.sqrt(np.mean((gray - back) ** 2)))


def _crop_to_region(image, bbox: Optional[Sequence[float]]):
    """Restrict to the animal. Blur on the BACKGROUND is not a defect.

    A phone at a wide aperture throws the field behind the animal out of
    focus, and a whole-frame score reads that as a bad photograph. What the
    pose model needs is detail on the animal, so that is what gets measured.
    """
    if bbox is None:
        return image, "frame"
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 32 or y2 - y1 < 32:
        return image, "frame"
    return image[y1:y2, x1:x2], "animal"


def assess(image, animal_bbox: Optional[Sequence[float]] = None
           ) -> SharpnessReport:
    """Classify an image as sharp, recoverable, or informationally empty."""
    region_img, region = _crop_to_region(image, animal_bbox)
    score = laplacian_variance(region_img)
    rmse = detail_round_trip_rmse(region_img)

    h, w = region_img.shape[:2]
    if score >= SHARP_ENOUGH:
        verdict = "sharp"
    elif rmse < MIN_ROUND_TRIP_RMSE:
        verdict = "no_detail"
    else:
        verdict = "recoverable"

    effective = None
    if verdict == "no_detail":
        # Rough, and labelled as rough: a round trip that left the image
        # unchanged puts the real detail at about 1/scale of nominal.
        effective = int(max(h, w) / float(DETAIL_ROUND_TRIP_SCALE))

    return SharpnessReport(score=score, verdict=verdict, detail_rmse=rmse,
                           effective_long_side=effective, region=region)


def unsharp(image, sigma: float = UNSHARP_SIGMA,
            amount: float = UNSHARP_AMOUNT):
    """out = img + amount * (img - blur(img)).

    Every pixel of the result is a linear combination of pixels that were
    actually recorded. Where the image is flat, img - blur(img) is zero and
    nothing is added. That is the guarantee a learned model cannot give.
    """
    cv2 = _get_cv2()
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


def recover(image, animal_bbox: Optional[Sequence[float]] = None
            ) -> Tuple[object, SharpnessReport, bool]:
    """Try to make a soft image usable. Returns (image, report, was_changed).

    Refuses in both directions: a sharp image is left alone, and an image with
    no detail to boost is returned untouched with its "no_detail" verdict
    intact so the caller can reject it. The returned report always describes
    the image being RETURNED, so a caller cannot accidentally record a
    pre-recovery score against a post-recovery image.
    """
    before = assess(image, animal_bbox)
    if before.verdict != "recoverable":
        return image, before, False

    boosted = unsharp(image)
    after = assess(boosted, animal_bbox)
    if after.score < before.score * MIN_USEFUL_GAIN:
        # Amplifying gained nothing real. Keep the original rather than ship
        # an image altered for no measurable benefit.
        return image, before, False
    return boosted, after, True
