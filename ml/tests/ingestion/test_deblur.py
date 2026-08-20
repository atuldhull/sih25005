"""Blur: what gets recovered, what gets refused, and what never gets refused for.

The behaviour under test is mostly restraint. Sharpening an image that has no
detail left produces contrast without information, and the result looks BETTER
to a human while being worse to measure from - so most of these tests check
that nothing happened.
"""
import numpy as np
import pytest

from ml.ingestion import deblur
from ml.ingestion.quality_validation import is_fatal, validate_image


def textured(w=900, h=800, seed=0, grain=34):
    """An image that behaves like a photograph under these measurements.

    Uniform random noise will not do. Its spectrum is flat, so it survives
    downscaling in a way no photograph does, and a fixture built from it reads
    as "full of detail" no matter what is done to it. A real photograph is
    mostly low-frequency scene structure, plus edges, plus fine grain - hair,
    grass, skin - and it is that grain the blur measurements are looking for.
    """
    import cv2
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(
        rng.integers(0, 255, (h, w, 3)).astype(np.float32), (0, 0), 9)
    base = cv2.normalize(base, None, 30, 225, cv2.NORM_MINMAX)
    for i in range(14):
        x, y = int(rng.integers(0, w - 90)), int(rng.integers(0, h - 90))
        c = tuple(float(v) for v in rng.integers(0, 255, 3))
        if i % 2:
            cv2.rectangle(base, (x, y), (x + 80, y + 70), c, -1)
        else:
            cv2.circle(base, (x + 40, y + 40), 34, c, -1)
    base += rng.normal(0, grain, base.shape)
    return np.clip(base, 0, 255).astype(np.uint8)


def upscaled_thumbnail(w=900, h=800, factor=6):
    """Real detail at 1/factor of the nominal size, stretched back up.

    This is what the repo's own demo assets turned out to be: roughly 200 px
    of information presented as 800. Informationally identical to a heavy
    blur, and equally unrecoverable.
    """
    import cv2
    full = textured(w, h, seed=1)
    small = cv2.resize(full, (w // factor, h // factor),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


# --- the recoverable / unrecoverable distinction --------------------------

def test_a_sharp_image_is_left_completely_alone():
    img = textured()
    out, report, changed = deblur.recover(img)
    assert report.verdict == "sharp"
    assert changed is False
    assert out is img, "a sharp image must not be touched at all"


def test_an_upscaled_thumbnail_is_refused_not_sharpened():
    """The failure this module exists to prevent.

    There is nothing to amplify, so sharpening would manufacture edges. The
    pose model would then place joints on those edges and the measurement
    layer would report centimetres derived from them, with no way for anything
    downstream to know the detail was invented.
    """
    img = upscaled_thumbnail()
    out, report, changed = deblur.recover(img)
    assert report.verdict == "no_detail"
    assert changed is False
    assert np.array_equal(out, img), "no pixels may be altered"


def test_the_round_trip_measure_separates_soft_from_empty():
    """A contrast measure cannot tell these apart; a resolution measure can.

    Both of these score low on Laplacian variance. Only one of them still has
    something under the softness worth boosting.
    """
    import cv2
    soft = cv2.GaussianBlur(textured(), (0, 0), 1.0)
    empty = upscaled_thumbnail()
    assert deblur.laplacian_variance(soft) < deblur.SHARP_ENOUGH
    assert deblur.laplacian_variance(empty) < deblur.SHARP_ENOUGH
    assert deblur.detail_round_trip_rmse(soft) > deblur.MIN_ROUND_TRIP_RMSE
    assert deblur.detail_round_trip_rmse(empty) < deblur.MIN_ROUND_TRIP_RMSE


def test_a_recoverable_image_actually_gets_boosted():
    """The other half: restraint is only a virtue if it is not total."""
    import cv2
    soft = cv2.GaussianBlur(textured(), (0, 0), 1.0)
    before = deblur.assess(soft)
    assert before.verdict == "recoverable"
    out, after, changed = deblur.recover(soft)
    assert changed is True
    assert after.score >= before.score * deblur.MIN_USEFUL_GAIN


def test_recovery_reports_the_image_it_actually_returns():
    """The report must never describe the pre-recovery image.

    A caller that recorded the old score against the new pixels would log a
    quality figure for an image that no longer exists.
    """
    import cv2
    soft = cv2.GaussianBlur(textured(), (0, 0), 1.0)
    out, report, changed = deblur.recover(soft)
    if changed:
        assert report.score == pytest.approx(deblur.laplacian_variance(out), rel=1e-6)
    else:
        assert report.score == pytest.approx(deblur.laplacian_variance(soft), rel=1e-6)


# --- unsharp masking invents nothing --------------------------------------

def test_unsharp_leaves_a_flat_region_flat():
    """img - blur(img) is zero where the image is flat, so nothing is added.

    This is the property that a learned deblurrer cannot offer, and the reason
    one is not used here.
    """
    flat = np.full((400, 400, 3), 128, np.uint8)
    out = deblur.unsharp(flat)
    assert np.abs(out.astype(int) - 128).max() <= 1


def test_unsharp_does_not_move_an_edge():
    """An edge that shifts is a joint that shifts.

    The boost may overshoot on either side of a step, but the crossing point
    itself must stay put, or every measurement spanning that edge moves with it.
    """
    img = np.zeros((200, 200, 3), np.uint8)
    img[:, 100:] = 255
    out = deblur.unsharp(img)
    col = out[100, :, 0].astype(int)
    crossing = int(np.argmax(col >= 128))
    assert abs(crossing - 100) <= 1, f"edge moved to {crossing}"


# --- blur is recorded, not refused ----------------------------------------

def test_blur_is_never_a_fatal_defect():
    """Measured: the pose model still produces usable joints on images scoring
    far below the old threshold of 100, and the blur score saturates in exactly
    the range where usable and hopeless diverge - so it cannot be the gate."""
    assert not is_fatal(["blur_too_high"])
    assert not is_fatal(["underexposed"])
    assert not is_fatal(["overexposed"])


def test_defects_that_genuinely_prevent_work_are_still_fatal():
    assert is_fatal(["file_not_found"])
    assert is_fatal(["unreadable_image"])
    assert is_fatal(["resolution_too_low"])
    assert is_fatal(["bad_aspect_ratio"])
    assert is_fatal(["blur_too_high", "resolution_too_low"]), "one fatal is enough"


def test_a_soft_image_reports_the_defect_while_staying_usable(tmp_path):
    """Both halves matter: the reason is recorded so confidence can be reduced
    and the farmer told, AND the pipeline is still allowed to try."""
    import cv2
    p = tmp_path / "soft.jpg"
    cv2.imwrite(str(p), cv2.GaussianBlur(textured(), (0, 0), 6))
    q = validate_image(str(p))
    assert "blur_too_high" in q["reasons"]
    assert q["passed"] is False, "the defect must be recorded"
    assert q["fatal"] is False, "but it must not stop the pipeline"


def test_a_missing_file_is_fatal_and_says_so(tmp_path):
    q = validate_image(str(tmp_path / "nothing.jpg"))
    assert q["fatal"] is True
    assert q["reasons"] == ["file_not_found"]
