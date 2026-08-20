"""A scale belongs to the photograph it was measured in.

The app captures the ear tag separately, as a close-up, because the tag is a
thumbnail in the side photograph and often undetectable there. That close-up
yields an excellent centimetres-per-pixel - for the CLOSE-UP. It is a
different shot from a different distance, and its scale is typically tens of
times finer than the side photograph's, because the tag fills one frame and is
a speck in the other.

Every keypoint the trait layer measures is in SIDE-photo pixels. Handing them
the close-up's scale multiplies every class-C trait by that factor. The result
is not an obvious failure - it is a set of plausible-looking centimetres that
are wrong by a factor nobody can see, and the plausibility guards would reject
only the most absurd of them, leaving the rest to be quietly believed.

Recovering a side-photo scale from a close-up IS possible in principle: the
close-up gives this tag's true panel size in centimetres, and that same panel
measured in the side photograph would then give the side photograph's scale.
It needs the panel located in the side photo and is not built. Until it is,
the honest answer is that there is no scale, and the class-C traits say so.
"""
import cv2
import numpy as np
import pytest

from ml.config.traits import CONTRACT_TRAITS
from ml.detection import detector
from ml.pipeline import score_animal

CLASS_C = [t["trait_id"] for t in CONTRACT_TRAITS if t.get("required_scale")]


@pytest.fixture(autouse=True)
def _unload_cached_models():
    """Put the module-level model caches back as they were found.

    score_animal loads the pose and breed checkpoints and caches them in module
    globals, deliberately - the server pays that cost once. In a test run it
    leaks: the tests that check "a missing checkpoint raises an actionable
    error" point at a path that does not exist, and a cached model means they
    never reach the loader at all. They passed alone and failed after this
    file, which is the worst way to find out.
    """
    from ml.pose_features import embedding_extractor, pose_extractor
    before = (pose_extractor._model, pose_extractor._load_error,
              embedding_extractor._verifier, embedding_extractor._load_error)
    yield
    (pose_extractor._model, pose_extractor._load_error,
     embedding_extractor._verifier, embedding_extractor._load_error) = before


@pytest.fixture
def images(tmp_path):
    rng = np.random.default_rng(0)
    img = rng.normal(120, 25, (600, 800, 3)).astype(np.uint8)
    side, rear = tmp_path / "side.png", tmp_path / "rear.png"
    cv2.imwrite(str(side), img)
    cv2.imwrite(str(rear), img)

    # a close-up of a spec-accurate tag, filling its frame
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).parents[1] / "tag_intelligence"))
    from test_tag_scale_recovery import render_tag
    tag_img, _bbox = render_tag(6.0)
    tag = tmp_path / "tag.png"
    cv2.imwrite(str(tag), tag_img)
    return str(side), str(rear), str(tag)


def _stub_detector(monkeypatch):
    """One animal box, no ear tag - the field case that motivated all this."""
    monkeypatch.setattr(detector, "load_rt_detr",
                        lambda device=None: (object(), object()))
    monkeypatch.setattr(
        detector, "_run_detector",
        lambda image_bgr, model_and_processor, threshold=0.5,
        decoder_layer=None: [
            {"label": 0, "score": 0.95, "bbox": (40, 40, 760, 560)}])


def test_a_closeup_scale_never_reaches_a_body_measurement(images, monkeypatch):
    """The regression this file exists for."""
    _stub_detector(monkeypatch)
    side, rear, tag = images
    result = score_animal(side, rear, None,
                          {"animal_id": "T1", "species": "cattle"},
                          tag_img=tag)

    by_name = {t["name"]: t for t in result["traits"]}
    from ml.config.traits import CONTRACT_TRAITS as CT
    name_of = {t["trait_id"]: t["name"] for t in CT}
    for tid in CLASS_C:
        t = by_name[name_of[tid]]
        assert t["score"] is None, (
            f"{tid} was scored from a scale measured in a different "
            f"photograph - every centimetre here is wrong by the ratio of the "
            f"two shots' distances")
        assert t["measured_value"] is None


def test_the_weight_is_not_computed_from_a_borrowed_scale(images, monkeypatch):
    """Weight goes as the CUBE of the scale, so this is the worst place for it."""
    _stub_detector(monkeypatch)
    side, rear, tag = images
    result = score_animal(side, rear, None,
                          {"animal_id": "T1", "species": "cattle"},
                          tag_img=tag)
    w = result["weight_kg"]
    assert w["low"] is None and w["high"] is None


def test_the_run_still_completes_and_stays_contract_shaped(images, monkeypatch):
    """Refusing a scale must not refuse the session. Angles need no scale."""
    _stub_detector(monkeypatch)
    side, rear, tag = images
    result = score_animal(side, rear, None,
                          {"animal_id": "T1", "species": "cattle"},
                          tag_img=tag)
    assert len(result["traits"]) == 20
    assert result["animal_id"] == "T1"
    for t in result["traits"]:
        if t["score"] is None:
            assert t["not_scored_reason"], "every refusal needs a reason"


def test_without_a_closeup_nothing_changes(images, monkeypatch):
    """The guard must be specific to the close-up, not a blanket disabling of
    scale - a tag found in the side photograph is measured in the right frame
    and is perfectly usable."""
    _stub_detector(monkeypatch)
    side, rear, _tag = images
    result = score_animal(side, rear, None,
                          {"animal_id": "T1", "species": "cattle"})
    assert len(result["traits"]) == 20
