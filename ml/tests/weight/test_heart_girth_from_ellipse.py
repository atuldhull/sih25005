"""Heart girth without a SMAL mesh.

The trait was classed SMAL and refused, for a good reason recorded in
ml/config/traits.py: a flat 2D chord is not a circumference, and feeding chest
depth straight into Schaeffer's formula was a real bug once. Refusing was the
right answer to that.

The ellipse model answers it by the route that note was really asking for. The
chest station is a CLOSED cross-section assembled from two photographs - depth
from the side, width from the rear - so its perimeter is a circumference, not
a chord. That is a 3D reconstruction; it is simply not a SMAL one.

What it is not is exact. A real chest is flatter over the back than an ellipse,
and the scale comes from the ear tag with its own error, which is what the
uncertainty on the replacement measurement is for.
"""
import math

import pytest

from ml.common.schemas import MeasurementResult
from ml.pipeline import (
    HEART_GIRTH_MODEL_ERROR,
    HEART_GIRTH_SCALE_ERROR,
    _with_heart_girth,
)
from ml.scoring.scorer import score_trait
from ml.weight.volume_3d import VolumeWeight


def _smal_refusal():
    return MeasurementResult(
        trait_id="heart_girth", trait_class="SMAL", value=None, unit="cm",
        confidence=0.0, flags=["not_measurable", "requires_3d_model"])


def _vol(girth):
    return VolumeWeight(low_kg=400.0, high_kg=450.0, method="m", cross_check="c",
                        heart_girth_cm=girth)


def test_the_smal_refusal_is_replaced_by_a_real_measurement():
    out = _with_heart_girth([_smal_refusal()], _vol(186.0))
    hg = {m.trait_id: m for m in out}["heart_girth"]
    assert hg.value == pytest.approx(186.0)
    assert hg.flags == [], "it is measured now, not refused"
    assert hg.trait_class == "C", "a scaled distance, no longer a 3D-model trait"


def test_it_scores_where_the_smal_refusal_could_not():
    before = score_trait(_smal_refusal(), "cattle")
    assert before.score_1_9 is None

    out = _with_heart_girth([_smal_refusal()], _vol(186.0))
    after = score_trait({m.trait_id: m for m in out}["heart_girth"], "cattle")
    assert after.score_1_9 is not None, "cattle heart girth band is 130-220 cm"


def test_the_uncertainty_combines_the_model_and_the_scale():
    out = _with_heart_girth([_smal_refusal()], _vol(200.0))
    hg = {m.trait_id: m for m in out}["heart_girth"]
    expected = 200.0 * math.sqrt(HEART_GIRTH_MODEL_ERROR ** 2
                                 + HEART_GIRTH_SCALE_ERROR ** 2)
    assert hg.uncertainty == pytest.approx(expected, rel=1e-9)
    assert hg.uncertainty > 0, "an approximated circumference is never exact"


def test_an_absurd_girth_is_still_refused_downstream():
    """The replacement is not a licence to report anything. A cattle band of
    130-220 cm still applies, and a reconstruction that lands outside it is
    telling us the two photographs were not what the model needed."""
    out = _with_heart_girth([_smal_refusal()], _vol(600.0))
    hg = {m.trait_id: m for m in out}["heart_girth"]
    assert score_trait(hg, "cattle").score_1_9 is None


def test_other_traits_are_untouched():
    others = [
        MeasurementResult(trait_id="stature", trait_class="C", value=None,
                          unit="cm", confidence=0.0, flags=["no_scale"]),
        _smal_refusal(),
        MeasurementResult(trait_id="foot_angle", trait_class="A", value=52.0,
                          unit="degrees", confidence=0.7, flags=[]),
    ]
    out = _with_heart_girth(others, _vol(186.0))
    by = {m.trait_id: m for m in out}
    assert len(out) == 3
    assert by["stature"].value is None and by["stature"].flags == ["no_scale"]
    assert by["foot_angle"].value == 52.0


def test_body_condition_score_stays_a_SMAL_trait():
    """Only heart_girth is answerable this way. BCS is a fat-cover judgement,
    not a cross-section, and the ellipse model says nothing about it."""
    bcs = MeasurementResult(trait_id="body_condition_score", trait_class="SMAL",
                            value=None, unit="score", confidence=0.0,
                            flags=["not_measurable", "requires_3d_model"])
    out = _with_heart_girth([bcs], _vol(186.0))
    assert {m.trait_id: m for m in out}["body_condition_score"].value is None
