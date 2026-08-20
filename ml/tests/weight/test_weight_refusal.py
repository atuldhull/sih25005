"""Regression test for honest weight refusal end-to-end (Rev 2 audit item 5 /
implementation item 3).

Once Heart Girth reliably refuses (SMAL branch, see test_smal_refusal.py),
estimate_weight() must return an honest None/None rather than fabricating a
weight from a missing/None girth value - and _weight_to_contract() must carry
that through unchanged into the contract's {low: None, high: None} shape.
Both were already correct; this test exists to catch a future regression.
"""

from ml.common.schemas import MeasurementResult
from ml.explainability.result_builder import _weight_to_contract
from ml.weight.estimator import estimate_weight


def test_weight_refuses_when_girth_is_none():
    girth = MeasurementResult(
        trait_id="heart_girth", trait_class="SMAL", value=None, unit="cm",
        confidence=0.0, flags=["not_measurable", "requires_3d_model"],
    )
    length = MeasurementResult(
        trait_id="body_length", trait_class="C", value=180.0, unit="cm",
        confidence=0.9, flags=[],
    )
    result = estimate_weight([girth, length])
    assert result.estimate_kg is None
    assert result.range_kg == (None, None)

    contract = _weight_to_contract(result)
    assert contract["low"] is None
    assert contract["high"] is None


def test_weight_refuses_with_no_measurements_at_all():
    result = estimate_weight([])
    assert result.estimate_kg is None
    assert result.range_kg == (None, None)
