"""Tag scale adapter: what happens when there is no scale.

A refused scale is a normal outcome, not an error. Class-A angle traits need
no scale at all, so the run must degrade to those rather than either crashing
or inventing centimetres.
"""
from ml.tag_intelligence import tag_reader


def test_no_tag_result_means_no_scale():
    assert tag_reader.scale_factor_from(None) == (None, 0.0)


def test_empty_tag_result_means_no_scale():
    assert tag_reader.scale_factor_from({}) == (None, 0.0)
    assert tag_reader.scale_factor_from(
        {"identity": None, "scale": None}) == (None, 0.0)


def test_scale_is_passed_through_unchanged():
    """cm_per_px goes straight to measure_all_traits as scale_factor.

    Measurement multiplies a pixel distance by it. If this ever converted
    units the traits would be wrong by orders of magnitude while still
    looking like plausible code, so the pass-through is asserted.
    """
    tag = {"scale": {"cm_per_px": 0.0421, "confidence": 0.77}}
    factor, conf = tag_reader.scale_factor_from(tag)
    assert factor == 0.0421
    assert conf == 0.77


def test_refusal_carries_a_reason_for_the_farmer():
    exc = tag_reader.TagScaleRefused(
        "Ear tag found but its round button could not be measured.",
        detail="no contour passed the circularity checks")
    assert "button" in exc.reason
    assert exc.detail


def test_read_tag_never_claims_an_identity():
    """OCR is not implemented. identity must stay None.

    Inventing an ear-tag number would be far worse than admitting we cannot
    read one - it would attach a measurement to the wrong animal.
    """
    import inspect
    src = inspect.getsource(tag_reader.read_tag)
    assert '"identity": None' in src
