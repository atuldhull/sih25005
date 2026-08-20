"""Regression test for overlay_points integer coordinates (Rev 2 audit item 3 /
implementation item 11).

overlay_points were emitted as floats (list(p) on a raw (x, y) tuple), but
the Flutter app expects integer pixel coordinates. result_builder.py must
round and cast to int at the contract boundary without changing the overall
shape (still a list of [x, y] pairs).
"""

from ml.explainability.explainer import generate_overlay_data


def test_overlay_points_are_ints_not_floats():
    keypoints = {
        "pastern_left": (100.4, 200.6, 0.9),
        "hoof_left": (105.7, 250.3, 0.9),
    }
    overlay = generate_overlay_data("foot_angle", keypoints)
    contract_points = [[int(round(p[0])), int(round(p[1]))] for p in overlay["points"]]

    assert all(isinstance(v, int) for pt in contract_points for v in pt)
    # Rounds correctly, doesn't truncate.
    assert contract_points[0] == [100, 201]
    assert contract_points[1] == [106, 250]
