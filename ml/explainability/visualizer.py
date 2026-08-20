"""Superseded by ml/explainability/explainer.py and the server's overlay renderer.

The architecture document named this module. In practice the work is split:

  ml/explainability/explainer.py   picks the overlay_points for each trait
                                   from the keypoints, and writes the
                                   human-readable summary lines
  ml/explainability/result_builder.py  maps the internal result onto the
                                   frozen contract shape
  server/overlays.py               renders the actual PNG the phone displays,
                                   because rendering belongs where the image
                                   files are

Nothing imports this file. It is kept as a signpost rather than deleted, so
the next person looking for "the visualizer" finds the three places that
replaced it.
"""
