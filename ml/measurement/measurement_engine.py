"""Superseded by ml/measurement/traits.py.

The architecture document named this module, but the work landed in traits.py
instead: measure_trait() / measure_all_traits() handle class A angles, class B
ratios and class C centimetre distances, and refuse rather than guess when a
landmark is missing, the scale is absent, or the value could not describe a
real animal.

Nothing imports this file. It is kept as a signpost so the next person looking
for "the measurement engine" is sent to the right place instead of assuming a
stage is unimplemented.

The SMAL fallback in the original name does NOT exist. Traits marked
measure_class "SMAL" (Heart Girth, Body Condition Score) need a 3D body model
that has not been built, and they report not_scored_reason rather than an
estimate.
"""
