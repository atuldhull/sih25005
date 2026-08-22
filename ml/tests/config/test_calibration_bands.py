"""The calibrated bands, checked for the kinds of error that stay silent.

A trait with no band cannot score. A band whose minimum exceeds its maximum
scores nothing. A band measured in the wrong unit scores everything wrongly.
None of those raise; they just produce a system that quietly refuses, or
quietly agrees, and the difference is invisible from the outside.

This is the same class of check as
test_trait_definitions_are_computable.py, which found four contract traits
that could never have produced a number.
"""
import pytest

from ml.config.rules import SPECIES_RULES
from ml.config.traits import CONTRACT_TRAITS, TRAIT_REGISTRY

SPECIES = sorted(SPECIES_RULES)
BY_ID = {t["trait_id"]: t for t in TRAIT_REGISTRY}

# What a value in each unit can plausibly be, for an animal. Deliberately
# generous - these catch a band in the wrong unit entirely, not a band that is
# a few degrees out.
UNIT_LIMITS = {
    "ratio": (0.0, 25.0),
    "degrees": (-180.0, 360.0),
    # cm may be NEGATIVE. Some traits are measured relative to a reference
    # landmark rather than as an absolute size: udder_depth runs from -10 to
    # 25 because an udder floor ABOVE the hock is a well-attached udder, and
    # that is a real and meaningful reading rather than a sign error.
    "cm": (-100.0, 400.0),
}


@pytest.mark.parametrize("species", SPECIES)
def test_every_contract_trait_has_a_band(species):
    """Without one, score_from_value has nothing to compare against and the
    trait refuses on every animal, forever."""
    missing = [t["trait_id"] for t in CONTRACT_TRAITS
               if t["trait_id"] not in SPECIES_RULES[species]]
    assert not missing, (
        f"{species}: contract traits with no calibrated band, so they can "
        f"never score: {missing}")


@pytest.mark.parametrize("species", SPECIES)
def test_no_band_is_inverted_or_empty(species):
    for trait_id, band in SPECIES_RULES[species].items():
        lo, hi = float(band["min"]), float(band["max"])
        assert lo < hi, f"{species}/{trait_id}: min {lo} is not below max {hi}"


@pytest.mark.parametrize("species", SPECIES)
def test_a_one_to_nine_score_has_nine_bins(species):
    for trait_id, band in SPECIES_RULES[species].items():
        bins = band.get("bins")
        if bins is None:
            continue
        assert len(bins) == 9, (
            f"{species}/{trait_id}: {len(bins)} bins for a 1-9 score")


@pytest.mark.parametrize("species", SPECIES)
def test_each_band_is_in_the_unit_its_trait_declares(species):
    """A band written in the wrong unit is the worst kind of silent error: it
    scores every animal, and every score is wrong."""
    for trait_id, band in SPECIES_RULES[species].items():
        trait = BY_ID.get(trait_id)
        if trait is None:
            continue
        limits = UNIT_LIMITS.get(trait["unit"])
        if limits is None:
            continue
        lo, hi = float(band["min"]), float(band["max"])
        assert limits[0] <= lo and hi <= limits[1], (
            f"{species}/{trait_id}: band {lo}-{hi} does not look like "
            f"{trait['unit']}")


@pytest.mark.parametrize("species", SPECIES)
def test_there_is_no_band_for_a_trait_that_does_not_exist(species):
    """A band for a renamed or deleted trait is dead weight that reads as
    coverage - and if the trait ever comes back under that name, it inherits
    a calibration nobody checked."""
    orphans = [t for t in SPECIES_RULES[species] if t not in BY_ID]
    assert not orphans, f"{species}: bands for unknown traits: {orphans}"
