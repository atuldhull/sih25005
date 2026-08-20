"""Trait registry: authoritative list of NDDB-style type traits and lookup helpers.

CONTRACT_TRAITS: the exact 20 traits emitted in `traits[]` per contract/scoring_result.json.
Every entry has trait_class in {"A","B","C","SMAL"}, plus `category` and `view` fields
required by the contract.

INTERNAL_EXTRA_TRAITS: your original angle/ratio traits that do NOT appear in the NDDB
scorecard. Keep computing them if useful internally (they may feed BCS/Angularity/etc.
in future), but NEVER emit them into the contract's traits[] list.

NOTE ON MISSING KEYPOINTS: the 8 Udder traits, Angularity, and Body Condition Score
reference keypoint names (e.g. "udder_floor", "teat_front_left") that do not yet exist
in ml/pose_features/keypoint_schema.py. Until that schema is extended, these traits will
correctly resolve to not_measurable (missing keypoint -> confidence 0), which is the
intended graceful-degradation behavior, not a bug. Extending the keypoint schema is a
separate, urgent follow-up task (see REVIEW-ml-dev.md B3).
"""

CONTRACT_TRAITS = [
    # ---------------- Dairy Strength ----------------
    {
        "trait_id": "stature",
        "name": "Stature",
        "category": "Dairy Strength",
        "trait_class": "C",
        "view": "side",
        "required_keypoints": ["withers", "hoof_left"],
        "required_scale": True,
        "unit": "cm",
        "smal_fallback": True,
        "species_variants": ["cattle", "buffalo"],
    },
    {
        "trait_id": "heart_girth",
        "name": "Heart Girth",
        "category": "Dairy Strength",
        "trait_class": "SMAL",
        "view": "side",
        "required_keypoints": ["withers", "chest_bottom"],
        "required_scale": True,
        "unit": "cm",
        "smal_fallback": True,
        # NOTE: 2D chest-depth is NOT true girth circumference (Fix #3 in
        # REVIEW-ml-dev.md). Do not feed the raw 2D distance into Schaeffer's
        # formula - that was a real bug, and refusing was the right answer to it.
        #
        # It is now answered without a SMAL mesh, by the route that note was
        # really asking for. ml/weight/volume_3d.py builds the chest station as
        # a CLOSED cross-section from both photographs - depth from the side,
        # width from the rear - so its perimeter is a circumference rather than
        # a chord. The pipeline substitutes that measurement (see
        # _with_heart_girth) whenever a scale is available, carrying an
        # uncertainty for the elliptical assumption and the tag scale.
        #
        # trait_class stays SMAL here because that is what it is WITHOUT the
        # substitution: with no rear silhouette or no scale, this trait still
        # refuses, and the definition has to describe that case.
        "species_variants": ["cattle", "buffalo"],
    },
    {
        "trait_id": "body_length",
        "name": "Body Length",
        "category": "Dairy Strength",
        "trait_class": "C",
        "view": "side",
        "required_keypoints": ["chest_front", "pin_right"],
        "required_scale": True,
        "unit": "cm",
        "smal_fallback": True,
        "species_variants": ["cattle", "buffalo"],
    },
    {
        "trait_id": "body_depth",
        "name": "Body Depth",
        "category": "Dairy Strength",
        "trait_class": "C",
        "view": "side",
        "required_keypoints": ["withers", "chest_bottom"],
        "required_scale": True,
        "unit": "cm",
        "smal_fallback": True,
        "species_variants": ["cattle", "buffalo"],
    },
    {
        "trait_id": "angularity",
        "name": "Angularity",
        "category": "Dairy Strength",
        "trait_class": "A",
        "view": "side",
        # TODO: rib_angle_* keypoints do not exist yet in keypoint_schema.py.
        "required_keypoints": ["rib_angle_top", "rib_angle_mid", "rib_angle_bottom"],
        "required_scale": False,
        "unit": "degrees",
        "smal_fallback": True,
        "species_variants": ["cattle", "buffalo"],
    },
    # ---------------- Rump ----------------
    {
        "trait_id": "rump_angle",
        "name": "Rump Angle",
        "category": "Rump",
        "trait_class": "A",
        "view": "side",
        "required_keypoints": ["hook_left", "pin_left", "hook_right", "pin_right"],
        "required_scale": False,
        "unit": "degrees",
        "smal_fallback": True,
        # Fix #minor: do NOT reverse-direction this trait; keep raw ICAR angle convention.
        "species_variants": ["cattle", "buffalo"],
    },
    {
        "trait_id": "rump_width",
        "name": "Rump Width",
        "category": "Rump",
        "trait_class": "C",
        "view": "rear",
        "required_keypoints": ["hook_left", "hook_right"],
        "required_scale": True,
        "unit": "cm",
        "smal_fallback": True,
        # NOTE: buffalo rump-width landmarks differ from cattle per REVIEW-ml-dev.md B3.
        # species_variants below is a placeholder; actual per-species keypoint mapping
        # should be resolved in measurement/traits.py, not here.
        "species_variants": ["cattle", "buffalo"],
    },
    # ---------------- Feet & Legs ----------------
    {
        "trait_id": "rear_legs_set",
        "name": "Rear Legs Set",
        "category": "Feet & Legs",
        "trait_class": "A",
        "view": "side",
        # Contract defines Rear Legs Set as the SIDE-VIEW hock angle, not the
        # rear-view hip->hock cow-hock deviation (that geometry now belongs to
        # rear_legs_rear_view below). Shares its geometry with the internal
        # "hock_angle" trait.
        # The upper landmark is hip_bone, NOT knee. In cattle the knee (carpus)
        # is a FORE-leg joint and the hock a REAR-leg one, so knee->hock->pastern
        # spans two different legs and measures nothing anatomical. The rear
        # leg's chain is hip_bone -> hock -> pastern, and the hock angle is the
        # angle that chain makes at the hock.
        #
        # Over 74 photographs, angle at the hock:
        #
        #     knee -> hock -> pastern (was)   n=11   median  67.1    18% in band
        #     hip_bone -> hock -> pastern     n=40   median 153.5    48% in band
        #
        # The median moves from far outside the 130-160 band to inside it, and
        # the sample count nearly quadruples because knee_left is one of the
        # joints that most often collapses onto its neighbours.
        "required_keypoints": ["hip_bone_left", "hock_left", "pastern_left"],
        "required_scale": False,
        "unit": "degrees",
        "smal_fallback": True,
        "species_variants": ["cattle", "buffalo"],
    },
    {
        "trait_id": "rear_legs_rear_view",
        "name": "Rear Legs Rear View",
        "category": "Feet & Legs",
        # Was "B" (ratio) with only 2 keypoints - structurally unmeasurable
        # (needs 4 points for _compute_ratio) and the wrong class for this
        # geometry anyway. Now hosts the hip->hock cow-hock deviation-from-
        # -vertical angle (previously mis-attached to rear_legs_set above),
        # which is an angle computation, so trait_class must be "A" to route
        # through the correct branch in measurement/traits.py.
        "trait_class": "A",
        # Was "video" (this trait previously assumed gait/video tracking was
        # required). The cow-hock deviation geometry is static per-frame
        # rear-image geometry, not motion tracking, so "rear" is correct.
        "view": "rear",
        "required_keypoints": ["hip_bone_left", "hip_bone_right", "hock_left", "hock_right"],
        "required_scale": False,
        "unit": "degrees",
        "smal_fallback": False,
        "species_variants": ["cattle", "buffalo"],
    },
    {
        "trait_id": "foot_angle",
        "name": "Foot Angle",
        "category": "Feet & Legs",
        "trait_class": "A",
        "view": "side",
        # 2 points (not 3): _compute_angle()'s 2-point branch returns the
        # pastern->hoof line's angle from horizontal, which is what Foot
        # Angle needs. 3 points would route into the interior-vertex-angle
        # branch instead (the wrong geometry - that measured the interior
        # pastern joint angle, not the pastern-to-hoof angle vs horizontal).
        "required_keypoints": ["pastern_left", "hoof_left"],
        "required_scale": False,
        "unit": "degrees",
        "smal_fallback": True,
        "species_variants": ["cattle", "buffalo"],
    },
    # ---------------- Udder ----------------
    {
        "trait_id": "fore_udder_attachment",
        "name": "Fore Udder Attachment",
        "category": "Udder",
        "trait_class": "B",
        "view": "side",
        # TODO: udder keypoints not yet in keypoint_schema.py.
        # A class-B trait is a RATIO of two distances, so _compute_ratio needs
        # four points: |p0-p1| / |p2-p3|. With two it returns None on every
        # image, forever - this trait could never have measured, no matter how
        # good the landmarks were. Four of the twenty were defined this way,
        # and all four are udder traits waiting on annotation, so nearly half
        # of that labelling effort would have produced nothing.
        #
        # The denominator below is chosen to make the ratio a FRACTION, which
        # is what the calibrated band of 0..1 implies. The exact ICAR
        # convention for this trait has not been confirmed against the
        # standard - the band may have been written for a different
        # normalisation - so treat the range as provisional and check it
        # before anyone scores an animal on it. What is no longer provisional
        # is that the definition can produce a number at all.
        "required_keypoints": ["fore_udder_top", "fore_udder_body_junction",
                               "chest_bottom", "back_mid"],
        "required_scale": False,
        "unit": "ratio",
        "smal_fallback": True,
        "species_variants": ["cattle", "buffalo"],
    },
    {
        "trait_id": "rear_udder_height",
        "name": "Rear Udder Height",
        "category": "Udder",
        "trait_class": "C",
        "view": "rear",
        # TODO: udder keypoints not yet in keypoint_schema.py.
        "required_keypoints": ["vulva_base", "rear_udder_top"],
        "required_scale": True,
        "unit": "cm",
        "smal_fallback": True,
        "species_variants": ["cattle", "buffalo"],
    },
    {
        "trait_id": "central_ligament",
        "name": "Central Ligament",
        "category": "Udder",
        "trait_class": "B",
        "view": "rear",
        # TODO: udder keypoints not yet in keypoint_schema.py.
        # A class-B trait is a RATIO of two distances, so _compute_ratio needs
        # four points: |p0-p1| / |p2-p3|. With two it returns None on every
        # image, forever - this trait could never have measured, no matter how
        # good the landmarks were. Four of the twenty were defined this way,
        # and all four are udder traits waiting on annotation, so nearly half
        # of that labelling effort would have produced nothing.
        #
        # The denominator below is chosen to make the ratio a FRACTION, which
        # is what the calibrated band of 0..1 implies. The exact ICAR
        # convention for this trait has not been confirmed against the
        # standard - the band may have been written for a different
        # normalisation - so treat the range as provisional and check it
        # before anyone scores an animal on it. What is no longer provisional
        # is that the definition can produce a number at all.
        "required_keypoints": ["udder_cleft_top", "udder_cleft_bottom",
                               "rear_udder_top", "udder_floor"],
        "required_scale": False,
        "unit": "ratio",
        "smal_fallback": True,
        "species_variants": ["cattle", "buffalo"],
    },
    {
        "trait_id": "udder_depth",
        "name": "Udder Depth",
        "category": "Udder",
        "trait_class": "C",
        "view": "rear",
        # rear_hock_left, NOT hock_left. udder_floor is merged from the rear
        # photograph and hock_left is not, so the pair spanned two coordinate
        # frames and produced 74.7 cm against a band of -10 to 25 - a units
        # error wearing the shape of a plausible measurement. See
        # REAR_FRAME_ALIASES in ml/pose_features/silhouette_landmarks.py.
        "required_keypoints": ["udder_floor", "rear_hock_left"],
        "required_scale": True,
        "unit": "cm",
        "smal_fallback": True,
        "species_variants": ["cattle", "buffalo"],
    },
    {
        "trait_id": "front_teat_placement",
        "name": "Front Teat Placement",
        "category": "Udder",
        "trait_class": "B",
        "view": "rear",
        # TODO: teat keypoints not yet in keypoint_schema.py.
        # A class-B trait is a RATIO of two distances, so _compute_ratio needs
        # four points: |p0-p1| / |p2-p3|. With two it returns None on every
        # image, forever - this trait could never have measured, no matter how
        # good the landmarks were. Four of the twenty were defined this way,
        # and all four are udder traits waiting on annotation, so nearly half
        # of that labelling effort would have produced nothing.
        #
        # The denominator below is chosen to make the ratio a FRACTION, which
        # is what the calibrated band of 0..1 implies. The exact ICAR
        # convention for this trait has not been confirmed against the
        # standard - the band may have been written for a different
        # normalisation - so treat the range as provisional and check it
        # before anyone scores an animal on it. What is no longer provisional
        # is that the definition can produce a number at all.
        "required_keypoints": ["teat_front_left", "teat_front_right",
                               "rear_udder_left", "rear_udder_right"],
        "required_scale": False,
        "unit": "ratio",
        "smal_fallback": False,
        "species_variants": ["cattle", "buffalo"],
    },
    {
        "trait_id": "teat_length",
        "name": "Teat Length",
        "category": "Udder",
        "trait_class": "C",
        "view": "rear",
        # NOTE: measured on left FRONT teat for cattle, left REAR teat for buffalo
        # (REVIEW-ml-dev.md B3 + tag_spec note). Resolve the correct pair per-species
        # in measurement/traits.py; this entry lists the cattle default.
        "required_keypoints": ["teat_front_left_top", "teat_front_left_bottom"],
        "required_scale": True,
        "unit": "cm",
        "smal_fallback": False,
        "species_variants": ["cattle", "buffalo"],
    },
    {
        "trait_id": "rear_teat_placement",
        "name": "Rear Teat Placement",
        "category": "Udder",
        "trait_class": "B",
        "view": "rear",
        # TODO: teat keypoints not yet in keypoint_schema.py.
        # A class-B trait is a RATIO of two distances, so _compute_ratio needs
        # four points: |p0-p1| / |p2-p3|. With two it returns None on every
        # image, forever - this trait could never have measured, no matter how
        # good the landmarks were. Four of the twenty were defined this way,
        # and all four are udder traits waiting on annotation, so nearly half
        # of that labelling effort would have produced nothing.
        #
        # The denominator below is chosen to make the ratio a FRACTION, which
        # is what the calibrated band of 0..1 implies. The exact ICAR
        # convention for this trait has not been confirmed against the
        # standard - the band may have been written for a different
        # normalisation - so treat the range as provisional and check it
        # before anyone scores an animal on it. What is no longer provisional
        # is that the definition can produce a number at all.
        "required_keypoints": ["teat_rear_left", "teat_rear_right",
                               "rear_udder_left", "rear_udder_right"],
        "required_scale": False,
        "unit": "ratio",
        "smal_fallback": False,
        "species_variants": ["cattle", "buffalo"],
    },
    {
        "trait_id": "rear_udder_width",
        "name": "Rear Udder Width",
        "category": "Udder",
        "trait_class": "C",
        "view": "rear",
        # TODO: udder keypoints not yet in keypoint_schema.py.
        "required_keypoints": ["rear_udder_left", "rear_udder_right"],
        "required_scale": True,
        "unit": "cm",
        "smal_fallback": True,
        "species_variants": ["cattle", "buffalo"],
    },
    {
        "trait_id": "teat_thickness",
        "name": "Teat Thickness",
        "category": "Udder",
        "trait_class": "C",
        "view": "rear",
        # TODO: teat keypoints not yet in keypoint_schema.py. Expect this trait to be
        # not_measurable / low-confidence for a while - the contract's own example
        # (see contract/scoring_result.json) shows Teat Thickness as not_scored,
        # confirming this is an accepted, expected state, not a bug to hide.
        "required_keypoints": ["teat_width_left_1", "teat_width_left_2"],
        "required_scale": True,
        "unit": "cm",
        "smal_fallback": False,
        "species_variants": ["cattle", "buffalo"],
    },
    # ---------------- General ----------------
    {
        "trait_id": "body_condition_score",
        "name": "Body Condition Score",
        "category": "General",
        "trait_class": "SMAL",
        "view": "side",
        # TODO: requires SMAL mesh fit over pelvis/tailhead surface shape.
        "required_keypoints": ["pin_left", "pin_right", "tail_head"],
        "required_scale": False,
        "unit": "score",
        "smal_fallback": True,
        "species_variants": ["cattle", "buffalo"],
    },
]

# Your original extra angle/ratio traits - NOT part of the NDDB scorecard, NOT emitted
# in the contract's traits[] list. Kept here in case they're useful as an internal
# feature layer (e.g. future auxiliary signals). result_builder.to_contract_dict()
# must only ever iterate CONTRACT_TRAITS, never this list.
INTERNAL_EXTRA_TRAITS = [
    {
        "trait_id": "hock_angle",
        "name": "Hock Angle",
        "trait_class": "A",
        # The upper landmark is hip_bone, NOT knee. In cattle the knee (carpus)
        # is a FORE-leg joint and the hock a REAR-leg one, so knee->hock->pastern
        # spans two different legs and measures nothing anatomical. The rear
        # leg's chain is hip_bone -> hock -> pastern, and the hock angle is the
        # angle that chain makes at the hock.
        #
        # Over 74 photographs, angle at the hock:
        #
        #     knee -> hock -> pastern (was)   n=11   median  67.1    18% in band
        #     hip_bone -> hock -> pastern     n=40   median 153.5    48% in band
        #
        # The median moves from far outside the 130-160 band to inside it, and
        # the sample count nearly quadruples because knee_left is one of the
        # joints that most often collapses onto its neighbours.
        "required_keypoints": ["hip_bone_left", "hock_left", "pastern_left"],
        "required_scale": False,
        "unit": "degrees",
        "smal_fallback": True,
        "species_variants": ["cattle", "buffalo"],
    },
    {
        "trait_id": "fore_leg_set",
        "name": "Fore Leg Set",
        "trait_class": "A",
        "required_keypoints": ["shoulder_left", "shoulder_right", "hoof_left", "hoof_right"],
        "required_scale": False,
        "unit": "degrees",
        "smal_fallback": True,
        "species_variants": ["cattle", "buffalo"],
    },
    {
        "trait_id": "shoulder_angle",
        "name": "Shoulder Angle",
        "trait_class": "A",
        # 2 points (not 3), for the same reason as foot_angle above. The rule
        # band 45-65 is the classic scapula slope - the angle of the shoulder
        # line from horizontal - not an interior angle at the shoulder joint.
        # With withers/shoulder_left/chest_front it routed into the
        # interior-vertex branch and measured a different quantity entirely.
        # Over 14 photographs where all the landmarks were available:
        #
        #     3-pt interior angle   median 15.0 deg    1/14 in band
        #     2-pt slope            median 63.3 deg    3/14 in band
        #
        # The median moving from far outside the band to inside it is what
        # identifies which quantity the band was written for.
        "required_keypoints": ["withers", "chest_front"],
        "required_scale": False,
        "unit": "degrees",
        "smal_fallback": True,
        "species_variants": ["cattle", "buffalo"],
    },
    {
        "trait_id": "body_length_to_height_ratio",
        "name": "Body Length to Height Ratio",
        "trait_class": "B",
        "required_keypoints": ["chest_front", "pin_left", "withers", "hoof_left"],
        "required_scale": False,
        "unit": "ratio",
        "smal_fallback": True,
        "species_variants": ["cattle", "buffalo"],
    },
    {
        "trait_id": "chest_width_to_depth_ratio",
        "name": "Chest Width to Depth Ratio",
        "trait_class": "B",
        "required_keypoints": ["chest_width_left", "chest_width_right", "withers", "chest_bottom"],
        "required_scale": False,
        "unit": "ratio",
        "smal_fallback": True,
        "species_variants": ["cattle", "buffalo"],
    },
    {
        "trait_id": "topline_symmetry",
        "name": "Topline Symmetry",
        "trait_class": "B",
        "required_keypoints": ["withers", "back_mid", "hook_left", "pin_left"],
        "required_scale": False,
        "unit": "ratio",
        "smal_fallback": True,
        "species_variants": ["cattle", "buffalo"],
    },
    {
        "trait_id": "pin_width_to_hook_distance_ratio",
        "name": "Pin Width to Hook Distance Ratio",
        "trait_class": "B",
        "required_keypoints": ["hook_left", "hook_right", "pin_left", "pin_right"],
        "required_scale": False,
        "unit": "ratio",
        "smal_fallback": True,
        "species_variants": ["cattle", "buffalo"],
    },
    {
        "trait_id": "udder_depth_ratio",
        "name": "Udder Depth Ratio",
        "trait_class": "B",
        "required_keypoints": ["rear_udder", "hock_left", "hock_right", "hoof_right"],
        "required_scale": False,
        "unit": "ratio",
        "smal_fallback": True,
        "species_variants": ["cattle", "buffalo"],
    },
    {
        "trait_id": "body_width_to_length_ratio",
        "name": "Body Width to Length Ratio",
        "trait_class": "B",
        "required_keypoints": ["chest_width_left", "chest_width_right", "chest_front", "pin_left"],
        "required_scale": False,
        "unit": "ratio",
        "smal_fallback": True,
        "species_variants": ["cattle", "buffalo"],
    },
    {
        "trait_id": "chest_width",
        "name": "Chest Width",
        "trait_class": "C",
        "required_keypoints": ["chest_width_left", "chest_width_right"],
        "required_scale": True,
        "unit": "cm",
        "smal_fallback": True,
        "species_variants": ["cattle", "buffalo"],
    },
    {
        "trait_id": "chest_depth",
        "name": "Chest Depth",
        "trait_class": "C",
        "required_keypoints": ["withers", "chest_bottom"],
        "required_scale": True,
        "unit": "cm",
        "smal_fallback": True,
        "species_variants": ["cattle", "buffalo"],
    },
    {
        "trait_id": "rump_length",
        "name": "Rump Length",
        "trait_class": "C",
        "required_keypoints": ["hook_left", "pin_left"],
        "required_scale": True,
        "unit": "cm",
        "smal_fallback": True,
        "species_variants": ["cattle", "buffalo"],
    },
]

# Kept for backward compatibility with any existing code that imports TRAIT_REGISTRY
# directly. New code (result_builder.py, measurement engine) should prefer
# CONTRACT_TRAITS for anything that touches the contract output.
TRAIT_REGISTRY = CONTRACT_TRAITS + INTERNAL_EXTRA_TRAITS


def get_trait(trait_id: str) -> dict:
    """Return the registry entry for a single trait id (searches contract + internal)."""
    for trait in TRAIT_REGISTRY:
        if trait["trait_id"] == trait_id:
            return trait
    raise KeyError(f"Unknown trait_id: {trait_id!r}")


def get_traits_by_class(trait_class: str) -> list:
    """Return all registry entries belonging to the given trait class (A, B, C, or SMAL)."""
    return [trait for trait in TRAIT_REGISTRY if trait["trait_class"] == trait_class]


def get_contract_traits() -> list:
    """Return only the 20 traits that belong in the contract's traits[] output."""
    return CONTRACT_TRAITS