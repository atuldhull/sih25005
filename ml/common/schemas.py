"""Shared data contracts (dataclasses) used across all ml_pipeline modules."""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple


@dataclass
class DetectionResult:
    """A single detection bounding box, e.g. an animal or its ear tag."""

    bbox: Tuple[float, float, float, float]
    confidence: float
    class_name: Literal["animal", "ear_tag"]


@dataclass
class TagResult:
    """Ear-tag reading result: geometry, OCR text, scale, and identity/breed resolution."""

    corners: List[Tuple[float, float]]
    scale_factor: float
    scale_confidence: float
    ocr_text: str
    ocr_confidence: float
    identity_match: Optional[str]
    breed_predicted: str
    breed_confidence: float


@dataclass
class PoseResult:
    """Skeletal keypoints of an animal, mapping joint names to (x, y, confidence)."""

    keypoints: Dict[str, Tuple[float, float, float]]
    overall_confidence: float


@dataclass
class FeatureResult:
    """Visual feature embedding extracted from a specific image viewpoint."""

    embedding: List[float]
    source_image: Literal["side", "rear"]


@dataclass
class MeasurementResult:
    """Physical measurement of a single trait, with quality flags."""

    trait_id: str
    trait_class: Literal["A", "B", "C", "SMAL"]
    value: Optional[float]
    unit: str
    confidence: float
    flags: List[str] = field(default_factory=list)
    # How much the value could be out by, in the trait's own unit, from the
    # keypoint error alone. An angle built on a SHORT segment amplifies that
    # error enormously: two joints 4% of the animal apart, each landing within
    # 1.3% of the box side, put the angle between them out by +/-25 degrees.
    # Optional so nothing that does not compute it is affected.
    uncertainty: Optional[float] = None


@dataclass
class ScoreResult:
    """Rule-based 1-9 score assigned to a single trait."""

    trait_id: str
    score_1_9: Optional[int]
    confidence: float


@dataclass
class WeightResult:
    """Estimated live weight with a plausible min-max range."""

    estimate_kg: Optional[float]
    range_kg: Tuple[Optional[float], Optional[float]]
    confidence: float
    # How this number was arrived at, and what a second route said about it.
    # Both optional so the older girth-length estimator is unaffected; the
    # contract mapping previously hardcoded the method string, which meant a
    # volume estimate would have been reported as a girth-length regression.
    method: Optional[str] = None
    cross_check: Optional[str] = None


@dataclass
class SymptomVector:
    """Presence signal for one screening category (e.g. gait, body condition)."""

    category: str
    present: bool
    confidence: float


@dataclass
class EligibilityResult:
    """Eligibility gate outcome for the animal based on the gathered evidence."""

    passed: bool
    reasons: List[str] = field(default_factory=list)


@dataclass
class ScoringResult:
    """Top-level pipeline output combining identity, trait scores, and screening."""

    animal_id: Optional[str]
    species: str
    status: Literal["SCORED", "NOT_SCORED", "PARTIAL"]
    tag: Optional[TagResult]
    traits: List[ScoreResult] = field(default_factory=list)
    weight: Optional[WeightResult] = None
    symptom_vector: List[SymptomVector] = field(default_factory=list)
    eligibility: Optional[EligibilityResult] = None
    warnings: List[str] = field(default_factory=list)
    model_versions: Dict[str, str] = field(default_factory=dict)