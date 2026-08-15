"""Detection module configuration: model paths, thresholds, and raw-label mapping."""

RAW_LABEL_TO_CLASS_NAME = {
    # Placeholder mapping between the fine-tuned RT-DETR checkpoint's native
    # labels and the canonical class names used downstream. Keys may be class
    # indices (int) or class-name strings, depending on the checkpoint.
    0: "animal",
    1: "ear_tag",
    "cow": "animal",
    "buffalo": "animal",
    "cattle": "animal",
    "animal": "animal",
    "ear_tag": "ear_tag",
}

CANONICAL_CLASS_NAMES = ("animal", "ear_tag")

RTDETR_MODEL_PATH = "models/rtdetr/animal_tag_rtdetr.pt"
RTDETR_CONFIDENCE_THRESHOLD = 0.5

SAM2_MODEL_PATH = "models/sam2/sam2_hiera_tiny.pt"
SAM2_CONFIG_PATH = "models/sam2/sam2_hiera_tiny.yaml"

DEFAULT_DEVICE = "auto"
DEBUG_DIR = "debug/detection"