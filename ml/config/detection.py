"""Detection module configuration: model paths, thresholds, and raw-label mapping."""

RAW_LABEL_TO_CLASS_NAME = {
    # Placeholder mapping between the fine-tuned RT-DETRv2 checkpoint's native
    # labels and the canonical class names used downstream. Keys are the
    # integer class indices the model was trained with (0=animal, 1=ear_tag,
    # matching data.yaml / the COCO categories used in Kaggle training).
    0: "animal",
    1: "ear_tag",
}
CANONICAL_CLASS_NAMES = ("animal", "ear_tag")

# CHANGED (REVIEW-ml-dev.md B4): now a DIRECTORY, not a single .pt file.
# HuggingFace `transformers` loads via RTDetrV2ForObjectDetection.from_pretrained(path),
# which expects a directory containing config.json + model weights (safetensors/bin)
# + preprocessor_config.json - the standard HF export format. When exporting the
# Kaggle-trained checkpoint, use model.save_pretrained(RTDETR_MODEL_PATH) and
# image_processor.save_pretrained(RTDETR_MODEL_PATH) into this same directory.
RTDETR_MODEL_PATH = "models/rtdetr/animal_tag_rtdetr"

RTDETR_CONFIDENCE_THRESHOLD = 0.5
SAM2_MODEL_PATH = "models/sam2/sam2_hiera_tiny.pt"
SAM2_CONFIG_PATH = "models/sam2/sam2_hiera_tiny.yaml"
DEFAULT_DEVICE = "auto"
DEBUG_DIR = "debug/detection"