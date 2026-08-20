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

# Decoder layer to read ear_tag predictions from.
#
# RT-DETR trains every decoder layer with the same auxiliary losses, so each
# layer carries a fully supervised classification + box head. In this
# checkpoint the FINAL layer's ear_tag head collapsed while layer 1 stayed
# healthy. Measured on the held-out TEST split (528 images, 399 tag boxes)
# at threshold 0.5:
#     layer 1 (this setting) : AP@0.5 0.954, precision 95.7%, recall 94.5%,
#                              0.032 false positives per image
#     final layer (previous) : AP@0.5 0.001 - effectively no detections
# All layers localise tags identically (~0.91 IoU); only the final layer's
# CONFIDENCE died, so this reads the model's own trained output from a head
# that works. Note those are AP@0.5 figures, not COCO mAP@[0.50:0.95].
#
# Set to None to restore the previous final-layer behaviour.
EAR_TAG_DECODER_LAYER = 1

SAM2_MODEL_PATH = "models/sam2/sam2_hiera_tiny.pt"
SAM2_CONFIG_PATH = "models/sam2/sam2_hiera_tiny.yaml"
DEFAULT_DEVICE = "auto"
DEBUG_DIR = "debug/detection"