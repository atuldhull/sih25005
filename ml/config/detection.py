from pathlib import Path as _Path

"""Detection module configuration: model paths, thresholds, and raw-label mapping."""

RAW_LABEL_TO_CLASS_NAME = {
    # Mapping between the fine-tuned RT-DETRv2 checkpoint's native labels and
    # the canonical class names used downstream. Keys are the integer class
    # indices the model was trained with.
    #
    # AUTHORITY: the checkpoint itself. models/rtdetr/animal_tag_rtdetr/config.json
    # declares id2label {"0": "animal", "1": "ear_tag"}, and its
    # denoising_class_embed is 3x256 (num_labels + 1), which independently
    # confirms two classes.
    #
    # This comment used to say the map matched "data.yaml / the COCO categories
    # used in Kaggle training". It does not match any data.yaml on this machine,
    # and saying so invited a reviewer to go and check. The eight Roboflow
    # exports under /d/bovine-pose/raw/catbuf-indian-breeds declare
    # ['body','ear','face','horn'] - class 1 there is the anatomical EAR, not an
    # ear TAG, and class 0 is 'body', not 'animal'. Scanning every COCO
    # annotation file under /d/bovine-pose for the string "ear_tag" returns zero
    # hits: no dataset on this machine could have supplied that class.
    #
    # The ear_tag boxes were labelled separately, as the build plan directs -
    # "a few hundred labeled tag boxes is enough - label with LabelStudio"
    # (SIH25005_Team_Build_Plan.txt:267-269) - and that labelled set lives with
    # Person 2 rather than in this repo.
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
# Anchored to the REPO ROOT, not the working directory.
#
# These used to be plain relative strings, and stage_ready.bat starts the
# server with `cd /d "%~dp0"` - the server/ folder - so "models/rtdetr/..."
# resolved to server/models/rtdetr/... which does not exist. Detection would
# have reported "weights not found" on the demo machine while the weights sat
# correctly in the repo root all along.
_REPO_ROOT = _Path(__file__).resolve().parents[2]


def _at_root(rel: str) -> str:
    return str(_REPO_ROOT / rel)


RTDETR_MODEL_PATH = _at_root("models/rtdetr/animal_tag_rtdetr")

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

SAM2_MODEL_PATH = _at_root("models/sam2/sam2_hiera_tiny.pt")
# SAM2 resolves its config through Hydra, whose search path includes
# pkg://sam2 - so this is a config NAME relative to the installed sam2
# package, NOT a path on disk. Passing a filesystem path fails with
# "Cannot find primary config", which is why segmentation silently fell back
# to the bounding-box rectangle and the silhouette was never real.
SAM2_CONFIG_NAME = "configs/sam2/sam2_hiera_t.yaml"
SAM2_CONFIG_PATH = SAM2_CONFIG_NAME     # kept for existing imports
DEFAULT_DEVICE = "auto"
DEBUG_DIR = "debug/detection"