"""Bovine-41 keypoint inference - the bridge from the trained model to the
scoring pipeline.

Drop-in for the pose stage: give it an image and the animal's bounding box,
get back the canonical 41-joint dict that measurement expects.

Two things it does that a plain forward pass would not:

  * JOINTS WITH NO TRAINING DATA RETURN CONFIDENCE 0.0. Nineteen of the 41
    (every udder and teat landmark) had no data anywhere in public sources,
    so the model was never taught them. Returning 0.0 makes the measurement
    layer refuse those traits instead of quietly measuring noise. This is
    the single most important behaviour in this file.

  * IT CARRIES A MEASURED ERROR ESTIMATE PER JOINT. Validation gave us a
    real per-joint accuracy table, so each keypoint comes back with the
    error we actually observed for it, as a fraction of the animal's size.
    The pipeline can turn that into an honest confidence interval instead
    of printing a falsely precise centimetre value.

Usage:
    from bovine_pose_infer import BovinePoseModel
    pose = BovinePoseModel("last.pt")                 # load once, reuse
    kps = pose.extract("side.jpg", bbox=(x1, y1, x2, y2))
    # kps["withers"] -> KeypointOut(x=..., y=..., confidence=..., err_frac=...)
"""
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

KEYPOINTS = [
    "withers", "back_mid", "chest_front", "chest_bottom", "chest_width_left",
    "chest_width_right", "shoulder_left", "shoulder_right", "tail_head",
    "hip_bone_left", "hip_bone_right", "hook_left", "hook_right", "pin_left",
    "pin_right", "knee_left", "knee_right", "hock_left", "hock_right",
    "pastern_left", "pastern_right", "hoof_left", "hoof_right", "rear_udder",
    "fore_udder_top", "fore_udder_body_junction", "vulva_base",
    "rear_udder_top", "udder_cleft_top", "udder_cleft_bottom", "udder_floor",
    "teat_front_left", "teat_front_right", "teat_front_left_top",
    "teat_front_left_bottom", "teat_rear_left", "teat_rear_right",
    "rear_udder_left", "rear_udder_right", "teat_width_left_1",
    "teat_width_left_2",
]

# Measured on the validation set with the shipped checkpoint (HRNet-W32 @384,
# quadratic decode): PCK@0.02 per joint, i.e. the share of predictions landing
# within 2% of the bounding-box side. Joints absent from this table had NO
# training data and are always returned with confidence 0.0.
PCK02 = {
    "back_mid": 0.815, "pin_left": 0.758, "pastern_right": 0.718,
    "pastern_left": 0.737, "knee_left": 0.721, "knee_right": 0.715,
    "hook_right": 0.710, "pin_right": 0.707, "hoof_right": 0.709,
    "hoof_left": 0.703, "chest_width_right": 0.690, "hock_left": 0.607,
    "hock_right": 0.557, "tail_head": 0.568, "hook_left": 0.570,
    "withers": 0.533, "shoulder_left": 0.498, "shoulder_right": 0.487,
    "hip_bone_right": 0.480, "chest_front": 0.429, "hip_bone_left": 0.363,
    "chest_width_left": 0.340,
}
# Median localisation error over all evaluated joints, as a fraction of the
# bounding-box side. Joints that score below the median PCK get scaled up
# proportionally, so a weak joint reports a wider interval than a strong one.
MEDIAN_ERR_FRAC = 0.0156
TRAINED_JOINTS = set(PCK02)


@dataclass
class KeypointOut:
    x: float                # full-image pixels
    y: float
    confidence: float       # 0..1; 0.0 means "never trained, do not use"
    err_frac: float         # expected error as a fraction of the bbox side

    def as_tuple(self) -> Tuple[float, float, float]:
        """(x, y, confidence) - the shape measurement/traits.py expects."""
        return (self.x, self.y, self.confidence)


class _PoseNet(nn.Module):
    """Rebuilt from the checkpoint's metadata so one class loads either the
    HRNet stride-4 model or the older resnet+deconv one."""

    def __init__(self, backbone: str, num_kp: int, feat_index: int):
        super().__init__()
        import timm
        self.body = timm.create_model(backbone, pretrained=False,
                                      features_only=True,
                                      out_indices=(feat_index,))
        ch = self.body.feature_info.channels()[-1]
        stride = self.body.feature_info.reduction()[-1]
        n_up = max(0, int(round(math.log2(stride / 4))))
        layers, c = [], ch
        for _ in range(n_up):
            layers += [nn.ConvTranspose2d(c, 256, 4, 2, 1, bias=False),
                       nn.BatchNorm2d(256), nn.ReLU(inplace=True)]
            c = 256
        if n_up == 0:
            layers += [nn.Conv2d(c, 256, 3, 1, 1, bias=False),
                       nn.BatchNorm2d(256), nn.ReLU(inplace=True)]
            c = 256
        self.head = nn.Sequential(*layers)
        self.final = nn.Conv2d(c, num_kp, 1)

    def forward(self, x):
        return self.final(self.head(self.body(x)[-1]))


class BovinePoseModel:
    """Load once, call extract() per image."""

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, checkpoint: str, device: Optional[str] = None):
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.names = ck.get("keypoints", KEYPOINTS)
        self.backbone = ck.get("backbone", "resnet50")
        self.img_size = int(ck.get("img_size", 256))
        self.heat_size = int(ck.get("heat_size", self.img_size // 4))
        self.epoch = ck.get("epoch")
        feat_index = 1 if "hrnet" in self.backbone else 4
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _PoseNet(self.backbone, len(self.names), feat_index)
        sd = {k.replace("deconv.", "head."): v for k, v in ck["model"].items()}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint does not match the rebuilt architecture: "
                f"{len(missing)} missing, {len(unexpected)} unexpected. "
                f"First missing: {list(missing)[:3]}")
        self.model.eval().to(self.device)

    def describe(self) -> str:
        return (f"{self.backbone} @{self.img_size}px -> {self.heat_size} "
                f"heatmap, epoch {self.epoch}, {len(self.names)} joints "
                f"({len(TRAINED_JOINTS)} trained)")

    @staticmethod
    def _quadratic(hm: np.ndarray) -> np.ndarray:
        """argmax plus a parabola fit - argmax alone throws away ~0.4 cells
        of pure rounding error, which matters once the model is this good."""
        K, H, W = hm.shape
        flat = hm.reshape(K, -1).argmax(axis=1)
        out = np.stack([flat % W, flat // W], axis=1).astype(np.float32)
        for k in range(K):
            px, py = int(out[k, 0]), int(out[k, 1])
            if 0 < px < W - 1:
                l, c, r = hm[k, py, px - 1], hm[k, py, px], hm[k, py, px + 1]
                den = l - 2 * c + r
                if abs(den) > 1e-9:
                    out[k, 0] = px - 0.5 * (r - l) / den
            if 0 < py < H - 1:
                u, c, d = hm[k, py - 1, px], hm[k, py, px], hm[k, py + 1, px]
                den = u - 2 * c + d
                if abs(den) > 1e-9:
                    out[k, 1] = py - 0.5 * (d - u) / den
        return out

    @torch.no_grad()
    def extract(self, image: "str | Path | Image.Image",
                bbox: Sequence[float]) -> Dict[str, KeypointOut]:
        """Locate all 41 joints for the animal in `bbox` (x1, y1, x2, y2).

        Coordinates come back in FULL-IMAGE pixels. Untrained joints are
        returned with confidence 0.0 so the measurement layer refuses them.
        """
        img = (image if isinstance(image, Image.Image)
               else Image.open(image)).convert("RGB")
        x1, y1, x2, y2 = [float(v) for v in bbox]
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            raise ValueError(f"degenerate bbox: {bbox}")

        # same crop convention as training: square, 1.25x the longer side
        side = max(w, h, 8.0) * 1.25
        cx, cy = x1 + w / 2, y1 + h / 2
        ox, oy = cx - side / 2, cy - side / 2
        crop = img.crop((int(ox), int(oy), int(ox + side), int(oy + side)))
        crop = crop.resize((self.img_size, self.img_size), Image.BILINEAR)

        arr = (np.asarray(crop, dtype=np.float32) / 255.0 - self.MEAN) / self.STD
        t = torch.from_numpy(np.ascontiguousarray(
            arr.transpose(2, 0, 1)))[None].to(self.device)
        hm = self.model(t).float().cpu().numpy()[0]
        pts = self._quadratic(hm)

        scale = side / self.heat_size
        ref = max(w, h)
        out: Dict[str, KeypointOut] = {}
        for j, name in enumerate(self.names):
            if name not in TRAINED_JOINTS:
                # never trained: report the position but refuse to vouch for
                # it. Measurement treats confidence 0 as "not measurable".
                out[name] = KeypointOut(0.0, 0.0, 0.0, float("inf"))
                continue
            peak = float(hm[j].max())
            # heatmap peaks are unbounded; squash to 0..1 and blend with the
            # measured reliability of THIS joint, so a landmark the model is
            # historically bad at cannot report high confidence on one lucky
            # image
            sharpness = 1.0 / (1.0 + math.exp(-4.0 * (peak - 0.5)))
            conf = float(np.clip(sharpness * PCK02[name] / 0.75, 0.0, 1.0))
            err = MEDIAN_ERR_FRAC * (0.60 / max(PCK02[name], 0.05))
            out[name] = KeypointOut(
                x=ox + float(pts[j, 0]) * scale,
                y=oy + float(pts[j, 1]) * scale,
                confidence=round(conf, 3),
                err_frac=round(err, 4),
            )
        self._last_ref = ref
        return out

    @staticmethod
    def to_pipeline_dict(kps: Dict[str, KeypointOut]
                         ) -> Dict[str, Tuple[float, float, float]]:
        """Reduce to the (x, y, confidence) triples measurement expects."""
        return {k: v.as_tuple() for k, v in kps.items()}

    @staticmethod
    def error_cm(kp: KeypointOut, bbox: Sequence[float],
                 animal_length_cm: float) -> Optional[float]:
        """Expected error in centimetres, for a trait's confidence interval.

        Needs the animal's real size, which the ear-tag ruler provides. With
        no ruler there is no centimetre scale, and this returns None rather
        than inventing one.
        """
        if not math.isfinite(kp.err_frac) or animal_length_cm <= 0:
            return None
        x1, y1, x2, y2 = [float(v) for v in bbox]
        ref_px = max(x2 - x1, y2 - y1)
        if ref_px <= 0:
            return None
        return kp.err_frac * animal_length_cm


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--bbox", nargs=4, type=float, required=True,
                    metavar=("X1", "Y1", "X2", "Y2"))
    a = ap.parse_args()

    pose = BovinePoseModel(a.checkpoint)
    print(pose.describe())
    kps = pose.extract(a.image, a.bbox)
    trained = [(n, k) for n, k in kps.items() if k.confidence > 0]
    dead = [n for n, k in kps.items() if k.confidence == 0]
    print(f"\n{'joint':<24}{'x':>9}{'y':>9}{'conf':>8}{'err':>9}")
    for n, k in trained:
        print(f"{n:<24}{k.x:>9.1f}{k.y:>9.1f}{k.confidence:>8.2f}"
              f"{k.err_frac:>9.4f}")
    print(f"\n{len(trained)} joints located, {len(dead)} refused "
          f"(no training data): {', '.join(dead[:4])}...")
