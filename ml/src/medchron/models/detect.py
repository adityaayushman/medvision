"""Lightweight single-box lesion localization: predicts one normalized
(cx, cy, w, h) bounding box per image, used to crop a full mammogram down
to the lesion region before classification (see inference.LocalizedPredictor).

Deliberately a separate, minimal module rather than extending train.py's
classification pipeline: regression needs a different loss/metric (IoU, not
accuracy) and — critically — no geometric augmentation (build_transforms'
RandomAffine would rotate/shift the image without moving the box target),
so reusing MedChronDataset/train() as-is isn't safe here.

The model is also architecturally distinct from the classification backbones
in backbone.py: create_model()'s heads global-average-pool the last feature
map to a single vector before the final Linear layer, which destroys the
spatial position information a box center needs. A first attempt reusing
that pattern for bbox regression collapsed to predicting a near-constant
y-center (val IoU 0.043) regardless of input. SpatialBBoxNet instead keeps
the last conv feature map and reads the center off it directly via a
spatial soft-argmax (a differentiable weighted average over actual grid
locations) — width/height still come from a pooled-feature path, since box
extent doesn't need the same pixel-precise grounding as its location.
"""

from __future__ import annotations

import copy
import csv
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..config import PreprocessConfig
from ..imaging import model_image
from .dataset import build_transforms
from .train import resolve_device, set_seed


@dataclass
class BBoxSample:
    path: str
    cx: float
    cy: float
    w: float
    h: float
    split: str = ""


def read_bbox_manifest(path: Union[str, Path]) -> List[BBoxSample]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [
            BBoxSample(row["path"], float(row["cx"]), float(row["cy"]), float(row["w"]), float(row["h"]), row["split"])
            for row in csv.DictReader(fh)
        ]


def write_bbox_manifest(samples: Sequence[BBoxSample], out_path: Union[str, Path]) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [f.name for f in fields(BBoxSample)]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for s in samples:
            writer.writerow(asdict(s))
    return out_path


def box_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = box.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def box_iou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Elementwise IoU between two (N, 4) batches of (cx, cy, w, h) boxes."""
    p = box_to_xyxy(pred)
    t = box_to_xyxy(target)
    x0 = torch.max(p[:, 0], t[:, 0])
    y0 = torch.max(p[:, 1], t[:, 1])
    x1 = torch.min(p[:, 2], t[:, 2])
    y1 = torch.min(p[:, 3], t[:, 3])
    inter = (x1 - x0).clamp(min=0) * (y1 - y0).clamp(min=0)
    area_p = (p[:, 2] - p[:, 0]).clamp(min=0) * (p[:, 3] - p[:, 1]).clamp(min=0)
    area_t = (t[:, 2] - t[:, 0]).clamp(min=0) * (t[:, 3] - t[:, 1]).clamp(min=0)
    union = area_p + area_t - inter
    return inter / union.clamp(min=1e-6)


class BBoxDataset(Dataset):
    """No geometric augmentation on purpose — see module docstring."""

    def __init__(self, samples: Sequence[BBoxSample], preprocess: Optional[PreprocessConfig] = None) -> None:
        self.samples = list(samples)
        self.cfg = preprocess or PreprocessConfig()
        self.tf = build_transforms(train=False, input_size=self.cfg.model_input_size[0])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = cv2.imread(s.path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {s.path}")
        rgb = model_image(image, self.cfg)
        tensor = self.tf(Image.fromarray(rgb))
        target = torch.tensor([s.cx, s.cy, s.w, s.h], dtype=torch.float32)
        return tensor, target


def build_bbox_dataloaders(
    samples: Sequence[BBoxSample],
    *,
    preprocess: Optional[PreprocessConfig] = None,
    batch_size: int = 16,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    by_split: Dict[str, List[BBoxSample]] = {"train": [], "val": [], "test": []}
    for s in samples:
        by_split.setdefault(s.split, []).append(s)

    loaders: Dict[str, DataLoader] = {}
    for split, items in by_split.items():
        if split not in ("train", "val", "test") or not items:
            continue
        ds = BBoxDataset(items, preprocess)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        )
    return loaders


class SpatialBBoxNet(nn.Module):
    """EfficientNet-B0 conv trunk (no classification head) + a spatial
    soft-argmax localization head. (cx, cy) is a softmax-weighted average
    over the last feature map's grid positions, so it's derived from *where*
    the network's attention is, not a black-box linear readout of pooled
    features.

    (w, h) reuse that same attention map to weight-pool the feature vector
    (an early version instead did a plain uniform global-average-pool for
    size — that collapsed to predicting close to the dataset's mean box size
    regardless of the actual lesion, since a uniformly-pooled vector mixes in
    the whole image, diluting whatever local size cue exists). Weighting by
    the same heatmap used for the center means the size head only sees
    features concentrated near the detected location, not the whole image."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.trunk = tvm.efficientnet_b0(weights=weights).features
        channels = 1280
        self.heat_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.size_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(channels, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.trunk(x)
        b, c, h, w = feats.shape
        heat = torch.softmax(self.heat_conv(feats).reshape(b, h * w), dim=1)  # (B, HW)

        gy, gx = torch.meshgrid(
            torch.linspace(0.0, 1.0, h, device=x.device),
            torch.linspace(0.0, 1.0, w, device=x.device),
            indexing="ij",
        )
        cx = (heat * gx.reshape(1, h * w)).sum(dim=1, keepdim=True)
        cy = (heat * gy.reshape(1, h * w)).sum(dim=1, keepdim=True)

        feats_flat = feats.reshape(b, c, h * w)
        local_pooled = torch.bmm(feats_flat, heat.unsqueeze(2)).squeeze(2)  # (B, C)
        wh = torch.sigmoid(self.size_head(local_pooled))
        return torch.cat([cx, cy, wh], dim=1)


@dataclass
class BBoxTrainConfig:
    backbone: str = "efficientnet_b0"  # currently the only supported value
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 16
    patience: int = 5
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"
    out_dir: str = "artifacts/mammography_bbox"


def _run_bbox_epoch(model, loader, device, optimizer=None) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    criterion = nn.SmoothL1Loss()
    total_loss, total_iou, seen = 0.0, 0.0, 0
    with torch.set_grad_enabled(training):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            preds = model(inputs)  # already constrained to [0,1] internally
            loss = criterion(preds, targets)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            total_iou += box_iou(preds.detach(), targets).sum().item()
            seen += inputs.size(0)
    return total_loss / max(seen, 1), total_iou / max(seen, 1)


def train_bbox_regressor(
    samples: Sequence[BBoxSample],
    tcfg: Optional[BBoxTrainConfig] = None,
    preprocess: Optional[PreprocessConfig] = None,
) -> dict:
    tcfg = tcfg or BBoxTrainConfig()
    preprocess = preprocess or PreprocessConfig()
    set_seed(tcfg.seed)
    device = resolve_device(tcfg.device)

    loaders = build_bbox_dataloaders(
        samples, preprocess=preprocess, batch_size=tcfg.batch_size, num_workers=tcfg.num_workers,
    )
    if "train" not in loaders:
        raise ValueError("No training samples found (check the manifest's 'split' column).")
    if tcfg.backbone != "efficientnet_b0":
        raise ValueError(f"SpatialBBoxNet only supports efficientnet_b0, got {tcfg.backbone!r}")

    model = SpatialBBoxNet(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)

    out_dir = Path(tcfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"bbox_{tcfg.backbone}.pt"

    print(f"Device: {device} | backbone: {tcfg.backbone} | bbox regression", flush=True)

    best_val, bad_epochs, best_state = float("inf"), 0, None
    history: List[dict] = []
    for epoch in range(tcfg.epochs):
        tr_loss, tr_iou = _run_bbox_epoch(model, loaders["train"], device, optimizer)
        if "val" in loaders:
            va_loss, va_iou = _run_bbox_epoch(model, loaders["val"], device)
        else:
            va_loss, va_iou = tr_loss, tr_iou

        history.append({
            "epoch": epoch + 1, "train_loss": round(tr_loss, 4), "train_iou": round(tr_iou, 4),
            "val_loss": round(va_loss, 4), "val_iou": round(va_iou, 4),
        })
        print(f"[bbox] epoch {epoch + 1:02d}/{tcfg.epochs}  "
              f"train_loss={tr_loss:.4f} iou={tr_iou:.3f}  "
              f"val_loss={va_loss:.4f} iou={va_iou:.3f}", flush=True)

        if va_loss < best_val - 1e-4:
            best_val, bad_epochs = va_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if "val" in loaders and bad_epochs >= tcfg.patience:
                print(f"[bbox] early stopping at epoch {epoch + 1}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "preprocess": preprocess.to_dict(),
            "train_config": asdict(tcfg),
            "history": history,
            "task": "bbox_regression",
        },
        ckpt_path,
    )
    (out_dir / "bbox_history.json").write_text(json.dumps(history, indent=2))
    print(f"Saved checkpoint -> {ckpt_path}", flush=True)
    return {"checkpoint": str(ckpt_path), "history": history}


def evaluate_bbox_checkpoint(
    ckpt_path: str,
    samples: Sequence[BBoxSample],
    *,
    split: str = "test",
    batch_size: int = 16,
    device: Optional[str] = None,
) -> Dict:
    device = resolve_device(device or "auto")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = SpatialBBoxNet(pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    preprocess = PreprocessConfig(**ckpt.get("preprocess", {}))
    loaders = build_bbox_dataloaders(samples, preprocess=preprocess, batch_size=batch_size)
    if split not in loaders:
        raise ValueError(f"No '{split}' samples to evaluate.")

    _, mean_iou = _run_bbox_epoch(model, loaders[split], device)
    metrics = {"mean_iou": mean_iou}
    print(json.dumps(metrics, indent=2))
    return metrics
