"""Pixel-level lesion segmentation: a U-Net (pretrained EfficientNet-B0
encoder + a skip-connected decoder) predicting a full-resolution lesion
mask, used to derive a crop region for LocalizedPredictor -- see
inference.py's _SegmentationLocalizer.

Exists because bbox regression (see detect.py) plateaued at IoU ~0.05-0.08:
a single 4-number regression target is a sparse training signal for this
dataset's size. A per-pixel mask gives thousands of supervised points per
image instead of four. Same "no geometric augmentation" rule as detect.py's
BBoxDataset applies here too, more critically -- RandomAffine would shift
the image without shifting the mask, breaking pixel alignment.
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
import torch.nn.functional as F
import torchvision.models as tvm
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..config import PreprocessConfig
from ..imaging import model_image
from .dataset import build_transforms
from .train import resolve_device, set_seed


@dataclass
class SegmentationSample:
    path: str
    mask_paths: List[str]
    split: str = ""


def read_segmentation_manifest(path: Union[str, Path]) -> List[SegmentationSample]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [
            SegmentationSample(row["path"], row["mask_paths"].split(";"), row["split"])
            for row in csv.DictReader(fh)
        ]


def write_segmentation_manifest(samples: Sequence[SegmentationSample], out_path: Union[str, Path]) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "mask_paths", "split"])
        writer.writeheader()
        for s in samples:
            writer.writerow({"path": s.path, "mask_paths": ";".join(s.mask_paths), "split": s.split})
    return out_path


def mask_to_bbox(mask: np.ndarray, threshold: float = 0.5) -> Optional[tuple]:
    """Returns normalized (cx, cy, w, h) of the thresholded nonzero region
    in a (H, W) mask array, or None if nothing is above threshold. Shared
    between manifest building (ground-truth boxes) and inference-time
    mask -> crop-region conversion (_SegmentationLocalizer)."""
    ys, xs = (mask > threshold).nonzero()
    if len(xs) == 0:
        return None
    h_img, w_img = mask.shape[:2]
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    cx = (x0 + x1) / 2 / w_img
    cy = (y0 + y1) / 2 / h_img
    w = (x1 - x0) / w_img
    h = (y1 - y0) / h_img
    return cx, cy, w, h


class SegmentationDataset(Dataset):
    """No geometric augmentation on purpose -- see module docstring."""

    def __init__(self, samples: Sequence[SegmentationSample], preprocess: Optional[PreprocessConfig] = None) -> None:
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

        size = self.cfg.model_input_size
        mask = np.zeros((size[1], size[0]), dtype=np.float32)
        for mp in s.mask_paths:
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            m_resized = cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)
            mask = np.maximum(mask, (m_resized > 10).astype(np.float32))
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        return tensor, mask_tensor


def build_segmentation_dataloaders(
    samples: Sequence[SegmentationSample],
    *,
    preprocess: Optional[PreprocessConfig] = None,
    batch_size: int = 8,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    by_split: Dict[str, List[SegmentationSample]] = {"train": [], "val": [], "test": []}
    for s in samples:
        by_split.setdefault(s.split, []).append(s)

    loaders: Dict[str, DataLoader] = {}
    for split, items in by_split.items():
        if split not in ("train", "val", "test") or not items:
            continue
        ds = SegmentationDataset(items, preprocess)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        )
    return loaders


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNetSegmenter(nn.Module):
    """EfficientNet-B0 encoder (features[0..8], pretrained) + a 4-level
    skip-connected decoder back to input resolution. Stage shapes for a
    224x224 input (verified empirically against this torchvision version):
    block 1 -> 16ch/112x112, block 2 -> 24ch/56x56, block 3 -> 40ch/28x28,
    block 5 -> 112ch/14x14, block 8 (bottleneck) -> 1280ch/7x7."""

    _SKIP_BLOCKS = (1, 2, 3, 5)
    _BOTTLENECK_BLOCK = 8

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.encoder = tvm.efficientnet_b0(weights=weights).features

        self.up4 = _UpBlock(1280, 112, 256)
        self.up3 = _UpBlock(256, 40, 128)
        self.up2 = _UpBlock(128, 24, 64)
        self.up1 = _UpBlock(64, 16, 32)
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: Dict[int, torch.Tensor] = {}
        h = x
        for i, block in enumerate(self.encoder):
            h = block(h)
            if i in self._SKIP_BLOCKS:
                skips[i] = h
            if i == self._BOTTLENECK_BLOCK:
                bottleneck = h

        d = self.up4(bottleneck, skips[5])
        d = self.up3(d, skips[3])
        d = self.up2(d, skips[2])
        d = self.up1(d, skips[1])
        logits = self.final_up(d)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits).flatten(1)
    target = target.flatten(1)
    intersection = (probs * target).sum(1)
    union = probs.sum(1) + target.sum(1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)


@torch.no_grad()
def _mean_dice(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    preds = (torch.sigmoid(logits) > threshold).float().flatten(1)
    target = target.flatten(1)
    intersection = (preds * target).sum(1)
    union = preds.sum(1) + target.sum(1)
    dice = (2 * intersection + eps) / (union + eps)
    return dice.sum().item()


@dataclass
class SegmentationTrainConfig:
    epochs: int = 15
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 8
    patience: int = 5
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"
    out_dir: str = "artifacts/mammography_segmentation"


def _run_segmentation_epoch(model, loader, device, optimizer=None) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss, total_dice, seen = 0.0, 0.0, 0
    with torch.set_grad_enabled(training):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = segmentation_loss(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            total_dice += _mean_dice(logits.detach(), targets)
            seen += inputs.size(0)
    return total_loss / max(seen, 1), total_dice / max(seen, 1)


def train_segmenter(
    samples: Sequence[SegmentationSample],
    tcfg: Optional[SegmentationTrainConfig] = None,
    preprocess: Optional[PreprocessConfig] = None,
) -> dict:
    tcfg = tcfg or SegmentationTrainConfig()
    preprocess = preprocess or PreprocessConfig()
    set_seed(tcfg.seed)
    device = resolve_device(tcfg.device)

    loaders = build_segmentation_dataloaders(
        samples, preprocess=preprocess, batch_size=tcfg.batch_size, num_workers=tcfg.num_workers,
    )
    if "train" not in loaders:
        raise ValueError("No training samples found (check the manifest's 'split' column).")

    model = UNetSegmenter(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)

    out_dir = Path(tcfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "unet_efficientnet_b0.pt"

    print(f"Device: {device} | UNetSegmenter (efficientnet_b0 encoder)", flush=True)

    best_val, bad_epochs, best_state = float("inf"), 0, None
    history: List[dict] = []
    for epoch in range(tcfg.epochs):
        tr_loss, tr_dice = _run_segmentation_epoch(model, loaders["train"], device, optimizer)
        if "val" in loaders:
            va_loss, va_dice = _run_segmentation_epoch(model, loaders["val"], device)
        else:
            va_loss, va_dice = tr_loss, tr_dice

        history.append({
            "epoch": epoch + 1, "train_loss": round(tr_loss, 4), "train_dice": round(tr_dice, 4),
            "val_loss": round(va_loss, 4), "val_dice": round(va_dice, 4),
        })
        print(f"[seg] epoch {epoch + 1:02d}/{tcfg.epochs}  "
              f"train_loss={tr_loss:.4f} dice={tr_dice:.3f}  "
              f"val_loss={va_loss:.4f} dice={va_dice:.3f}", flush=True)

        if va_loss < best_val - 1e-4:
            best_val, bad_epochs = va_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if "val" in loaders and bad_epochs >= tcfg.patience:
                print(f"[seg] early stopping at epoch {epoch + 1}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "preprocess": preprocess.to_dict(),
            "train_config": asdict(tcfg),
            "history": history,
            "task": "segmentation",
        },
        ckpt_path,
    )
    (out_dir / "segmentation_history.json").write_text(json.dumps(history, indent=2))
    print(f"Saved checkpoint -> {ckpt_path}", flush=True)
    return {"checkpoint": str(ckpt_path), "history": history}


def evaluate_segmenter(
    ckpt_path: str,
    samples: Sequence[SegmentationSample],
    *,
    split: str = "test",
    batch_size: int = 8,
    device: Optional[str] = None,
) -> Dict:
    device = resolve_device(device or "auto")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = UNetSegmenter(pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    preprocess = PreprocessConfig(**ckpt.get("preprocess", {}))
    loaders = build_segmentation_dataloaders(samples, preprocess=preprocess, batch_size=batch_size)
    if split not in loaders:
        raise ValueError(f"No '{split}' samples to evaluate.")

    _, mean_dice = _run_segmentation_epoch(model, loaders[split], device)
    metrics = {"mean_dice": mean_dice}
    print(json.dumps(metrics, indent=2))
    return metrics
