"""Torch ``Dataset`` / ``DataLoader`` built on top of a MedChron manifest.

Each item runs the fast DIP enhancement path (denoise + CLAHE) before handing a
normalised tensor to the backbone, so the model always sees the *enhanced* image
— the same preprocessing the rest of the platform uses.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ..config import PreprocessConfig
from ..data.manifest import Sample
from ..imaging import model_image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(train: bool, input_size: int = 224) -> transforms.Compose:
    """Augmentation for train (mild — laterality-preserving) vs. plain eval."""
    if train:
        return transforms.Compose(
            [
                transforms.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.ColorJitter(brightness=0.10, contrast=0.10),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )


class MedChronDataset(Dataset):
    """Serves ``(tensor, label_index)`` pairs from a list of :class:`Sample`."""

    def __init__(
        self,
        samples: Sequence[Sample],
        class_to_idx: Dict[str, int],
        preprocess: Optional[PreprocessConfig] = None,
        train: bool = False,
    ) -> None:
        self.samples = list(samples)
        self.class_to_idx = class_to_idx
        self.cfg = preprocess or PreprocessConfig()
        self.tf = build_transforms(train, self.cfg.model_input_size[0])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = cv2.imread(s.path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {s.path}")
        rgb = model_image(image, self.cfg)          # HxWx3 uint8, enhanced
        tensor = self.tf(Image.fromarray(rgb))
        return tensor, self.class_to_idx[s.label]


def build_class_index(samples: Sequence[Sample]) -> Dict[str, int]:
    """Deterministic label -> index mapping (sorted for reproducibility)."""
    labels = sorted({s.label for s in samples})
    return {label: i for i, label in enumerate(labels)}


def class_weights(samples: Sequence[Sample], class_to_idx: Dict[str, int]) -> torch.Tensor:
    """Inverse-frequency weights for ``CrossEntropyLoss`` to counter imbalance."""
    counts = Counter(s.label for s in samples)
    total = sum(counts.values())
    n_classes = len(class_to_idx)
    weights = torch.ones(n_classes, dtype=torch.float32)
    for label, idx in class_to_idx.items():
        c = counts.get(label, 0)
        weights[idx] = total / (n_classes * c) if c else 0.0
    return weights


@dataclass
class LoaderBundle:
    loaders: Dict[str, DataLoader]
    class_to_idx: Dict[str, int]
    class_weights: torch.Tensor


def build_dataloaders(
    samples: Sequence[Sample],
    *,
    preprocess: Optional[PreprocessConfig] = None,
    batch_size: int = 16,
    num_workers: int = 0,
) -> LoaderBundle:
    """Split samples by their ``.split`` field into train/val/test loaders."""
    class_to_idx = build_class_index(samples)
    by_split: Dict[str, List[Sample]] = {"train": [], "val": [], "test": []}
    for s in samples:
        by_split.setdefault(s.split, []).append(s)

    loaders: Dict[str, DataLoader] = {}
    for split, items in by_split.items():
        if split not in ("train", "val", "test") or not items:
            continue
        ds = MedChronDataset(items, class_to_idx, preprocess, train=(split == "train"))
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    weights = class_weights(by_split.get("train", samples), class_to_idx)
    return LoaderBundle(loaders=loaders, class_to_idx=class_to_idx, class_weights=weights)
