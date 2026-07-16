
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ..config import PreprocessConfig
from ..data.manifest import LABEL_DELIM, Sample, build_multilabel_class_index
from ..imaging import model_image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

Task = Literal["multiclass", "multilabel"]


def build_transforms(train: bool, input_size: int = 224) -> transforms.Compose:
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

    def __init__(
        self,
        samples: Sequence[Sample],
        class_to_idx: Dict[str, int],
        preprocess: Optional[PreprocessConfig] = None,
        train: bool = False,
        task: Task = "multiclass",
        label_delim: str = LABEL_DELIM,
    ) -> None:
        self.samples = list(samples)
        self.class_to_idx = class_to_idx
        self.cfg = preprocess or PreprocessConfig()
        self.tf = build_transforms(train, self.cfg.model_input_size[0])
        self.task = task
        self.label_delim = label_delim

    def __len__(self) -> int:
        return len(self.samples)

    def _target(self, label: str):
        if self.task == "multilabel":
            target = torch.zeros(len(self.class_to_idx), dtype=torch.float32)
            for part in label.split(self.label_delim):
                part = part.strip()
                if part in self.class_to_idx:
                    target[self.class_to_idx[part]] = 1.0
            return target
        return self.class_to_idx[label]

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = cv2.imread(s.path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {s.path}")
        rgb = model_image(image, self.cfg)
        tensor = self.tf(Image.fromarray(rgb))
        return tensor, self._target(s.label)


def build_class_index(samples: Sequence[Sample]) -> Dict[str, int]:
    labels = sorted({s.label for s in samples})
    return {label: i for i, label in enumerate(labels)}


def class_weights(samples: Sequence[Sample], class_to_idx: Dict[str, int]) -> torch.Tensor:
    counts = Counter(s.label for s in samples)
    total = sum(counts.values())
    n_classes = len(class_to_idx)
    weights = torch.ones(n_classes, dtype=torch.float32)
    for label, idx in class_to_idx.items():
        c = counts.get(label, 0)
        weights[idx] = total / (n_classes * c) if c else 0.0
    return weights


def multilabel_pos_weights(
    samples: Sequence[Sample], class_to_idx: Dict[str, int], delim: str = LABEL_DELIM
) -> torch.Tensor:
    n = len(samples)
    pos_counts: Counter = Counter()
    for s in samples:
        for part in s.label.split(delim):
            part = part.strip()
            if part in class_to_idx:
                pos_counts[part] += 1
    weights = torch.ones(len(class_to_idx), dtype=torch.float32)
    for label, idx in class_to_idx.items():
        pos = pos_counts.get(label, 0)
        weights[idx] = (n - pos) / pos if pos > 0 else 1.0
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
    task: Task = "multiclass",
    label_delim: str = LABEL_DELIM,
) -> LoaderBundle:
    if task == "multilabel":
        class_to_idx = build_multilabel_class_index(samples, delim=label_delim)
    else:
        class_to_idx = build_class_index(samples)

    by_split: Dict[str, List[Sample]] = {"train": [], "val": [], "test": []}
    for s in samples:
        by_split.setdefault(s.split, []).append(s)

    loaders: Dict[str, DataLoader] = {}
    for split, items in by_split.items():
        if split not in ("train", "val", "test") or not items:
            continue
        ds = MedChronDataset(
            items, class_to_idx, preprocess, train=(split == "train"),
            task=task, label_delim=label_delim,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    train_samples = by_split.get("train") or samples
    if task == "multilabel":
        weights = multilabel_pos_weights(train_samples, class_to_idx, delim=label_delim)
    else:
        weights = class_weights(train_samples, class_to_idx)
    return LoaderBundle(loaders=loaders, class_to_idx=class_to_idx, class_weights=weights)
