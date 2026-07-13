"""Dataset manifests — a uniform, framework-agnostic index of samples.

A *manifest* is just a list of :class:`Sample` rows (path, label, patient, split)
that can be written to / read from CSV. Every downstream consumer (TensorFlow
``tf.data`` or PyTorch ``Dataset``) is built on top of a manifest, so the data
plumbing never depends on how a particular dataset happens to lay out its files.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class Sample:
    """One image row in a manifest."""

    path: str
    label: str
    patient_id: str = ""     # empty string == unknown; enables leak-free splits
    split: str = ""          # "train" | "val" | "test" | "" (unassigned)


PatientResolver = Callable[[Path], str]


def build_manifest_from_folders(
    root: Union[str, Path],
    patient_from: Optional[PatientResolver] = None,
) -> List[Sample]:
    """Index an ``ImageFolder``-style dataset: ``root/<class_name>/*.png``.

    ``patient_from`` optionally derives a patient id from each file path (e.g.
    parsing it out of the filename) so splits can be made patient-aware.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root not found: {root}")

    samples: List[Sample] = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for img in sorted(class_dir.rglob("*")):
            if img.suffix.lower() in IMAGE_EXTS:
                pid = patient_from(img) if patient_from else ""
                samples.append(Sample(str(img), class_dir.name, pid))
    return samples


def build_manifest_from_csv(
    csv_path: Union[str, Path],
    *,
    path_col: str = "path",
    label_col: str = "label",
    patient_col: Optional[str] = None,
) -> List[Sample]:
    """Index a dataset described by a labels CSV."""
    samples: List[Sample] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            samples.append(
                Sample(
                    path=row[path_col],
                    label=str(row[label_col]),
                    patient_id=str(row[patient_col]) if patient_col else "",
                )
            )
    return samples


def write_manifest(samples: List[Sample], out_path: Union[str, Path]) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [f.name for f in fields(Sample)]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for s in samples:
            writer.writerow(asdict(s))
    return out_path


def read_manifest(path: Union[str, Path]) -> List[Sample]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [Sample(**row) for row in csv.DictReader(fh)]


def class_distribution(samples: List[Sample]) -> Dict[str, int]:
    """Count samples per label — used to detect and report class imbalance."""
    return dict(Counter(s.label for s in samples))


def split_distribution(samples: List[Sample]) -> Dict[str, Dict[str, int]]:
    """Per-split class counts, e.g. ``{'train': {'normal': 80, ...}, ...}``."""
    out: Dict[str, Dict[str, int]] = {}
    for s in samples:
        out.setdefault(s.split or "unassigned", Counter())[s.label] += 1  # type: ignore[index]
    return {k: dict(v) for k, v in out.items()}
