
from __future__ import annotations

import csv
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
LABEL_DELIM = "|"


@dataclass
class Sample:

    path: str
    label: str
    patient_id: str = ""
    split: str = ""


PatientResolver = Callable[[Path], str]


def build_manifest_from_folders(
    root: Union[str, Path],
    patient_from: Optional[PatientResolver] = None,
) -> List[Sample]:
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
    path_prefix: Union[str, Path, None] = None,
) -> List[Sample]:
    prefix = Path(path_prefix) if path_prefix else None
    samples: List[Sample] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            path = row[path_col]
            if prefix is not None:
                path = str(prefix / path)
            samples.append(
                Sample(
                    path=path,
                    label=str(row[label_col]),
                    patient_id=str(row[patient_col]) if patient_col else "",
                )
            )
    return samples


def build_multilabel_class_index(
    samples: List[Sample], delim: str = LABEL_DELIM
) -> Dict[str, int]:
    vocab = set()
    for s in samples:
        vocab.update(part.strip() for part in s.label.split(delim) if part.strip())
    return {label: i for i, label in enumerate(sorted(vocab))}


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
    return dict(Counter(s.label for s in samples))


def split_distribution(samples: List[Sample]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for s in samples:
        out.setdefault(s.split or "unassigned", Counter())[s.label] += 1
    return {k: dict(v) for k, v in out.items()}
