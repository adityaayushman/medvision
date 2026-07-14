"""Convert the MIAS mammography dataset (PGM + Info.txt) into a MedChron manifest.

MIAS ships one grayscale PGM per scan plus a text file of abnormality
annotations (``Info.txt``: REFNUM BG CLASS SEVERITY X Y RADIUS). This derives
the standard literature 3-class task from it:

    CLASS == NORM        -> Normal
    CLASS != NORM, SEV=B  -> Benign
    CLASS != NORM, SEV=M  -> Malignant

A scan with more than one annotated abnormality (a handful of REFNUMs appear
on multiple rows) keeps its *first* row's label -- consistent with how this
dataset is conventionally used for classification.

Usage:
    python ml/scripts/prepare_mias.py \
        --raw-dir ml/data/mammography/raw/all-mias \
        --out-dir ml/data/mammography/prepared
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from medchron.data import class_distribution, split_distribution, stratified_split, write_manifest
from medchron.data.manifest import Sample
from medchron.data.split import assert_no_patient_leakage


def load_labels(info_path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    lines = info_path.read_text(encoding="utf-8").splitlines()[1:]  # skip header
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        ref = parts[0]
        if ref in labels:
            continue  # keep first occurrence only
        cls = parts[2] if len(parts) > 2 else "NORM"
        sev = parts[3] if len(parts) > 3 else None
        if cls == "NORM":
            labels[ref] = "Normal"
        elif sev == "B":
            labels[ref] = "Benign"
        elif sev == "M":
            labels[ref] = "Malignant"
    return labels


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a MedChron manifest from the MIAS dataset")
    ap.add_argument("--raw-dir", required=True, help="folder with mdb*.pgm + Info.txt")
    ap.add_argument("--out-dir", default="ml/data/mammography/prepared")
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--test-size", type=float, default=0.15)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    png_dir = out_dir / "images"
    png_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(raw_dir / "Info.txt")
    print(f"Labeled refs: {len(labels)}")

    samples: list[Sample] = []
    skipped = 0
    for pgm_path in sorted(raw_dir.glob("*.pgm")):
        ref = pgm_path.stem
        label = labels.get(ref)
        if label is None:
            skipped += 1
            continue
        png_path = png_dir / f"{ref}.png"
        if not png_path.exists():
            img = cv2.imread(str(pgm_path), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(str(png_path), img)
        samples.append(Sample(str(png_path), label, patient_id=ref))

    print(f"Converted {len(samples)} images ({skipped} skipped: unlabeled).")

    samples = stratified_split(samples, val_size=args.val_size, test_size=args.test_size)
    assert_no_patient_leakage(samples)

    manifest_path = write_manifest(samples, out_dir / "manifest.csv")
    print(f"\nManifest: {manifest_path} ({len(samples)} samples)")
    print("Class distribution:", json.dumps(class_distribution(samples)))
    print("Per-split:", json.dumps(split_distribution(samples), indent=2))


if __name__ == "__main__":
    main()
