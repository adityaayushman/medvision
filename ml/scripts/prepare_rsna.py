"""Convert the RSNA Pneumonia DICOM images to PNG and build a manifest.

RSNA ships images as DICOM (.dcm), which OpenCV cannot read directly — this
script decodes each one with pydicom, normalises to 8-bit grayscale, and
writes a flat PNG. Labels come from the richer 3-class breakdown
(``stage_2_detailed_class_info.csv``: Normal / No Lung Opacity - Not Normal /
Lung Opacity) rather than the raw binary Target, since it's what the dataset
itself provides and gives the classifier a more clinically useful distinction.

Only ``stage_2_train_images`` is used — ``stage_2_test_images`` is the
competition's unlabeled holdout set, unusable for training or evaluation here.
We build our own leak-safe train/val/test split from the labeled images.

Usage:
    python ml/scripts/prepare_rsna.py \
        --raw-dir "ml/data/rsna_pneumonia/drive-download-20240112T131344Z-002" \
        --out-dir ml/data/rsna_pneumonia/prepared
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import pydicom

from medchron.data import stratified_split, write_manifest
from medchron.data.manifest import Sample
from medchron.data.split import assert_no_patient_leakage


def dicom_to_png_array(dcm_path: Path) -> np.ndarray:
    """Read a DICOM file and return an 8-bit grayscale array."""
    ds = pydicom.dcmread(str(dcm_path))
    arr = ds.pixel_array.astype(np.float32)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    else:
        arr = np.zeros_like(arr)
    return arr.astype(np.uint8)


def load_labels(raw_dir: Path) -> dict[str, str]:
    """patientId -> class name, from stage_2_detailed_class_info.csv."""
    labels: dict[str, str] = {}
    with open(raw_dir / "stage_2_detailed_class_info.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            labels[row["patientId"]] = row["class"]
    return labels


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert RSNA DICOMs to PNG + build a manifest")
    ap.add_argument("--raw-dir", required=True, help="folder with stage_2_train_images/ + CSVs")
    ap.add_argument("--out-dir", default="ml/data/rsna_pneumonia/prepared")
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--limit", type=int, default=None, help="cap #images (debugging only)")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    images_dir = raw_dir / "stage_2_train_images"
    out_dir = Path(args.out_dir)
    png_dir = out_dir / "images"
    png_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(raw_dir)
    dcm_files = sorted(images_dir.glob("*.dcm"))
    if args.limit:
        dcm_files = dcm_files[: args.limit]
    print(f"Found {len(dcm_files)} DICOM files, {len(labels)} labeled patients.")

    samples: list[Sample] = []
    skipped = 0
    start = time.time()
    for i, dcm_path in enumerate(dcm_files, 1):
        patient_id = dcm_path.stem
        label = labels.get(patient_id)
        if label is None:
            skipped += 1
            continue

        png_path = png_dir / f"{patient_id}.png"
        if not png_path.exists():
            arr = dicom_to_png_array(dcm_path)
            cv2.imwrite(str(png_path), arr)
        samples.append(Sample(str(png_path), label, patient_id=patient_id))

        if i % 1000 == 0 or i == len(dcm_files):
            elapsed = time.time() - start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(dcm_files) - i) / rate if rate > 0 else 0
            print(f"  {i}/{len(dcm_files)}  ({rate:.1f} img/s, ETA {eta/60:.1f} min)", flush=True)

    print(f"Converted {len(samples)} images ({skipped} skipped: no label).")

    samples = stratified_split(samples, val_size=args.val_size, test_size=args.test_size)
    assert_no_patient_leakage(samples)

    manifest_path = write_manifest(samples, out_dir / "manifest.csv")
    from medchron.data import class_distribution, split_distribution
    import json
    print(f"\nManifest: {manifest_path}  ({len(samples)} samples)")
    print("Class distribution:", json.dumps(class_distribution(samples)))
    print("Per-split:", json.dumps(split_distribution(samples), indent=2))


if __name__ == "__main__":
    main()
