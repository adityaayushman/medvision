"""Materialize training crops framed EXACTLY the way LocalizedPredictor
crops at inference time -- ground-truth box + crop_to_bbox(pad_frac=0.15)
-- instead of CBIS-DDSM's official hand-verified crop images.

Why: the full detect->crop->classify pipeline underperformed (48.9% acc)
despite a real, working segmenter (Dice 0.254), even though the classifier
alone scores 71.1% acc on official crops. Diagnosis (see ROADMAP.md /
mammography_cbis_cropped commit): the classifier was trained only on
official crops -- a specific framing/padding/scale convention -- and
doesn't generalize to automatically-derived crops, regardless of how
accurately they're centered. This script tests the fix directly: train on
crops that already look like what the pipeline will feed the classifier at
inference, using ground-truth boxes (not the segmenter's own predictions,
to isolate "does matching the framing convention alone help" from "does
exposure to the segmenter's real localization noise help" -- two different
questions, worth testing separately).

Joins the existing classification manifest (path/label/patient_id/split)
against the existing bbox manifest (path/cx/cy/w/h/split, built by
prepare_cbis_ddsm_bbox.py from the same ROI masks) on path, crops each full
mammogram with the exact same crop_to_bbox() LocalizedPredictor calls at
inference, and writes a new manifest pointing at the crops.

Usage:
    python ml/scripts/crop_mammography_from_masks.py \
        --classification-manifest ml/data/mammography/cbis_prepared/manifest.csv \
        --bbox-manifest ml/data/mammography/cbis_bbox_prepared/bbox_manifest.csv \
        --out-dir ml/data/mammography/cbis_gtcrop_prepared
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from medchron.data.manifest import Sample, write_manifest
from medchron.models.detect import crop_to_bbox


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classification-manifest", required=True)
    ap.add_argument("--bbox-manifest", required=True)
    ap.add_argument("--out-dir", default="ml/data/mammography/cbis_gtcrop_prepared")
    ap.add_argument("--pad-frac", type=float, default=0.15,
                     help="must match LocalizedPredictor's pad_frac default (0.15)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("Loading classification manifest (labels)...")
    labels: dict[str, tuple[str, str]] = {}  # path -> (label, patient_id)
    with open(args.classification_manifest, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            labels[row["path"]] = (row["label"], row["patient_id"])
    print(f"  {len(labels)} labeled full-image rows")

    print("Loading bbox manifest (ground-truth boxes)...")
    boxes: list[dict] = []
    with open(args.bbox_manifest, newline="", encoding="utf-8") as fh:
        boxes = list(csv.DictReader(fh))
    print(f"  {len(boxes)} bbox rows")

    samples: list[Sample] = []
    skipped_no_label = 0
    skipped_unreadable = 0
    counts = {"train": 0, "val": 0, "test": 0}
    for row in boxes:
        path = row["path"]
        label_info = labels.get(path)
        if label_info is None:
            skipped_no_label += 1
            continue
        label, patient_id = label_info

        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            skipped_unreadable += 1
            continue

        crop, _box = crop_to_bbox(
            image, float(row["cx"]), float(row["cy"]), float(row["w"]), float(row["h"]),
            pad_frac=args.pad_frac,
        )
        split = row["split"]
        out_path = images_dir / f"{len(samples)}_{split}.jpg"
        cv2.imwrite(str(out_path), crop)

        samples.append(Sample(str(out_path), label, patient_id=patient_id, split=split))
        counts[split] = counts.get(split, 0) + 1

    manifest_path = write_manifest(samples, out_dir / "manifest.csv")
    print(f"\nMatched {len(samples)} images "
          f"({skipped_no_label} no label, {skipped_unreadable} unreadable, skipped).")
    print(f"Manifest: {manifest_path}")
    print("Per-split:", counts)


if __name__ == "__main__":
    main()
