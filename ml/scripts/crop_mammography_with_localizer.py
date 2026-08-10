"""Materialize training crops using the trained LOCALIZER's own predicted
boxes -- as opposed to crop_mammography_from_masks.py, which crops with
ground-truth boxes.

Why: crop_mammography_from_masks.py already tested "does matching
LocalizedPredictor's exact framing/padding convention help" in isolation
(cbis_gtcrop_prepared, 62.5% acc standalone -- still below the 71.1%
official-crop classifier, and its full pipeline result, 53.0%, is still
below baseline). That leaves the other half of the diagnosis untested: the
classifier has never been trained on a real localizer's noisy, imperfectly-
centered crops, only ever on ground-truth ones. This script closes that gap
by running the trained segmentation localizer's predict_box() over every
image and cropping from ITS predictions, not the annotation.

Reuses medchron.models.inference._load_localizer -- the exact dispatch
LocalizedPredictor uses at inference time (segmentation vs bbox regression,
selected by the checkpoint's "task" field) -- so the crops here are produced
by literally the same code path production inference will use, not a
reimplementation of it.

Usage:
    python ml/scripts/crop_mammography_with_localizer.py \
        --classification-manifest ml/data/mammography/cbis_prepared/manifest.csv \
        --localizer-checkpoint ml/artifacts/mammography_segmentation/unet_efficientnet_b0.pt \
        --out-dir ml/data/mammography/cbis_autocrop_prepared
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from medchron.data.manifest import Sample, write_manifest
from medchron.models.detect import crop_to_bbox
from medchron.models.inference import _load_localizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classification-manifest", required=True)
    ap.add_argument("--localizer-checkpoint", required=True)
    ap.add_argument("--out-dir", default="ml/data/mammography/cbis_autocrop_prepared")
    ap.add_argument("--pad-frac", type=float, default=0.15,
                     help="must match LocalizedPredictor's pad_frac default (0.15)")
    ap.add_argument("--device", default=None, help="e.g. cuda, cpu (default: auto)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading localizer checkpoint {args.localizer_checkpoint}...")
    localizer = _load_localizer(args.localizer_checkpoint, device=args.device)

    print("Loading classification manifest (labels)...")
    with open(args.classification_manifest, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"  {len(rows)} labeled full-image rows")

    samples: list[Sample] = []
    skipped_unreadable = 0
    counts = {"train": 0, "val": 0, "test": 0}
    for i, row in enumerate(rows):
        path = row["path"]
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            skipped_unreadable += 1
            continue

        cx, cy, w, h = localizer.predict_box(image)
        crop, _box = crop_to_bbox(image, cx, cy, w, h, pad_frac=args.pad_frac)

        split = row["split"]
        out_path = images_dir / f"{i}_{split}.jpg"
        cv2.imwrite(str(out_path), crop)

        samples.append(Sample(str(out_path), row["label"], patient_id=row["patient_id"], split=split))
        counts[split] = counts.get(split, 0) + 1

        if (i + 1) % 250 == 0:
            print(f"  cropped {i + 1}/{len(rows)}", flush=True)

    manifest_path = write_manifest(samples, out_dir / "manifest.csv")
    print(f"\nCropped {len(samples)} images ({skipped_unreadable} unreadable, skipped).")
    print(f"Manifest: {manifest_path}")
    print("Per-split:", counts)


if __name__ == "__main__":
    main()
