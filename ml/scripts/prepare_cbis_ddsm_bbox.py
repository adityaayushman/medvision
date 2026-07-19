"""Build a lesion bounding-box manifest from CBIS-DDSM's ROI mask images.

Joins dicom_info.csv's "ROI mask images" rows against the *existing*
full-image manifest (built by prepare_cbis_ddsm.py, default --series full)
so the bbox regressor's split assignment is identical to the classifier's
already patient-safe split -- the two are directly comparable later.

For each ROI mask: threshold to find the nonzero region, normalize the
bounding box by the mask's own width/height. Masks are pixel-aligned with
their full mammogram for the large majority of cases (verified: ~97.6%);
the rest (dimension mismatch between mask and full image) are dropped as a
known CBIS-DDSM data-quality issue. A full image with multiple abnormality
masks gets one unioned box covering all of them.

Usage:
    python ml/scripts/prepare_cbis_ddsm_bbox.py \
        --raw-dir ml/data/mammography/cbis_raw \
        --manifest ml/data/mammography/cbis_prepared/manifest.csv \
        --out-dir ml/data/mammography/cbis_bbox_prepared
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2
import pandas as pd


def load_full_image_paths(raw_dir: Path, dinfo: pd.DataFrame) -> dict[str, str]:
    """join_key -> full mammogram path, using the same construction as
    prepare_cbis_ddsm.py so the strings match the existing manifest."""
    full = dinfo[dinfo["SeriesDescription"] == "full mammogram images"]
    out: dict[str, str] = {}
    for _, row in full.iterrows():
        rel = str(row["image_path"]).split("CBIS-DDSM/", 1)[-1]
        out[row["PatientID"]] = str(raw_dir / rel)
    return out


def mask_bbox(mask_path: Path) -> tuple[float, float, float, float] | None:
    """Returns normalized (cx, cy, w, h) of the nonzero region, or None."""
    img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    ys, xs = (img > 10).nonzero()
    if len(xs) == 0:
        return None
    h_img, w_img = img.shape[:2]
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    cx = (x0 + x1) / 2 / w_img
    cy = (y0 + y1) / 2 / h_img
    w = (x1 - x0) / w_img
    h = (y1 - y0) / h_img
    return cx, cy, w, h


def union_boxes(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    x0 = min(b[0] - b[2] / 2 for b in boxes)
    x1 = max(b[0] + b[2] / 2 for b in boxes)
    y0 = min(b[1] - b[3] / 2 for b in boxes)
    y1 = max(b[1] + b[3] / 2 for b in boxes)
    return (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a bbox manifest from CBIS-DDSM ROI masks")
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--manifest", required=True, help="existing full-image manifest.csv (path/label/patient_id/split)")
    ap.add_argument("--out-dir", default="ml/data/mammography/cbis_bbox_prepared")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading existing full-image manifest (for split reuse)...")
    path_to_split: dict[str, str] = {}
    with open(args.manifest, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            path_to_split[row["path"]] = row["split"]
    print(f"  {len(path_to_split)} full-image rows")

    print("Loading dicom_info.csv...")
    metadata_dir = raw_dir / "csv" if (raw_dir / "csv").exists() else raw_dir
    dinfo = pd.read_csv(metadata_dir / "dicom_info.csv")
    full_paths = load_full_image_paths(raw_dir, dinfo)

    full_dims = {
        row["PatientID"]: (row["Rows"], row["Columns"])
        for _, row in dinfo[dinfo["SeriesDescription"] == "full mammogram images"].iterrows()
    }
    roi = dinfo[dinfo["SeriesDescription"] == "ROI mask images"]
    print(f"  {len(roi)} 'ROI mask images' rows")

    per_image_boxes: dict[str, list[tuple[float, float, float, float]]] = {}
    dim_mismatch = 0
    no_full_match = 0
    unreadable = 0
    for _, row in roi.iterrows():
        raw_key = row["PatientID"]
        key = re.sub(r"_\d+$", "", raw_key)
        full_path = full_paths.get(key)
        if full_path is None or full_path not in path_to_split:
            no_full_match += 1
            continue

        full_rows, full_cols = full_dims.get(key, (None, None))
        if (row["Rows"], row["Columns"]) != (full_rows, full_cols):
            dim_mismatch += 1
            continue

        rel = str(row["image_path"]).split("CBIS-DDSM/", 1)[-1]
        box = mask_bbox(raw_dir / rel)
        if box is None:
            unreadable += 1
            continue
        per_image_boxes.setdefault(full_path, []).append(box)

    print(f"Matched {len(per_image_boxes)} full images with a usable box "
          f"({dim_mismatch} dim mismatches, {no_full_match} no full-image match, "
          f"{unreadable} unreadable masks skipped).")

    out_path = out_dir / "bbox_manifest.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "cx", "cy", "w", "h", "split"])
        writer.writeheader()
        counts = {"train": 0, "val": 0, "test": 0}
        for path, boxes in per_image_boxes.items():
            cx, cy, w, h = union_boxes(boxes) if len(boxes) > 1 else boxes[0]
            split = path_to_split[path]
            writer.writerow({"path": path, "cx": cx, "cy": cy, "w": w, "h": h, "split": split})
            counts[split] = counts.get(split, 0) + 1

    print(f"\nBbox manifest: {out_path} ({len(per_image_boxes)} samples)")
    print("Per-split:", counts)


if __name__ == "__main__":
    main()
