"""Build a MedChron manifest from CBIS-DDSM (calc + mass cases, JPEG mirror).

CBIS-DDSM ships three separate signals that have to be joined:
  * calc/mass_case_description_{train,test}_set.csv — patient_id, side, view,
    pathology (the label)
  * dicom_info.csv — maps a DICOM SeriesInstanceUID directory to a
    human-readable PatientID string ("Calc-Training_P_00232_RIGHT_CC") and a
    SeriesDescription ("full mammogram images" | "cropped images" |
    "ROI mask images")

Only "full mammogram images" are used (not the cropped lesion patches or ROI
masks) — the same whole-image classification task as the other modalities.

CBIS-DDSM has no "Normal" cases (every case has an annotated abnormality), so
unlike MIAS this is a 2-class task: Benign vs Malignant.
BENIGN_WITHOUT_CALLBACK is folded into Benign. A handful of images have
multiple abnormalities with different severities; the most severe one wins
(if any abnormality on the image is malignant, the image is labeled
Malignant) — standard practice for whole-image CBIS-DDSM classification.

Usage:
    python ml/scripts/prepare_cbis_ddsm.py \
        --raw-dir ml/data/mammography/cbis_raw \
        --out-dir ml/data/mammography/cbis_prepared
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from medchron.data import class_distribution, split_distribution, stratified_split, write_manifest
from medchron.data.manifest import Sample
from medchron.data.split import assert_no_patient_leakage

PATHOLOGY_MAP = {
    "MALIGNANT": "Malignant",
    "BENIGN": "Benign",
    "BENIGN_WITHOUT_CALLBACK": "Benign",
}
SEVERITY_RANK = {"Benign": 0, "Malignant": 1}


def load_case_labels(raw_dir: Path) -> dict[str, str]:
    """join_key ('Calc-Training_P_00232_RIGHT_CC') -> label, most-severe-wins."""
    labels: dict[str, str] = {}
    csvs = [
        ("Calc", "Training", "calc_case_description_train_set.csv"),
        ("Calc", "Test", "calc_case_description_test_set.csv"),
        ("Mass", "Training", "mass_case_description_train_set.csv"),
        ("Mass", "Test", "mass_case_description_test_set.csv"),
    ]
    metadata_dir = raw_dir / "csv" if (raw_dir / "csv").exists() else raw_dir
    for abn_type, split_name, fname in csvs:
        path = metadata_dir / fname
        if not path.exists():
            print(f"  (skip, not found) {fname}")
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            key = f"{abn_type}-{split_name}_{row['patient_id']}_{row['left or right breast']}_{row['image view']}"
            label = PATHOLOGY_MAP.get(str(row["pathology"]).strip())
            if label is None:
                continue
            if key not in labels or SEVERITY_RANK[label] > SEVERITY_RANK[labels[key]]:
                labels[key] = label
    return labels


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a MedChron manifest from CBIS-DDSM")
    ap.add_argument("--raw-dir", required=True, help="folder with csv/ and jpeg/ from the Kaggle download")
    ap.add_argument("--out-dir", default="ml/data/mammography/cbis_prepared")
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--test-size", type=float, default=0.15)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading case labels...")
    labels = load_case_labels(raw_dir)
    print(f"  {len(labels)} labeled cases (Benign+Malignant, deduped by image)")

    print("Loading dicom_info.csv...")
    metadata_dir = raw_dir / "csv" if (raw_dir / "csv").exists() else raw_dir
    dinfo = pd.read_csv(metadata_dir / "dicom_info.csv")
    full = dinfo[dinfo["SeriesDescription"] == "full mammogram images"]
    print(f"  {len(full)} 'full mammogram images' rows")

    samples: list[Sample] = []
    missing_file = 0
    unlabeled = 0
    for _, row in full.iterrows():
        key = row["PatientID"]
        label = labels.get(key)
        if label is None:
            unlabeled += 1
            continue
       
        rel = str(row["image_path"]).split("CBIS-DDSM/", 1)[-1]
        img_path = raw_dir / rel
        if not img_path.exists():
            missing_file += 1
            continue
        
        patient_id = key.split("_P_")[-1].split("_")[0]
        samples.append(Sample(str(img_path), label, patient_id=f"P_{patient_id}"))

    print(f"Matched {len(samples)} images ({unlabeled} unlabeled, {missing_file} missing files).")

    samples = stratified_split(samples, val_size=args.val_size, test_size=args.test_size)
    assert_no_patient_leakage(samples)

    manifest_path = write_manifest(samples, out_dir / "manifest.csv")
    print(f"\nManifest: {manifest_path} ({len(samples)} samples)")
    print("Class distribution:", json.dumps(class_distribution(samples)))
    print("Per-split:", json.dumps(split_distribution(samples), indent=2))


if __name__ == "__main__":
    main()
