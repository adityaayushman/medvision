"""Seed the live backend's Patients + Records with real RSNA dataset cases.

Picks a stratified sample of patients from the held-out *test* split (never
seen during training/model-selection — the same split reported on the
Evaluation page), pulls each patient's real (de-identified) age/sex from the
source DICOM header, creates a Patient, and runs their prepared PNG through
POST /api/analyze against the live backend — so each Study gets a genuine
timestamp, quality report, ROI/Grad-CAM images and model prediction, exactly
like a scan uploaded through the UI.

Usage:
    python ml/scripts/seed_patients_from_dataset.py --per-class 6
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pydicom
import requests


def _post_with_retry(url: str, *, tries: int = 4, backoff: float = 3.0, **kwargs) -> requests.Response:
    """Render's free tier throws transient 502s under load/cold-start; retry those."""
    last_exc: Exception | None = None
    for attempt in range(1, tries + 1):
        try:
            r = requests.post(url, **kwargs)
            if r.status_code == 502 and attempt < tries:
                time.sleep(backoff * attempt)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last_exc = e
            if attempt < tries:
                time.sleep(backoff * attempt)
    raise last_exc 

RAW_DICOM_DIR = Path("ml/data/rsna_pneumonia/drive-download-20240112T131344Z-002/stage_2_train_images")
MANIFEST = Path("ml/data/rsna_pneumonia/prepared/manifest.csv")

SEX_MAP = {"M": "M", "F": "F"}
THIS_YEAR = datetime.now(timezone.utc).year


def load_rows(manifest: Path, split: str) -> dict[str, list[dict]]:
    by_label: dict[str, list[dict]] = defaultdict(list)
    with open(manifest, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["split"] == split:
                by_label[row["label"]].append(row)
    return by_label


def dicom_demographics(patient_id: str) -> tuple[str | None, int | None]:
    dcm_path = RAW_DICOM_DIR / f"{patient_id}.dcm"
    if not dcm_path.exists():
        return None, None
    ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
    sex = SEX_MAP.get(str(ds.get("PatientSex", "")).strip())
    age_raw = str(ds.get("PatientAge", "")).strip()
    birth_year = None
    if age_raw:
        digits = "".join(ch for ch in age_raw if ch.isdigit())
        if digits:
            birth_year = THIS_YEAR - int(digits)
    return sex, birth_year


def main() -> None:
    ap = argparse.ArgumentParser(description="Seed live patients + analyzed records from the RSNA test split")
    ap.add_argument("--backend", default="https://medchron-backend.onrender.com")
    ap.add_argument("--per-class", type=int, default=6, help="patients per class (3 classes)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only-patient-ids", default=None,
                     help="comma-separated patient_ids to (re)seed, bypassing random sampling")
    args = ap.parse_args()

    random.seed(args.seed)
    by_label = load_rows(MANIFEST, args.split)
    for label, rows in by_label.items():
        print(f"  {label}: {len(rows)} available in '{args.split}' split")

    if args.only_patient_ids:
        wanted = set(args.only_patient_ids.split(","))
        sample = [r for rows in by_label.values() for r in rows if r["patient_id"] in wanted]
    else:
        sample = []
        for label, rows in by_label.items():
            sample.extend(random.sample(rows, min(args.per_class, len(rows))))
        random.shuffle(sample)
    print(f"\nSeeding {len(sample)} patients (backend: {args.backend})\n")

    ok = 0
    for i, row in enumerate(sample, 1):
        patient_id = row["patient_id"]
        label = row["label"]
        img_path = Path(row["path"])
        if not img_path.exists():
            print(f"[{i}/{len(sample)}] SKIP {patient_id}: image missing at {img_path}")
            continue

        sex, birth_year = dicom_demographics(patient_id)
        display_name = f"RSNA Case {patient_id[:8]}"

        try:
            r = _post_with_retry(
                f"{args.backend}/api/patients",
                json={"name": display_name, "sex": sex, "birth_year": birth_year},
                timeout=30,
            )
            pid = r.json()["id"]

            with open(img_path, "rb") as fh:
                r = _post_with_retry(
                    f"{args.backend}/api/analyze",
                    files={"file": (img_path.name, fh, "image/png")},
                    data={"patient_id": str(pid)},
                    timeout=120,
                )
            result = r.json()
            pred = result.get("prediction") or {}
            ok += 1
            print(
                f"[{i}/{len(sample)}] {display_name} "
                f"(true={label}, sex={sex}, birth_year={birth_year}) "
                f"-> predicted={pred.get('label', 'n/a')} "
                f"conf={pred.get('confidence', 0):.2f} study_id={result.get('study_id')}"
            )
        except requests.RequestException as e:
            print(f"[{i}/{len(sample)}] FAILED {patient_id}: {e}")

        time.sleep(0.3)  

    print(f"\nDone: {ok}/{len(sample)} patients seeded with a real analyzed scan.")


if __name__ == "__main__":
    main()
