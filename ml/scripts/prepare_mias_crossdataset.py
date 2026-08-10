"""Build a MIAS Benign/Malignant manifest for zero-shot cross-dataset
evaluation of the CBIS-DDSM-trained mammography models.

Why MIAS is a legitimate cross-dataset test here: every deployed/best
mammography model in this project was trained on CBIS-DDSM, while MIAS is a
completely separate, much older database (different institutions, different
digitisation). There is no shared-source leakage risk of the kind that
plagues the Kaggle brain-tumour compilations -- so unlike the brain MRI
case, no deduplication step is required.

Drops MIAS's 'Normal' class because the CBIS classifier is 2-class
(Benign/Malignant); scoring a 3rd class it was never trained to emit would
measure nothing. Everything is marked split=test, since no training happens
against this set -- it exists purely to be evaluated on.

Small-n warning, stated here rather than discovered later: 115 images
supports "does this collapse to chance?" but not a precise accuracy claim.
Expect roughly a +/-9 point confidence interval.

Usage:
    python ml/scripts/prepare_mias_crossdataset.py \
        --manifest ml/data/mammography/prepared/manifest.csv \
        --out ml/data/mammography/mias_crossdataset_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from medchron.data.manifest import Sample, write_manifest

KEEP = {"Benign", "Malignant"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="ml/data/mammography/prepared/manifest.csv")
    ap.add_argument("--out", default="ml/data/mammography/mias_crossdataset_manifest.csv")
    args = ap.parse_args()

    with open(args.manifest, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    samples = [
        Sample(r["path"], r["label"], patient_id=r.get("patient_id", ""), split="test")
        for r in rows if r["label"] in KEEP
    ]
    missing = [s.path for s in samples if not Path(s.path).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} manifest images not found on disk (first: {missing[0]})")

    out = write_manifest(samples, args.out)
    print(f"Source rows: {len(rows)}  ->  kept {len(samples)} Benign/Malignant (dropped "
          f"{len(rows) - len(samples)} Normal)")
    print("class balance:", dict(Counter(s.label for s in samples)))
    majority = max(Counter(s.label for s in samples).values()) / len(samples)
    print(f"majority-class baseline on this set: {majority * 100:.2f}%")
    print(f"Manifest -> {out}")


if __name__ == "__main__":
    main()
