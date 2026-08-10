"""Build zero-shot evaluation manifests from an external brain-MRI dataset,
excluding every image that overlaps this project's own training data.

Context: masoudnickparvar/brain-tumor-mri-dataset shares the same 4 classes
as ml/data/brain_mri (a rare label-compatible match) but is a compilation
that re-packages the SARTAJ set this project trained on. Measured overlap is
63.5% -- so evaluating on it naively would score mostly on memorised
training images. This script writes the deduplicated manifest that makes a
cross-dataset claim meaningful.

Emits three manifests so the leakage effect is measurable rather than merely
asserted:
  manifest_all.csv    -- everything (CONTAMINATED; reference number only)
  manifest_clean.csv  -- conservative: drops exact + near-duplicates (dHash<=N)
  manifest_strict.csv -- sensitivity check: drops only exact MD5 matches

"clean" is the headline set. It over-removes (brain MRIs collide at 8x8, so
roughly a third of the far near-dup flags are coincidental), which costs
test-set size but cannot leave leakage in -- and leakage is the error that
would inflate a generalization result. "strict" bounds the other side.

Usage:
    python ml/scripts/prepare_external_brain_mri.py \
        --root ml/data/brain_mri_external \
        --overlap-csv ml/artifacts/brain_mri_overlap.csv \
        --out-dir ml/data/brain_mri_external
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

from medchron.data.manifest import Sample, write_manifest

# External folder name -> this project's label vocabulary. A mismatch here
# would silently score the wrong classes, so the result is asserted below.
LABEL_MAP: Dict[str, str] = {
    "glioma": "glioma_tumor",
    "meningioma": "meningioma_tumor",
    "notumor": "no_tumor",
    "pituitary": "pituitary_tumor",
}
EXPECTED_LABELS = set(LABEL_MAP.values())
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _norm(p: str) -> str:
    return str(p).replace("\\", "/")


def collect(root: Path) -> List[Sample]:
    samples: List[Sample] = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        parts = [x.lower() for x in p.parts]
        label = next((LABEL_MAP[k] for k in LABEL_MAP if k in parts), None)
        if label is None:
            continue
        # split="test" for everything: nothing is ever trained on this set.
        samples.append(Sample(str(p), label, patient_id="", split="test"))
    return samples


def load_flagged(csv_path: Path, exact_only: bool) -> Set[str]:
    flagged: Set[str] = set()
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if exact_only and row["kind"] != "exact_md5":
                continue
            flagged.add(_norm(row["b_path"]))
    return flagged


def _report(name: str, samples: List[Sample], total: int) -> None:
    counts = Counter(s.label for s in samples)
    pct = 100.0 * len(samples) / max(total, 1)
    print(f"\n{name}: {len(samples)} images ({pct:.1f}% of {total})")
    for lab in sorted(counts):
        print(f"    {lab:20s} {counts[lab]:5d}")
    if samples:
        majority = max(counts.values()) / len(samples)
        print(f"    majority-class baseline: {majority * 100:.2f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="ml/data/brain_mri_external")
    ap.add_argument("--overlap-csv", default="ml/artifacts/brain_mri_overlap.csv")
    ap.add_argument("--out-dir", default="ml/data/brain_mri_external")
    args = ap.parse_args()

    root = Path(args.root)
    all_samples = collect(root)
    if not all_samples:
        raise SystemExit(f"No labelled images found under {root}")

    found = {s.label for s in all_samples}
    if found != EXPECTED_LABELS:
        raise SystemExit(
            f"Label set mismatch: got {sorted(found)}, expected {sorted(EXPECTED_LABELS)}. "
            "Evaluating with a mismatched vocabulary would score the wrong classes."
        )

    overlap = Path(args.overlap_csv)
    flagged_near = load_flagged(overlap, exact_only=False)
    flagged_exact = load_flagged(overlap, exact_only=True)

    clean = [s for s in all_samples if _norm(s.path) not in flagged_near]
    strict = [s for s in all_samples if _norm(s.path) not in flagged_exact]

    out_dir = Path(args.out_dir)
    total = len(all_samples)
    _report("ALL (contaminated)", all_samples, total)
    _report("CLEAN (exact + near removed)", clean, total)
    _report("STRICT (exact removed only)", strict, total)

    for name, rows in [("manifest_all.csv", all_samples),
                       ("manifest_clean.csv", clean),
                       ("manifest_strict.csv", strict)]:
        path = write_manifest(rows, out_dir / name)
        print(f"\nwrote {path} ({len(rows)} rows)")

    print(f"\nRemoved as overlapping training data: "
          f"{total - len(clean)} of {total} ({100.0 * (total - len(clean)) / total:.1f}%)")


if __name__ == "__main__":
    main()
