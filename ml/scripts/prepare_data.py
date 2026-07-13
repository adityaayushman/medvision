"""Build a leakage-safe train/val/test manifest from a dataset folder.

Expected layout (ImageFolder-style):

    <root>/<class_name>/<image files...>

Usage:
    python ml/scripts/prepare_data.py --root ml/data/rsna --out ml/data/rsna/manifest.csv
    python ml/scripts/prepare_data.py --list        # show the dataset registry
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from medchron.data import (
    REGISTRY,
    build_manifest_from_folders,
    class_distribution,
    recommended_v1,
    split_distribution,
    stratified_split,
    write_manifest,
)
from medchron.data.split import assert_no_patient_leakage


def _list_registry() -> None:
    print("Dataset registry:\n")
    for spec in REGISTRY.values():
        roi = "ROI+" if spec.roi_support else "ROI-"
        print(f"  {spec.key:22s} {spec.modality:14s} {spec.task:13s} "
              f"{spec.access:12s} {roi:5s} {spec.approx_images:>7s}")
        print(f"      {spec.name}")
        print(f"      {spec.url}")
    rec = ", ".join(s.key for s in recommended_v1())
    print(f"\nRecommended V1 pairing: {rec}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare a MedChron dataset manifest")
    ap.add_argument("--list", action="store_true", help="list the dataset registry and exit")
    ap.add_argument("--root", help="dataset root (root/<class>/<images>)")
    ap.add_argument("--out", default="ml/data/manifest.csv")
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.list or not args.root:
        _list_registry()
        return

    samples = build_manifest_from_folders(args.root)
    if not samples:
        raise SystemExit(f"No images found under {args.root!r} (expected root/<class>/<images>)")

    samples = stratified_split(
        samples, val_size=args.val_size, test_size=args.test_size, seed=args.seed
    )
    assert_no_patient_leakage(samples)
    out = write_manifest(samples, args.out)

    print(f"Manifest: {out}  ({len(samples)} samples)")
    print("Class distribution:", json.dumps(class_distribution(samples)))
    print("Per-split:", json.dumps(split_distribution(samples), indent=2))


if __name__ == "__main__":
    main()
