"""Evaluate a trained checkpoint on the test split of a manifest.

    python ml/scripts/evaluate.py --manifest ml/data/rsna/manifest.csv \
        --checkpoint ml/artifacts/model_vgg16.pt --split test
"""

from __future__ import annotations

import argparse

from medchron.data import read_manifest, stratified_split
from medchron.models import evaluate_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a MedChron checkpoint")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    samples = read_manifest(args.manifest)
    if not any(s.split for s in samples):
        samples = stratified_split(samples)

    evaluate_checkpoint(
        args.checkpoint, samples, split=args.split,
        batch_size=args.batch_size, out_dir=args.out_dir, device=args.device,
    )


if __name__ == "__main__":
    main()
