"""Evaluate a trained checkpoint (or an ensemble of checkpoints) on the test
split of a manifest.

    python ml/scripts/evaluate.py --manifest ml/data/rsna/manifest.csv \
        --checkpoint ml/artifacts/model_vgg16.pt --split test

    # ensemble: comma-separated checkpoints, soft-voted
    python ml/scripts/evaluate.py --manifest ml/data/brain_mri/manifest.csv \
        --checkpoint ml/artifacts/brain_mri/model_efficientnet_b0.pt,ml/artifacts/brain_mri/model_resnet50.pt \
        --out-dir ml/artifacts/brain_mri_ensemble

    # localized pipeline (bbox regressor -> crop -> classifier), evaluated
    # end-to-end on FULL mammograms -- the honest, production-comparable
    # number for a classifier trained on lesion-cropped patches
    python ml/scripts/evaluate.py --manifest ml/data/mammography/cbis_prepared/manifest.csv \
        --checkpoint ml/artifacts/mammography_cbis_cropped/model_efficientnet_b0.pt \
        --bbox-checkpoint ml/artifacts/mammography_bbox/bbox_efficientnet_b0.pt \
        --out-dir ml/artifacts/mammography_localized
"""

from __future__ import annotations

import argparse

from medchron.data import read_manifest, stratified_split
from medchron.models import evaluate_checkpoint, evaluate_ensemble, evaluate_localized_pipeline


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a MedChron checkpoint")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--checkpoint", required=True,
                    help="one checkpoint path, or comma-separated paths for an ensemble")
    ap.add_argument("--bbox-checkpoint", default=None,
                    help="if set, evaluate the localized pipeline (this bbox regressor -> "
                         "crop -> --checkpoint as the classifier) instead of --checkpoint alone")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    samples = read_manifest(args.manifest)
    if not any(s.split for s in samples):
        samples = stratified_split(samples)

    if args.bbox_checkpoint:
        evaluate_localized_pipeline(
            args.bbox_checkpoint, args.checkpoint, samples,
            split=args.split, out_dir=args.out_dir, device=args.device,
        )
        return

    ckpt_paths = [p.strip() for p in args.checkpoint.split(",") if p.strip()]
    if len(ckpt_paths) > 1:
        evaluate_ensemble(
            ckpt_paths, samples, split=args.split,
            batch_size=args.batch_size, out_dir=args.out_dir, device=args.device,
        )
    else:
        evaluate_checkpoint(
            ckpt_paths[0], samples, split=args.split,
            batch_size=args.batch_size, out_dir=args.out_dir, device=args.device,
        )


if __name__ == "__main__":
    main()
