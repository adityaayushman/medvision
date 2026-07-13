"""Train a MedChron classifier from a manifest (or straight from a folder).

Examples
--------
# from a prepared manifest (has train/val/test split column)
python ml/scripts/train.py --manifest ml/data/rsna/manifest.csv --backbone vgg16

# or straight from an ImageFolder-style directory (build + split on the fly)
python ml/scripts/train.py --root ml/data/rsna --epochs-finetune 15
"""

from __future__ import annotations

import argparse

from medchron.config import PreprocessConfig, get_config
from medchron.data import (
    build_manifest_from_folders,
    read_manifest,
    stratified_split,
)
from medchron.data.split import assert_no_patient_leakage
from medchron.models import TrainConfig, train


def load_samples(args):
    if args.manifest:
        samples = read_manifest(args.manifest)
        if not any(s.split for s in samples):
            samples = stratified_split(samples)
        return samples
    samples = build_manifest_from_folders(args.root)
    if not samples:
        raise SystemExit(f"No images under {args.root!r} (expected root/<class>/<images>)")
    samples = stratified_split(samples)
    assert_no_patient_leakage(samples)
    return samples


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a MedChron classifier")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="path to a manifest CSV")
    src.add_argument("--root", help="ImageFolder-style dataset root")

    ap.add_argument("--backbone", default="vgg16",
                    choices=["vgg16", "resnet50", "densenet121", "efficientnet_b0"])
    ap.add_argument("--task", default="multiclass", choices=["multiclass", "multilabel"],
                    help="multiclass: one label/image (e.g. RSNA). "
                         "multilabel: several findings/image, '|'-delimited (e.g. NIH ChestX-ray14)")
    ap.add_argument("--modality", default="chest_xray", help="preprocess preset")
    ap.add_argument("--epochs-head", type=int, default=5)
    ap.add_argument("--epochs-finetune", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-finetune", type=float, default=1e-5)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--resume", action="store_true",
                    help="resume from the last saved epoch in --out-dir (safe after a shutdown)")
    ap.add_argument("--out-dir", default="ml/artifacts")
    args = ap.parse_args()

    samples = load_samples(args)
    preprocess: PreprocessConfig = get_config(args.modality)
    tcfg = TrainConfig(
        backbone=args.backbone,
        task=args.task,
        epochs_head=args.epochs_head,
        epochs_finetune=args.epochs_finetune,
        batch_size=args.batch_size,
        lr_head=args.lr_head,
        lr_finetune=args.lr_finetune,
        num_workers=args.num_workers,
        device=args.device,
        out_dir=args.out_dir,
    )
    result = train(samples, tcfg, preprocess, resume=args.resume)
    print("\nDone. Checkpoint:", result["checkpoint"])


if __name__ == "__main__":
    main()
