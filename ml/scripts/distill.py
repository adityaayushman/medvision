"""Distill a soft-voting ensemble into a single deployable student model.

The student is the same architecture as the currently-deployed single model,
so if it recovers a useful share of the ensemble's accuracy it ships at zero
marginal memory cost -- the point being to clear a hosting memory ceiling
that more training cannot fix. See docs/ENSEMBLE_MODELS.md.

Requires a teacher cache built first:

    python ml/scripts/cache_teacher_logits.py \
        --manifest ml/data/brain_mri/manifest.csv \
        --checkpoints <effnet>.pt,<resnet>.pt,<densenet>.pt \
        --out ml/artifacts/brain_mri_teacher_soft.npz

Then:

    python ml/scripts/distill.py \
        --manifest ml/data/brain_mri/manifest.csv \
        --teacher-cache ml/artifacts/brain_mri_teacher_soft.npz \
        --modality brain_mri --seed 42 \
        --out-dir ml/artifacts/brain_mri_distilled
"""

from __future__ import annotations

import argparse

from medchron.config import PreprocessConfig, get_config
from medchron.data import read_manifest, stratified_split
from medchron.models.distill import DistillConfig, TeacherCache, distill


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--teacher-cache", required=True, help=".npz from cache_teacher_logits.py")
    ap.add_argument("--backbone", default="efficientnet_b0",
                    choices=["vgg16", "resnet50", "densenet121", "efficientnet_b0"])
    ap.add_argument("--modality", default="brain_mri", help="preprocess preset")
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.7,
                    help="weight on the teacher term; (1-alpha) goes to hard labels")
    ap.add_argument("--epochs-head", type=int, default=5)
    ap.add_argument("--epochs-finetune", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out-dir", default="ml/artifacts/brain_mri_distilled")
    args = ap.parse_args()

    samples = read_manifest(args.manifest)
    if not any(s.split for s in samples):
        samples = stratified_split(samples)

    preprocess: PreprocessConfig = get_config(args.modality)
    cache = TeacherCache(args.teacher_cache)
    cfg = DistillConfig(
        backbone=args.backbone,
        temperature=args.temperature,
        alpha=args.alpha,
        epochs_head=args.epochs_head,
        epochs_finetune=args.epochs_finetune,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        out_dir=args.out_dir,
    )
    result = distill(samples, cache, cfg, preprocess)
    print("\nDone. Checkpoint:", result["checkpoint"])


if __name__ == "__main__":
    main()
