"""Cache a soft-voting ensemble's output distribution for every image in a
manifest, so a distilled student can train against it without ever loading
the teacher models.

Why cache instead of running the teacher inside the training loop: the whole
point of distillation here is to escape a memory ceiling (see
docs/ENSEMBLE_MODELS.md). Holding 3 teacher models resident *during* student
training would reintroduce exactly the footprint we're trying to avoid, and
would re-run identical forward passes every epoch for no benefit -- the
teacher is frozen, so its output for a given image never changes.

Keyed by sample path, not row order, so the cache stays correct even if the
manifest is later re-split, re-shuffled, or filtered.

The stored vector is the ensemble's *averaged probability* distribution --
the exact thing EnsemblePredictor.predict() returns -- so a student trained
on it is distilling the same teacher whose accuracy was actually measured,
not a differently-combined approximation of it.

Usage:
    python ml/scripts/cache_teacher_logits.py \
        --manifest ml/data/brain_mri/manifest.csv \
        --checkpoints ml/artifacts/brain_mri/model_efficientnet_b0.pt,ml/artifacts/brain_mri/model_resnet50.pt,ml/artifacts/brain_mri/model_densenet121.pt \
        --out ml/artifacts/brain_mri_teacher_soft.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from medchron.data import read_manifest
from medchron.models import EnsemblePredictor


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--checkpoints", required=True,
                    help="comma-separated teacher checkpoint paths (2+)")
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    args = ap.parse_args()

    ckpts = [p.strip() for p in args.checkpoints.split(",") if p.strip()]
    if len(ckpts) < 2:
        raise SystemExit("Need at least 2 teacher checkpoints for an ensemble.")

    print(f"Loading {len(ckpts)} teacher checkpoints...")
    teacher = EnsemblePredictor(ckpts, device=args.device)
    class_names = [n for n, _ in sorted(teacher.class_to_idx.items(), key=lambda kv: kv[1])]
    print(f"  classes: {class_names}")

    samples = read_manifest(args.manifest)
    print(f"{len(samples)} manifest rows")

    paths: list[str] = []
    probs: list[np.ndarray] = []
    skipped = 0
    for i, s in enumerate(samples):
        image = cv2.imread(s.path, cv2.IMREAD_COLOR)
        if image is None:
            skipped += 1
            continue
        pred = teacher.predict(image)
        # Re-order the dict into class-index order so the cached row lines up
        # with the student's logit layout (dict iteration order is not a
        # contract we should lean on here).
        vec = np.array([pred["probabilities"][c] for c in class_names], dtype=np.float32)
        paths.append(s.path)
        probs.append(vec)
        if (i + 1) % 250 == 0:
            print(f"  cached {i + 1}/{len(samples)}", flush=True)

    probs_arr = np.stack(probs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        paths=np.array(paths, dtype=object),
        probs=probs_arr,
        class_names=np.array(class_names, dtype=object),
    )

    row_sums = probs_arr.sum(axis=1)
    print(f"\nCached {len(paths)} rows ({skipped} unreadable, skipped) -> {out}")
    print(f"  prob row sums: min={row_sums.min():.6f} max={row_sums.max():.6f} (expect ~1.0)")
    print(f"  mean max-prob (teacher confidence): {probs_arr.max(axis=1).mean():.4f}")


if __name__ == "__main__":
    main()
