"""Paired bootstrap test of the ROC-AUC difference between two checkpoints
evaluated on the SAME test split.

Why this exists: comparing two models by accuracy on a small test split is
underpowered. The chest X-ray full-dataset retrain differed from the
5k-subset baseline by +1.87 accuracy points at p=0.449 (not significant) on
750 images -- but by +0.0254 ROC-AUC at p<0.0001 on those same 750 images.
Accuracy is thresholded and high-variance; ROC-AUC is threshold-independent
and lower-variance, so it resolves what accuracy cannot. For a screening
tool, ranking ability independent of operating point is also the more
relevant property.

Paired design: each bootstrap iteration resamples the same test indices for
both models, so the comparison controls for which images happened to be
drawn rather than treating the two score sets as independent.

Usage:
    python ml/scripts/auc_bootstrap.py \
        --manifest ml/data/rsna_pneumonia/manifest_subset5k.csv \
        --checkpoint-a ml/artifacts/rsna_full_seed42/model_efficientnet_b0.pt \
        --checkpoint-b ml/artifacts/rsna_real/model_efficientnet_b0.pt

Reproduces (2026-08): full-data AUC 0.8507 vs subset 0.8253,
difference +0.0254, 95% CI [+0.0135, +0.0379], p < 0.0001.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from medchron.config import PreprocessConfig
from medchron.data import read_manifest
from medchron.models import ModelConfig, build_dataloaders, create_model


def predict(ckpt_path: str, samples, device, split: str):
    """Returns (y_true, probs, class_to_idx) for one checkpoint on one split."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    mcfg = ModelConfig(**ck["model_config"])
    mcfg.pretrained = False
    model = create_model(mcfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    preprocess = PreprocessConfig(**ck.get("preprocess", {}))
    bundle = build_dataloaders(samples, preprocess=preprocess, batch_size=32)
    if split not in bundle.loaders:
        raise SystemExit(f"No '{split}' split in this manifest.")

    ys, ps = [], []
    with torch.no_grad():
        for x, y in bundle.loaders[split]:
            ps.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps), ck["class_to_idx"]


def macro_auc(y, p) -> float:
    return roc_auc_score(y, p, multi_class="ovr", average="macro")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--checkpoint-a", required=True, help="model A (e.g. the new one)")
    ap.add_argument("--checkpoint-b", required=True, help="model B (e.g. the incumbent)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    samples = read_manifest(args.manifest)

    ya, pa, ca = predict(args.checkpoint_a, samples, device, args.split)
    yb, pb, cb = predict(args.checkpoint_b, samples, device, args.split)

    # Both must score the identical images in the identical class order, or
    # the "paired" comparison is meaningless.
    if ca != cb:
        raise SystemExit(f"class_to_idx mismatch: {ca} vs {cb}")
    if not np.array_equal(ya, yb):
        raise SystemExit("The two runs saw different test-set ordering; pairing is invalid.")
    y = ya

    a, b = macro_auc(y, pa), macro_auc(y, pb)
    print(f"n = {len(y)} images | classes = {list(ca)}")
    print(f"  A ({args.checkpoint_a}): AUC {a:.4f}")
    print(f"  B ({args.checkpoint_b}): AUC {b:.4f}")
    print(f"  difference (A - B): {a - b:+.4f}")

    rng = np.random.default_rng(args.seed)
    n = len(y)
    diffs = []
    for _ in range(args.resamples):
        idx = rng.integers(0, n, n)
        # A resample missing a class entirely makes macro AUC undefined; skip it.
        if len(np.unique(y[idx])) < len(ca):
            continue
        diffs.append(macro_auc(y[idx], pa[idx]) - macro_auc(y[idx], pb[idx]))

    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    pval = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    print(f"\npaired bootstrap ({len(diffs)} usable resamples):")
    print(f"  95% CI of difference: [{lo:+.4f}, {hi:+.4f}]")
    print(f"  p = {pval:.4f}  -> {'SIGNIFICANT' if pval < 0.05 else 'not significant'}")
    print(f"  CI excludes zero: {'YES' if lo > 0 or hi < 0 else 'no'}")


if __name__ == "__main__":
    main()
