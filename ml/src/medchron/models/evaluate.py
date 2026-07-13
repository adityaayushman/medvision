"""Model evaluation: metrics, confusion matrix and ROC/AUC with saved plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from ..config import PreprocessConfig
from ..data.manifest import Sample
from .backbone import ModelConfig, create_model
from .dataset import build_dataloaders


@torch.no_grad()
def collect_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    """Return (y_true, y_pred, y_prob) over a loader."""
    model.eval()
    ys, preds, probs = [], [], []
    for inputs, targets in loader:
        inputs = inputs.to(device)
        logits = model(inputs)
        p = F.softmax(logits, dim=1).cpu().numpy()
        probs.append(p)
        preds.append(p.argmax(1))
        ys.append(targets.numpy())
    return (np.concatenate(ys), np.concatenate(preds), np.concatenate(probs))


def compute_metrics(y_true, y_pred, y_prob, class_names: Sequence[str]) -> Dict:
    """Accuracy, macro P/R/F1, per-class P/R/F1, and ROC-AUC."""
    n_classes = len(class_names)
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")  # e.g. a class missing from y_true

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "roc_auc": float(auc),
        "per_class": {
            class_names[i]: {
                "precision": float(p[i]), "recall": float(r[i]),
                "f1": float(f1[i]), "support": int(support[i]),
            }
            for i in range(n_classes)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).tolist(),
    }


def _save_plots(y_true, y_prob, cm, class_names, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # confusion matrix
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i][j], ha="center",
                    color="white" if cm[i][j] > np.max(cm) / 2 else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(out_dir / "confusion_matrix.png", dpi=120); plt.close(fig)

    # ROC (binary only)
    if len(class_names) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        auc = roc_auc_score(y_true, y_prob[:, 1])
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title("ROC curve"); ax.legend(loc="lower right")
        fig.tight_layout(); fig.savefig(out_dir / "roc_curve.png", dpi=120); plt.close(fig)


def evaluate_checkpoint(
    ckpt_path: str,
    samples: Sequence[Sample],
    *,
    split: str = "test",
    batch_size: int = 16,
    out_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict:
    """Load a checkpoint, evaluate on one split, save metrics.json + plots."""
    from .train import resolve_device
    device = resolve_device(device or "auto")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    mcfg = ModelConfig(**ckpt["model_config"])
    model = create_model(mcfg).to(device)
    model.load_state_dict(ckpt["state_dict"])

    class_to_idx = ckpt["class_to_idx"]
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
    preprocess = PreprocessConfig(**ckpt.get("preprocess", {}))

    bundle = build_dataloaders(samples, preprocess=preprocess, batch_size=batch_size)
    if split not in bundle.loaders:
        raise ValueError(f"No '{split}' samples to evaluate.")

    y_true, y_pred, y_prob = collect_predictions(model, bundle.loaders[split], device)
    metrics = compute_metrics(y_true, y_pred, y_prob, class_names)

    out = Path(out_dir or Path(ckpt_path).parent)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _save_plots(y_true, y_prob, np.array(metrics["confusion_matrix"]), class_names, out)

    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2))
    print(f"Saved metrics + plots -> {out}")
    return metrics
