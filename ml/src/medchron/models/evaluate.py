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
    """Return (y_true, y_pred, y_prob) over a loader (multiclass)."""
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


@torch.no_grad()
def collect_predictions_multilabel(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    """Return (y_true, y_prob) over a loader — y_true is multi-hot, y_prob is per-class sigmoid."""
    model.eval()
    ys, probs = [], []
    for inputs, targets in loader:
        inputs = inputs.to(device)
        logits = model(inputs)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(targets.numpy())
    return np.concatenate(ys), np.concatenate(probs)


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


def compute_multilabel_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, class_names: Sequence[str], threshold: float = 0.5
) -> Dict:
    """Per-class precision/recall/F1/AUC, mean AUC, exact-match and Hamming accuracy.

    Accuracy is not a meaningful single number for multi-label — "exact match"
    (every finding correct) is a harsh lower bound, "Hamming" (per-label
    correctness) is more forgiving; both are reported alongside per-class AUC,
    which is the standard NIH ChestX-ray14 benchmark metric.
    """
    y_true_int = y_true.astype(int)
    y_pred = (y_prob >= threshold).astype(int)

    per_class: Dict[str, Dict] = {}
    aucs = []
    for i, name in enumerate(class_names):
        yt, yp, ypr = y_true_int[:, i], y_pred[:, i], y_prob[:, i]
        p, r, f1, _ = precision_recall_fscore_support(
            yt, yp, average="binary", zero_division=0
        )
        try:
            auc = roc_auc_score(yt, ypr) if len(set(yt)) > 1 else float("nan")
        except ValueError:
            auc = float("nan")
        if not np.isnan(auc):
            aucs.append(auc)
        per_class[name] = {
            "precision": float(p), "recall": float(r), "f1": float(f1),
            "support": int(yt.sum()), "auc": float(auc),
        }

    return {
        "mean_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "exact_match_accuracy": float((y_pred == y_true_int).all(axis=1).mean()),
        "hamming_accuracy": float((y_pred == y_true_int).mean()),
        "threshold": threshold,
        "per_class": per_class,
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
    mcfg.pretrained = False  # weights come from the checkpoint
    model = create_model(mcfg).to(device)
    model.load_state_dict(ckpt["state_dict"])

    class_to_idx = ckpt["class_to_idx"]
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
    preprocess = PreprocessConfig(**ckpt.get("preprocess", {}))
    task = ckpt.get("task", "multiclass")
    label_delim = ckpt.get("train_config", {}).get("label_delim", "|")

    bundle = build_dataloaders(
        samples, preprocess=preprocess, batch_size=batch_size, task=task, label_delim=label_delim
    )
    if split not in bundle.loaders:
        raise ValueError(f"No '{split}' samples to evaluate.")

    out = Path(out_dir or Path(ckpt_path).parent)
    out.mkdir(parents=True, exist_ok=True)

    if task == "multilabel":
        y_true, y_prob = collect_predictions_multilabel(model, bundle.loaders[split], device)
        metrics = compute_multilabel_metrics(y_true, y_prob, class_names)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps({k: v for k, v in metrics.items() if k != "per_class"}, indent=2))
        print(f"Saved metrics -> {out}")
        return metrics

    y_true, y_pred, y_prob = collect_predictions(model, bundle.loaders[split], device)
    metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _save_plots(y_true, y_prob, np.array(metrics["confusion_matrix"]), class_names, out)

    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2))
    print(f"Saved metrics + plots -> {out}")
    return metrics
