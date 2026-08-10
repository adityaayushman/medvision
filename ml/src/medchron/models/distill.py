"""Knowledge distillation: train a single small student to imitate the
soft-voting ensemble's output distribution.

Why this module exists: the 3-way brain MRI ensemble is measurably better
than the deployed single model (88.12% vs 82.04% accuracy, 5 seeds each),
but its ~583MB marginal footprint is over the entire 512MB production
ceiling, and two live deploy attempts crash-looped because of it (see
docs/ENSEMBLE_MODELS.md). Improving the ensemble further doesn't help --
the blocker is memory, not accuracy. Distillation attacks the actual
blocker: the student is the SAME architecture and size as the model already
deployed, so any accuracy it recovers from the teacher ships at zero
marginal memory cost.

Deliberately mirrors train.py's two-phase schedule, early stopping, and
checkpoint format rather than inventing a parallel training path -- the
resulting checkpoint is loadable by the unmodified Predictor and
evaluate_checkpoint, which is what makes "drop-in replacement" a fact
rather than a claim. The only genuinely new piece is the loss.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..config import PreprocessConfig
from ..data.manifest import Sample
from .backbone import (
    ModelConfig,
    create_model,
    freeze_backbone,
    trainable_parameter_count,
    unfreeze_top_fraction,
)
from .dataset import MedChronDataset, build_class_index, class_weights
from .train import resolve_device, set_seed


@dataclass
class DistillConfig:
    backbone: str = "efficientnet_b0"
    temperature: float = 4.0
    alpha: float = 0.7          # weight on the soft (teacher) term; 1-alpha on hard labels
    epochs_head: int = 5
    epochs_finetune: int = 10
    lr_head: float = 1e-3
    lr_finetune: float = 1e-5
    finetune_fraction: float = 0.25
    batch_size: int = 16
    weight_decay: float = 1e-4
    patience: int = 5
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"
    out_dir: str = "ml/artifacts/brain_mri_distilled"


class TeacherCache:
    """Path -> teacher probability vector, loaded from the .npz written by
    ml/scripts/cache_teacher_logits.py. Keyed by path so it survives any
    later re-split or re-shuffle of the manifest."""

    def __init__(self, npz_path: str) -> None:
        data = np.load(npz_path, allow_pickle=True)
        self.class_names: List[str] = [str(c) for c in data["class_names"]]
        probs = data["probs"].astype(np.float32)
        self.by_path: Dict[str, np.ndarray] = {
            str(p): probs[i] for i, p in enumerate(data["paths"])
        }

    def __contains__(self, path: str) -> bool:
        return path in self.by_path

    def get(self, path: str) -> np.ndarray:
        return self.by_path[path]


class _TeacherDataset(Dataset):
    """Wraps MedChronDataset so each item also carries its teacher
    distribution. Composition rather than a subclass: MedChronDataset's
    preprocessing/augmentation is exactly what the student must see, and
    reimplementing it here would be a silent drift risk."""

    def __init__(self, base: MedChronDataset, cache: TeacherCache) -> None:
        self.base = base
        self.cache = cache

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        tensor, target = self.base[idx]
        teacher = torch.from_numpy(self.cache.get(self.base.samples[idx].path))
        return tensor, target, teacher


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    hard_target: torch.Tensor,
    *,
    temperature: float,
    alpha: float,
    hard_criterion: nn.Module,
) -> torch.Tensor:
    """Hinton-style KD.

    The teacher term is KL(student_T || teacher_T), scaled by T^2 so its
    gradient magnitude stays comparable to the hard-label term as T varies.

    Note on the teacher side: EnsemblePredictor soft-votes by averaging
    *probabilities*, so what's cached is a distribution, not logits. Taking
    log(p) recovers logits up to an additive constant, and softmax is
    shift-invariant, so softmax(log(p)/T) is exactly correct temperature
    scaling of that distribution. This keeps the student distilling the same
    prob-averaged teacher whose 88.12% accuracy was actually measured,
    rather than silently switching to logit-averaging (a different ensemble
    with a different, unmeasured accuracy).
    """
    teacher_logits = torch.log(teacher_probs.clamp_min(1e-8))
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)

    hard_loss = hard_criterion(student_logits, hard_target)
    return alpha * soft_loss + (1.0 - alpha) * hard_loss


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: DistillConfig,
    hard_criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for inputs, targets, teacher in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            teacher = teacher.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = distillation_loss(
                logits, teacher, targets,
                temperature=cfg.temperature, alpha=cfg.alpha,
                hard_criterion=hard_criterion,
            )
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            correct += (logits.argmax(1) == targets).sum().item()
            seen += inputs.size(0)
    return total_loss / max(seen, 1), correct / max(seen, 1)


def _run_phase(
    name: str,
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    device: torch.device,
    cfg: DistillConfig,
    hard_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    history: List[dict],
    scheduler=None,
) -> None:
    best, bad, best_state = float("inf"), 0, None
    has_val = "val" in loaders
    for epoch in range(epochs):
        tr_loss, tr_acc = _run_epoch(model, loaders["train"], device, cfg, hard_criterion, optimizer)
        if has_val:
            va_loss, va_acc = _run_epoch(model, loaders["val"], device, cfg, hard_criterion)
        else:
            va_loss, va_acc = tr_loss, tr_acc
        if scheduler is not None:
            scheduler.step(va_loss)

        history.append({
            "phase": name, "epoch": epoch + 1,
            "train_loss": round(tr_loss, 4), "train_acc": round(tr_acc, 4),
            "val_loss": round(va_loss, 4), "val_acc": round(va_acc, 4),
            "lr": optimizer.param_groups[0]["lr"],
        })
        print(f"[{name}] epoch {epoch + 1:02d}/{epochs}  "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.3f}  "
              f"val_loss={va_loss:.4f} acc={va_acc:.3f}", flush=True)

        if va_loss < best - 1e-4:
            best, bad = va_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if has_val and bad >= cfg.patience:
                print(f"[{name}] early stopping at epoch {epoch + 1}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)


def distill(
    samples: Sequence[Sample],
    teacher_cache: TeacherCache,
    cfg: Optional[DistillConfig] = None,
    preprocess: Optional[PreprocessConfig] = None,
) -> dict:
    cfg = cfg or DistillConfig()
    preprocess = preprocess or PreprocessConfig()
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    missing = [s.path for s in samples if s.path not in teacher_cache]
    if missing:
        raise ValueError(
            f"{len(missing)} manifest samples have no cached teacher output "
            f"(first: {missing[0]!r}). Re-run cache_teacher_logits.py against "
            "this same manifest."
        )

    class_to_idx = build_class_index(samples)
    # The student's output layer must line up with the cached teacher vector,
    # which was written in the teacher's own class-index order.
    if list(teacher_cache.class_names) != [n for n, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]:
        raise ValueError(
            f"Teacher class order {teacher_cache.class_names} != student class order "
            f"{sorted(class_to_idx, key=class_to_idx.get)} — distilling would map "
            "probabilities onto the wrong classes."
        )

    by_split: Dict[str, List[Sample]] = {}
    for s in samples:
        by_split.setdefault(s.split, []).append(s)

    loaders: Dict[str, DataLoader] = {}
    for split, items in by_split.items():
        if split not in ("train", "val", "test") or not items:
            continue
        base = MedChronDataset(items, class_to_idx, preprocess, train=(split == "train"))
        loaders[split] = DataLoader(
            _TeacherDataset(base, teacher_cache),
            batch_size=cfg.batch_size, shuffle=(split == "train"),
            num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
        )
    if "train" not in loaders:
        raise ValueError("No training samples found (check the manifest's 'split' column).")

    mcfg = ModelConfig(backbone=cfg.backbone, num_classes=len(class_to_idx))
    model = create_model(mcfg).to(device)
    weights = class_weights(by_split.get("train") or list(samples), class_to_idx)
    hard_criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    print(f"Device: {device} | student: {cfg.backbone} | T={cfg.temperature} alpha={cfg.alpha} | "
          f"classes: {class_to_idx}", flush=True)

    history: List[dict] = []

    freeze_backbone(model, cfg.backbone)
    print(f"Phase 1 trainable params: {trainable_parameter_count(model):,}", flush=True)
    opt1 = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.lr_head, weight_decay=cfg.weight_decay,
    )
    _run_phase("head", model, loaders, device, cfg, hard_criterion, opt1, cfg.epochs_head, history)

    unfreeze_top_fraction(model, cfg.finetune_fraction)
    print(f"Phase 2 trainable params: {trainable_parameter_count(model):,}", flush=True)
    opt2 = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.lr_finetune, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt2, mode="min", factor=0.5, patience=2)
    _run_phase("finetune", model, loaders, device, cfg, hard_criterion, opt2,
               cfg.epochs_finetune, history, scheduler=scheduler)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"model_{cfg.backbone}.pt"
    # Same key set train.py writes -- Predictor/evaluate_checkpoint and the
    # backend load this with no special-casing. "train_config" carries the
    # distillation settings so a checkpoint is self-describing.
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_to_idx": class_to_idx,
            "model_config": asdict(mcfg),
            "train_config": asdict(cfg),
            "task": "multiclass",
            "preprocess": preprocess.to_dict(),
            "history": history,
            "distilled_from": teacher_cache.class_names and "3-way soft-voting ensemble",
        },
        ckpt_path,
    )
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Saved distilled checkpoint -> {ckpt_path}", flush=True)
    return {"checkpoint": str(ckpt_path), "class_to_idx": class_to_idx, "history": history}
