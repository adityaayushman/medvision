"""Two-phase transfer-learning trainer (PyTorch).

Phase 1 trains only the new head (backbone frozen); phase 2 fine-tunes the top of
the network at a low learning rate. Includes class-weighted loss, mixed precision
(for the 6 GB GPU), ``ReduceLROnPlateau``, early stopping, and best-checkpointing.
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import PreprocessConfig
from ..data.manifest import Sample
from .backbone import (
    ModelConfig,
    create_model,
    freeze_backbone,
    trainable_parameter_count,
    unfreeze_top_fraction,
)
from .dataset import build_dataloaders


@dataclass
class TrainConfig:
    backbone: str = "vgg16"
    epochs_head: int = 5
    epochs_finetune: int = 10
    lr_head: float = 1e-3
    lr_finetune: float = 1e-5
    finetune_fraction: float = 0.25
    batch_size: int = 16
    weight_decay: float = 1e-4
    patience: int = 5
    amp: bool = True
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"          # "auto" | "cpu" | "cuda"
    out_dir: str = "ml/artifacts"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(pref: str = "auto") -> torch.device:
    """Pick a device honouring the preference, falling back to CPU safely."""
    if pref == "cpu":
        return torch.device("cpu")
    if pref in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "cuda":
        print("[train] CUDA requested but unavailable — falling back to CPU.")
    return torch.device("cpu")


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional["torch.cuda.amp.GradScaler"] = None,
) -> tuple[float, float]:
    """One pass over ``loader``. Trains if ``optimizer`` is given, else evaluates."""
    training = optimizer is not None
    model.train(training)
    use_amp = scaler is not None and device.type == "cuda"

    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            if training:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            seen += inputs.size(0)

    return total_loss / max(seen, 1), correct / max(seen, 1)


class _EarlyStopper:
    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best = float("inf")
        self.bad = 0
        self.best_state: Optional[dict] = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Return True if training should stop."""
        if val_loss < self.best - 1e-4:
            self.best = val_loss
            self.bad = 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False
        self.bad += 1
        return self.bad >= self.patience


def _train_phase(
    name: str,
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    patience: int,
    scaler,
    scheduler=None,
    history: Optional[List[dict]] = None,
) -> List[dict]:
    history = history if history is not None else []
    stopper = _EarlyStopper(patience)
    has_val = "val" in loaders

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, loaders["train"], criterion, device, optimizer, scaler)
        if has_val:
            va_loss, va_acc = _run_epoch(model, loaders["val"], criterion, device)
        else:
            va_loss, va_acc = tr_loss, tr_acc
        if scheduler is not None:
            scheduler.step(va_loss)

        record = {
            "phase": name, "epoch": epoch,
            "train_loss": round(tr_loss, 4), "train_acc": round(tr_acc, 4),
            "val_loss": round(va_loss, 4), "val_acc": round(va_acc, 4),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(f"[{name}] epoch {epoch:02d}/{epochs}  "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.3f}  "
              f"val_loss={va_loss:.4f} acc={va_acc:.3f}")

        if has_val and stopper.step(va_loss, model):
            print(f"[{name}] early stopping at epoch {epoch}")
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)  # restore best weights
    return history


def train(
    samples: Sequence[Sample],
    tcfg: Optional[TrainConfig] = None,
    preprocess: Optional[PreprocessConfig] = None,
) -> dict:
    """Run the full two-phase training and save the best checkpoint.

    Returns a summary dict with the checkpoint path, class map and history.
    """
    tcfg = tcfg or TrainConfig()
    preprocess = preprocess or PreprocessConfig()
    set_seed(tcfg.seed)

    device = resolve_device(tcfg.device)
    bundle = build_dataloaders(
        samples, preprocess=preprocess, batch_size=tcfg.batch_size, num_workers=tcfg.num_workers
    )
    if "train" not in bundle.loaders:
        raise ValueError("No training samples found (check the manifest's 'split' column).")

    mcfg = ModelConfig(backbone=tcfg.backbone, num_classes=len(bundle.class_to_idx))
    model = create_model(mcfg).to(device)
    criterion = nn.CrossEntropyLoss(weight=bundle.class_weights.to(device))
    scaler = torch.amp.GradScaler("cuda", enabled=tcfg.amp and device.type == "cuda")

    print(f"Device: {device} | backbone: {tcfg.backbone} | classes: {bundle.class_to_idx}")

    history: List[dict] = []

    # ---- Phase 1: feature extraction (head only) ----
    freeze_backbone(model, tcfg.backbone)
    print(f"Phase 1 trainable params: {trainable_parameter_count(model):,}")
    opt1 = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=tcfg.lr_head, weight_decay=tcfg.weight_decay,
    )
    _train_phase("head", model, bundle.loaders, criterion, opt1, device,
                 tcfg.epochs_head, tcfg.patience, scaler, history=history)

    # ---- Phase 2: fine-tuning (unfreeze top) ----
    unfreeze_top_fraction(model, tcfg.finetune_fraction)
    print(f"Phase 2 trainable params: {trainable_parameter_count(model):,}")
    opt2 = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=tcfg.lr_finetune, weight_decay=tcfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt2, mode="min", factor=0.5, patience=2)
    _train_phase("finetune", model, bundle.loaders, criterion, opt2, device,
                 tcfg.epochs_finetune, tcfg.patience, scaler, scheduler, history=history)

    # ---- Persist ----
    out_dir = Path(tcfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"model_{tcfg.backbone}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_to_idx": bundle.class_to_idx,
            "model_config": asdict(mcfg),
            "train_config": asdict(tcfg),
            "preprocess": preprocess.to_dict(),
            "history": history,
        },
        ckpt_path,
    )
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Saved checkpoint -> {ckpt_path}")

    return {"checkpoint": str(ckpt_path), "class_to_idx": bundle.class_to_idx, "history": history}
