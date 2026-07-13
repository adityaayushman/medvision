"""Model layer: backbones, datasets, training, evaluation, and inference."""

from .backbone import (
    ModelConfig,
    create_model,
    default_gradcam_layer,
    freeze_backbone,
    trainable_parameter_count,
    unfreeze_top_fraction,
)
from .dataset import MedChronDataset, build_dataloaders, build_transforms
from .evaluate import compute_metrics, evaluate_checkpoint
from .inference import Predictor
from .train import TrainConfig, train

__all__ = [
    "ModelConfig",
    "create_model",
    "freeze_backbone",
    "unfreeze_top_fraction",
    "trainable_parameter_count",
    "default_gradcam_layer",
    "MedChronDataset",
    "build_dataloaders",
    "build_transforms",
    "TrainConfig",
    "train",
    "compute_metrics",
    "evaluate_checkpoint",
    "Predictor",
]
