
from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
import torchvision.models as tvm

_BACKBONES = {
    "vgg16": (tvm.vgg16, tvm.VGG16_Weights.DEFAULT, "classifier"),
    "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.DEFAULT, "fc"),
    "densenet121": (tvm.densenet121, tvm.DenseNet121_Weights.DEFAULT, "classifier"),
    "efficientnet_b0": (tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights.DEFAULT, "classifier"),
}

HEAD_PREFIX = {k: v[2] for k, v in _BACKBONES.items()}


@dataclass
class ModelConfig:
    backbone: str = "vgg16"
    num_classes: int = 2
    pretrained: bool = True
    dropout: float = 0.5


def create_model(cfg: ModelConfig) -> nn.Module:
    if cfg.backbone not in _BACKBONES:
        raise ValueError(f"Unknown backbone {cfg.backbone!r}. Known: {sorted(_BACKBONES)}")
    ctor, weights, head = _BACKBONES[cfg.backbone]
    model = ctor(weights=weights if cfg.pretrained else None)

    if cfg.backbone == "vgg16":
        in_f = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_f, cfg.num_classes)
    elif cfg.backbone == "resnet50":
        in_f = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(in_f, cfg.num_classes))
    elif cfg.backbone == "densenet121":
        in_f = model.classifier.in_features
        model.classifier = nn.Linear(in_f, cfg.num_classes)
    elif cfg.backbone == "efficientnet_b0":
        in_f = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_f, cfg.num_classes)

    return model


def freeze_backbone(model: nn.Module, backbone: str) -> None:
    head = HEAD_PREFIX[backbone]
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(head)


def unfreeze_top_fraction(model: nn.Module, fraction: float = 0.25) -> int:
    params = list(model.parameters())
    cut = int(len(params) * (1.0 - fraction))
    unfrozen = 0
    for i, p in enumerate(params):
        if i >= cut:
            p.requires_grad = True
            unfrozen += 1
    return unfrozen


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def default_gradcam_layer(model: nn.Module, backbone: str) -> nn.Module:
    if backbone == "vgg16":
        return model.features[28]
    if backbone == "resnet50":
        return model.layer4[-1]
    if backbone == "densenet121":
        return model.features.norm5
    if backbone == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(f"No Grad-CAM layer mapping for {backbone!r}")
