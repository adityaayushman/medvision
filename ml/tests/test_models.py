"""Smoke tests for the PyTorch model layer.

Uses ``pretrained=False`` so tests need no weight download and run on CPU.
These verify wiring/shapes, not accuracy (that needs real data + training).

Default tests use EfficientNet-B0 (~20 MB) so they're fast and reliable on
memory-constrained machines. The full VGG16 path (~550 MB, can OOM a loaded
laptop) is covered by a ``@pytest.mark.heavy`` test, deselected by default; run
it with:  ``pytest -m heavy``.
"""

from __future__ import annotations

import gc
from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from medchron.config import PreprocessConfig
from medchron.explain import GradCAM
from medchron.models import (
    ModelConfig,
    Predictor,
    build_transforms,
    create_model,
    freeze_backbone,
    trainable_parameter_count,
    unfreeze_top_fraction,
)
from medchron.models.backbone import default_gradcam_layer

CPU = torch.device("cpu")
LIGHT = "efficientnet_b0"


@pytest.fixture
def make_model():
    """Create models and free them at teardown to bound memory use."""
    created = []

    def _make(backbone: str = LIGHT, num_classes: int = 2):
        model = create_model(ModelConfig(backbone=backbone, num_classes=num_classes, pretrained=False))
        created.append(model)
        return model

    yield _make
    created.clear()
    gc.collect()


def test_create_forward_shape(make_model):
    model = make_model(LIGHT, 3)
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 3)


def test_freeze_then_unfreeze_changes_trainable_count(make_model):
    model = make_model(LIGHT, 2)
    freeze_backbone(model, LIGHT)
    frozen = trainable_parameter_count(model)
    assert all(not p.requires_grad for n, p in model.named_parameters()
               if not n.startswith("classifier"))
    unfreeze_top_fraction(model, 0.5)
    assert trainable_parameter_count(model) > frozen


def test_gradcam_heatmap_shape_and_range(make_model):
    model = make_model(LIGHT, 2)
    layer = default_gradcam_layer(model, LIGHT)
    with GradCAM(model, layer) as cam:
        heatmap, cls, probs = cam(torch.randn(1, 3, 224, 224))
    assert heatmap.ndim == 2
    assert 0.0 <= float(heatmap.min()) and float(heatmap.max()) <= 1.0 + 1e-5
    assert cls in (0, 1)
    assert abs(float(np.sum(probs)) - 1.0) < 1e-4


def test_build_transforms_output_tensor():
    from PIL import Image
    tf = build_transforms(train=False)
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), np.uint8))
    t = tf(img)
    assert t.shape == (3, 224, 224)


def test_predictor_roundtrip(tmp_path, make_model):
    cfg = ModelConfig(backbone=LIGHT, num_classes=2, pretrained=False)
    model = make_model(LIGHT, 2)
    path = tmp_path / "model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_to_idx": {"normal": 0, "pneumonia": 1},
            "model_config": asdict(cfg),
            "train_config": {},
            "preprocess": PreprocessConfig().to_dict(),
            "history": [],
        },
        path,
    )
    gc.collect()

    predictor = Predictor(str(path), device=CPU)
    image = np.random.randint(0, 255, (300, 300, 3), np.uint8)

    pred = predictor.predict(image)
    assert pred["label"] in {"normal", "pneumonia"}
    assert 0.0 <= pred["confidence"] <= 1.0
    assert set(pred["probabilities"]) == {"normal", "pneumonia"}

    overlay, pred2 = predictor.explain(image)
    assert overlay.shape[2] == 3
    assert "explained_class" in pred2


@pytest.mark.heavy
def test_vgg16_specific_wiring():
    """VGG16 is the plan's baseline: verify its head + Grad-CAM layer wiring.

    Heavy (~550 MB): deselected by default, run with ``pytest -m heavy``.
    """
    model = create_model(ModelConfig(backbone="vgg16", num_classes=2, pretrained=False))
    out = model(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 2)
    assert isinstance(default_gradcam_layer(model, "vgg16"), torch.nn.Conv2d)
    del model
    gc.collect()
