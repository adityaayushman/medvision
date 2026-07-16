
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        for module in model.modules():
            if getattr(module, "inplace", False):
                module.inplace = False
        self._handle = target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, _module, _inp, output) -> None:
        self.activations = output

    def __call__(
        self, input_tensor: torch.Tensor, class_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        self.model.eval()
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())

        score = logits[0, class_idx]
        (grads,) = torch.autograd.grad(score, self.activations, retain_graph=False)

        acts = self.activations.detach()
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1)).squeeze(0)
        cam = cam.cpu().numpy().astype(np.float32)
        cam -= cam.min()
        if cam.max() > 1e-8:
            cam /= cam.max()
        return cam, class_idx, probs

    def remove(self) -> None:
        self._handle.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()


def overlay_heatmap(image_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(image_bgr, 1 - alpha, heatmap, alpha, 0)
