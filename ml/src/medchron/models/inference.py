
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from ..config import PreprocessConfig
from ..explain.gradcam import GradCAM, overlay_heatmap
from ..imaging import model_image
from .backbone import ModelConfig, create_model, default_gradcam_layer
from .dataset import build_transforms


class Predictor:

    def __init__(self, ckpt_path: str, device: Optional[torch.device] = None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        self.model_config = ModelConfig(**ckpt["model_config"])
        self.model_config.pretrained = False
        self.model = create_model(self.model_config).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self.class_to_idx: Dict[str, int] = ckpt["class_to_idx"]
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.preprocess = PreprocessConfig(**ckpt.get("preprocess", {}))
        self._tf = build_transforms(train=False, input_size=self.preprocess.model_input_size[0])
        self._target_layer = default_gradcam_layer(self.model, self.model_config.backbone)

        num_epochs = len(ckpt.get("history", []))
        trained_at = datetime.fromtimestamp(Path(ckpt_path).stat().st_mtime, tz=timezone.utc)
        task = ckpt.get("task", "classification")
        self.model_version = (
            f"{self.model_config.backbone} ({task}, {num_epochs} epochs, "
            f"trained {trained_at.strftime('%Y-%m-%d')})"
        )

    def _to_tensor(self, image_bgr: np.ndarray) -> torch.Tensor:
        rgb = model_image(image_bgr, self.preprocess)
        tensor = self._tf(Image.fromarray(rgb))
        return tensor.unsqueeze(0).to(self.device)

    def _format(self, probs: np.ndarray) -> Dict:
        idx = int(probs.argmax())
        return {
            "label": self.idx_to_class[idx],
            "confidence": float(probs[idx]),
            "probabilities": {self.idx_to_class[i]: float(p) for i, p in enumerate(probs)},
        }

    @torch.no_grad()
    def predict(self, image_bgr: np.ndarray) -> Dict:
        logits = self.model(self._to_tensor(image_bgr))
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return self._format(probs)

    def explain(
        self, image_bgr: np.ndarray, class_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict]:
        t0 = time.perf_counter()
        tensor = self._to_tensor(image_bgr)
        with GradCAM(self.model, self._target_layer) as cam:
            heatmap, used_idx, probs = cam(tensor, class_idx)
        inference_time_ms = round((time.perf_counter() - t0) * 1000, 1)

        base = cv2.resize(image_bgr, self.preprocess.target_size, interpolation=cv2.INTER_AREA)
        overlay = overlay_heatmap(base, heatmap)
        prediction = self._format(probs)
        prediction["explained_class"] = self.idx_to_class[used_idx]
        prediction["model_version"] = self.model_version
        prediction["inference_time_ms"] = inference_time_ms
        return overlay, prediction

    def predict_path(self, path: str) -> Dict:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return self.predict(image)
