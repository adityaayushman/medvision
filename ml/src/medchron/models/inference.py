
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from ..config import PreprocessConfig
from ..explain.gradcam import GradCAM, overlay_heatmap
from ..imaging import model_image
from .backbone import ModelConfig, create_model, default_gradcam_layer
from .dataset import build_transforms
from .detect import SpatialBBoxNet


class Predictor:

    def __init__(self, ckpt_path: str, device: Optional[torch.device] = None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        self.model_config = ModelConfig(**ckpt["model_config"])
        self.model_config.pretrained = False
        self.backbone = self.model_config.backbone
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


class EnsemblePredictor:
    """Soft-votes across several single-backbone Predictors trained on the
    same manifest/split. Grad-CAM comes from the first (primary) member only —
    averaging heatmaps across architecturally different models isn't a
    well-defined operation, so this doesn't invent one."""

    def __init__(self, ckpt_paths: Sequence[str], device: Optional[torch.device] = None) -> None:
        if len(ckpt_paths) < 2:
            raise ValueError("EnsemblePredictor needs at least 2 checkpoints")
        self.members: List[Predictor] = [Predictor(p, device=device) for p in ckpt_paths]

        primary = self.members[0]
        for m in self.members[1:]:
            if m.class_to_idx != primary.class_to_idx:
                raise ValueError(
                    f"Ensemble members disagree on class_to_idx: "
                    f"{primary.class_to_idx} vs {m.class_to_idx}"
                )
        self.class_to_idx = primary.class_to_idx
        self.idx_to_class = primary.idx_to_class
        self.model_config = primary.model_config  # for callers that need e.g. gradcam target layer info
        self.backbone = "ensemble"

        backbones = ", ".join(m.backbone for m in self.members)
        self.model_version = f"ensemble({backbones})"

    def _format(self, probs: np.ndarray, per_model: List[Dict]) -> Dict:
        idx = int(probs.argmax())
        return {
            "label": self.idx_to_class[idx],
            "confidence": float(probs[idx]),
            "probabilities": {self.idx_to_class[i]: float(p) for i, p in enumerate(probs)},
            "per_model": per_model,
        }

    def predict(self, image_bgr: np.ndarray) -> Dict:
        per_model = []
        probs_sum = None
        for m in self.members:
            single = m.predict(image_bgr)
            per_model.append({
                "backbone": m.backbone,
                "label": single["label"],
                "confidence": single["confidence"],
            })
            p = np.array([single["probabilities"][self.idx_to_class[i]] for i in range(len(self.idx_to_class))])
            probs_sum = p if probs_sum is None else probs_sum + p
        avg_probs = probs_sum / len(self.members)
        return self._format(avg_probs, per_model)

    def explain(
        self, image_bgr: np.ndarray, class_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict]:
        t0 = time.perf_counter()
        result = self.predict(image_bgr)
        # Explain the ENSEMBLE's predicted class, not whatever the primary
        # member alone would have argmax'd to — otherwise the Grad-CAM overlay
        # could visualize a different class than the one reported as the
        # prediction when the primary model disagrees with the vote.
        ensemble_idx = class_idx if class_idx is not None else self.class_to_idx[result["label"]]
        primary = self.members[0]
        overlay, primary_pred = primary.explain(image_bgr, ensemble_idx)
        inference_time_ms = round((time.perf_counter() - t0) * 1000, 1)

        result["explained_class"] = primary_pred["explained_class"]
        result["model_version"] = self.model_version
        result["inference_time_ms"] = inference_time_ms
        return overlay, result


class _BBoxRegressor:
    """Loads a bbox_regression checkpoint (see models.detect.SpatialBBoxNet)
    and predicts a single normalized (cx, cy, w, h) box. Separate from
    Predictor because the checkpoint has no class_to_idx and the model's
    output is already constrained to [0,1] internally (spatial soft-argmax
    for the center, sigmoid for width/height) -- not a classifier."""

    def __init__(self, ckpt_path: str, device: Optional[torch.device] = None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model = SpatialBBoxNet(pretrained=False).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.preprocess = PreprocessConfig(**ckpt.get("preprocess", {}))
        self._tf = build_transforms(train=False, input_size=self.preprocess.model_input_size[0])

    @torch.no_grad()
    def predict_box(self, image_bgr: np.ndarray) -> Tuple[float, float, float, float]:
        rgb = model_image(image_bgr, self.preprocess)
        tensor = self._tf(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        cx, cy, w, h = self.model(tensor)[0].cpu().tolist()
        return cx, cy, w, h


class _SegmentationLocalizer:
    """Loads a segmentation checkpoint (see models.segment.UNetSegmenter) and
    derives a normalized (cx, cy, w, h) box from the predicted mask -- same
    predict_box(image_bgr) interface as _BBoxRegressor, so LocalizedPredictor
    doesn't need to know which localizer backs it. Exists because bbox
    regression plateaued at IoU ~0.05-0.08 (a 4-number target is too sparse a
    signal for this dataset size); a per-pixel mask gives far denser
    supervision -- see segment.py's module docstring."""

    def __init__(self, ckpt_path: str, device: Optional[torch.device] = None) -> None:
        from .segment import UNetSegmenter, mask_to_bbox

        self._mask_to_bbox = mask_to_bbox
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model = UNetSegmenter(pretrained=False).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.preprocess = PreprocessConfig(**ckpt.get("preprocess", {}))
        self._tf = build_transforms(train=False, input_size=self.preprocess.model_input_size[0])

    @torch.no_grad()
    def predict_box(self, image_bgr: np.ndarray) -> Tuple[float, float, float, float]:
        rgb = model_image(image_bgr, self.preprocess)
        tensor = self._tf(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        mask = torch.sigmoid(self.model(tensor))[0, 0].cpu().numpy()
        box = self._mask_to_bbox(mask)
        if box is None:
            # Nothing above threshold -- fall back to the full frame rather
            # than crash; LocalizedPredictor's crop clamp handles this the
            # same way a degenerate bbox-regressor output would.
            return 0.5, 0.5, 1.0, 1.0
        return box


def _load_localizer(ckpt_path: str, device: Optional[torch.device] = None):
    """Dispatches on the checkpoint's "task" marker so LocalizedPredictor can
    be built from either a bbox regressor or a segmentation model with the
    same constructor call."""
    probe = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    task = probe.get("task", "bbox_regression")
    if task == "segmentation":
        return _SegmentationLocalizer(ckpt_path, device=device)
    return _BBoxRegressor(ckpt_path, device=device)


class LocalizedPredictor:
    """Two-stage mammography pipeline: a localizer (bbox regressor or
    segmentation model) locates the lesion in a full mammogram, then the
    (unmodified) cropped-patch Predictor classifies the crop. Exists because
    a classifier trained on lesion-cropped patches can't be fed a full
    mammogram directly -- see ROADMAP / the mammography v2 write-up for why
    full-image classification underperforms."""

    def __init__(
        self,
        localizer_ckpt_path: str,
        classifier_ckpt_path: str,
        device: Optional[torch.device] = None,
        pad_frac: float = 0.15,
    ) -> None:
        self._detector = _load_localizer(localizer_ckpt_path, device=device)
        self.classifier = Predictor(classifier_ckpt_path, device=device)
        self.pad_frac = pad_frac

        self.class_to_idx = self.classifier.class_to_idx
        self.idx_to_class = self.classifier.idx_to_class
        self.backbone = "localized"
        localizer_kind = "segmentation" if isinstance(self._detector, _SegmentationLocalizer) else "bbox"
        self.model_version = f"localized({localizer_kind}+{self.classifier.backbone})"

    def _crop(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, Dict]:
        h_img, w_img = image_bgr.shape[:2]
        cx, cy, w, h = self._detector.predict_box(image_bgr)
        w = w * (1 + self.pad_frac)
        h = h * (1 + self.pad_frac)
        x0 = max(0, int((cx - w / 2) * w_img))
        x1 = min(w_img, int((cx + w / 2) * w_img))
        y0 = max(0, int((cy - h / 2) * h_img))
        y1 = min(h_img, int((cy + h / 2) * h_img))
        if x1 <= x0 or y1 <= y0:
            x0, y0, x1, y1 = 0, 0, w_img, h_img
        box = {"cx": cx, "cy": cy, "w": w, "h": h}
        return image_bgr[y0:y1, x0:x1], box

    def predict(self, image_bgr: np.ndarray) -> Dict:
        t0 = time.perf_counter()
        crop, box = self._crop(image_bgr)
        result = self.classifier.predict(crop)
        result["detected_bbox"] = box
        result["model_version"] = self.model_version
        result["inference_time_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return result

    def explain(
        self, image_bgr: np.ndarray, class_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict]:
        t0 = time.perf_counter()
        crop, box = self._crop(image_bgr)
        overlay, prediction = self.classifier.explain(crop, class_idx)
        prediction["detected_bbox"] = box
        prediction["model_version"] = self.model_version
        prediction["inference_time_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return overlay, prediction
