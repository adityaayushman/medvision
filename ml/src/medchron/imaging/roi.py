
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from ..config import PreprocessConfig


@dataclass
class ROI:

    x: int
    y: int
    w: int
    h: int
    area: float
    crop: np.ndarray

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    def to_dict(self) -> dict:
        return {"bbox": list(self.bbox), "area": round(self.area, 1)}


def extract_rois(
    mask: np.ndarray, source: np.ndarray, cfg: PreprocessConfig
) -> List[ROI]:
    h_img, w_img = mask.shape[:2]
    min_area = cfg.min_roi_area_ratio * h_img * w_img

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois: List[ROI] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        p = cfg.roi_padding
        x0, y0 = max(0, x - p), max(0, y - p)
        x1, y1 = min(w_img, x + w + p), min(h_img, y + h + p)
        crop = source[y0:y1, x0:x1].copy()
        rois.append(ROI(x0, y0, x1 - x0, y1 - y0, float(area), crop))

    rois.sort(key=lambda r: r.area, reverse=True)
    return rois[: cfg.max_rois]


def draw_rois(image: np.ndarray, rois: List[ROI]) -> np.ndarray:
    out = image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    for i, r in enumerate(rois, 1):
        cv2.rectangle(out, (r.x, r.y), (r.x + r.w, r.y + r.h), (0, 255, 0), 2)
        cv2.putText(
            out,
            f"ROI{i}",
            (r.x, max(12, r.y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return out
