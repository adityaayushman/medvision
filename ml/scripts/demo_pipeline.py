"""Run the MedChron preprocessing pipeline on a sample image and save a montage.

Uses only OpenCV/NumPy (no matplotlib), so it runs headless anywhere.

    python ml/scripts/demo_pipeline.py \
        --image assets/samples/chest_xray_sample.jpg \
        --modality chest_xray \
        --out ml/artifacts/pipeline_demo.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from medchron import MedicalImagePipeline, get_config
from medchron.imaging.pipeline import PipelineResult


def _to_bgr(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Normalise any stage image to a fixed-size 3-channel BGR tile."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def _label(tile: np.ndarray, text: str) -> np.ndarray:
    """Draw a caption bar along the bottom of a tile."""
    out = tile.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, h - 24), (w, h), (0, 0, 0), -1)
    cv2.putText(
        out, text, (6, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    )
    return out


def build_montage(result: PipelineResult, tile: int = 256, cols: int = 4) -> np.ndarray:
    """Grid every named pipeline stage into one image."""
    tiles = [
        _label(_to_bgr(img, (tile, tile)), name)
        for name, img in result.stages.items()
    ]
    while len(tiles) % cols != 0:
        tiles.append(np.zeros((tile, tile, 3), np.uint8))  # pad the last row
    rows = [np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="MedChron preprocessing demo")
    parser.add_argument("--image", default="assets/samples/chest_xray_sample.jpg")
    parser.add_argument("--modality", default="chest_xray")
    parser.add_argument("--out", default="ml/artifacts/pipeline_demo.png")
    args = parser.parse_args()

    pipe = MedicalImagePipeline(get_config(args.modality))
    result = pipe.run_path(args.image)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), build_montage(result))

    print(json.dumps(result.summary(), indent=2))
    print(f"\nMontage saved -> {out_path}")


if __name__ == "__main__":
    main()
