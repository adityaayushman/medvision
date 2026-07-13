"""Run a prediction + Grad-CAM explanation on a single image.

    python ml/scripts/explain.py --checkpoint ml/artifacts/model_vgg16.pt \
        --image assets/samples/chest_xray_sample.jpg --out ml/artifacts/gradcam.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from medchron.models import Predictor
from medchron.models.train import resolve_device


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict + Grad-CAM for one image")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out", default="ml/artifacts/gradcam.png")
    args = ap.parse_args()

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")

    predictor = Predictor(args.checkpoint, device=resolve_device(args.device))
    overlay, prediction = predictor.explain(image)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), overlay)

    print(json.dumps(prediction, indent=2))
    print(f"\nGrad-CAM overlay saved -> {out}")


if __name__ == "__main__":
    main()
