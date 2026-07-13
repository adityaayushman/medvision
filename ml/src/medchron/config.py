"""Configuration objects for the MedChron imaging pipeline.

Everything the preprocessing pipeline does is driven by :class:`PreprocessConfig`,
so behaviour is reproducible, serialisable, and per-modality tunable. Never
hard-code a kernel size or threshold inside the pipeline — add it here instead.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Tuple

DenoiseMethod = Literal["gaussian", "median", "bilateral", "none"]
ThresholdMethod = Literal["otsu", "adaptive_mean", "adaptive_gaussian"]
MorphOp = Literal["none", "open", "close", "erode", "dilate"]


@dataclass(frozen=True)
class PreprocessConfig:
    """Reproducible settings for one modality's preprocessing.

    Sizes are ``(width, height)`` to match OpenCV's ``cv2.resize`` convention.
    Area thresholds are expressed as *ratios* of the image area so the same
    config works regardless of source resolution.
    """

    # --- geometry ---
    target_size: Tuple[int, int] = (512, 512)        # working resolution for DIP + display
    model_input_size: Tuple[int, int] = (224, 224)   # VGG16 / most ImageNet backbones

    # --- denoising ---
    denoise_method: DenoiseMethod = "gaussian"
    denoise_ksize: int = 5                            # forced odd at runtime
    bilateral_d: int = 9

    # --- contrast enhancement ---
    use_hist_eq: bool = True
    clahe_clip: float = 2.0
    clahe_tile: int = 8

    # --- segmentation ---
    threshold_method: ThresholdMethod = "otsu"
    adaptive_block_size: int = 35                     # forced odd at runtime
    adaptive_c: int = 5

    # --- morphological cleanup ---
    morph_op: MorphOp = "open"
    morph_ksize: int = 5
    morph_iterations: int = 1

    # --- region of interest ---
    min_roi_area_ratio: float = 0.002                 # ignore blobs smaller than 0.2% of the image
    max_rois: int = 8
    roi_padding: int = 6

    def to_dict(self) -> dict:
        """JSON-serialisable view, handy for logging with each result/report."""
        return asdict(self)


# Ready-made presets. V1 targets chest X-ray; the others are seeds for later
# modalities and prove the point that a new modality is *config*, not a rewrite.
PRESETS: dict[str, PreprocessConfig] = {
    "chest_xray": PreprocessConfig(),
    "brain_mri": PreprocessConfig(
        denoise_method="bilateral",
        clahe_clip=3.0,
        threshold_method="otsu",
        morph_op="close",
    ),
    "mammography": PreprocessConfig(
        clahe_clip=2.5,
        min_roi_area_ratio=0.001,
        max_rois=4,
    ),
}


def get_config(name: str = "chest_xray") -> PreprocessConfig:
    """Look up a preset by modality name."""
    try:
        return PRESETS[name]
    except KeyError as exc:  # pragma: no cover - guard rail
        raise KeyError(
            f"Unknown modality preset {name!r}. Known: {sorted(PRESETS)}"
        ) from exc
