
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Tuple

DenoiseMethod = Literal["gaussian", "median", "bilateral", "none"]
ThresholdMethod = Literal["otsu", "adaptive_mean", "adaptive_gaussian"]
MorphOp = Literal["none", "open", "close", "erode", "dilate"]


@dataclass(frozen=True)
class PreprocessConfig:

    target_size: Tuple[int, int] = (512, 512)
    model_input_size: Tuple[int, int] = (224, 224)

    denoise_method: DenoiseMethod = "gaussian"
    denoise_ksize: int = 5
    bilateral_d: int = 9

    min_focus: float = 80.0
    brightness_lo: float = 25.0
    brightness_hi: float = 235.0
    min_contrast: float = 12.0

    use_hist_eq: bool = True
    clahe_clip: float = 2.0
    clahe_tile: int = 8

    threshold_method: ThresholdMethod = "otsu"
    adaptive_block_size: int = 35
    adaptive_c: int = 5

    morph_op: MorphOp = "open"
    morph_ksize: int = 5
    morph_iterations: int = 1

    min_roi_area_ratio: float = 0.002
    max_rois: int = 8
    roi_padding: int = 6

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS: dict[str, PreprocessConfig] = {
    "chest_xray": PreprocessConfig(),
    "brain_mri": PreprocessConfig(
        denoise_method="bilateral",
        clahe_clip=3.0,
        threshold_method="otsu",
        morph_op="close",
        min_focus=20.0,
    ),
    "mammography": PreprocessConfig(
        clahe_clip=2.5,
        min_roi_area_ratio=0.001,
        max_rois=4,
    ),
}


def get_config(name: str = "chest_xray") -> PreprocessConfig:
    try:
        return PRESETS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown modality preset {name!r}. Known: {sorted(PRESETS)}"
        ) from exc
