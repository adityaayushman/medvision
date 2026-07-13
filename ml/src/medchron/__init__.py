"""MedChron AI — medical imaging intelligence platform (ML core).

Research / educational software. **Not a medical device and not for clinical use.**
"""

from .config import PRESETS, PreprocessConfig, get_config
from .imaging import (
    MedicalImagePipeline,
    PipelineResult,
    QualityReport,
    ROI,
)

__version__ = "0.1.0"

__all__ = [
    "PreprocessConfig",
    "PRESETS",
    "get_config",
    "MedicalImagePipeline",
    "PipelineResult",
    "QualityReport",
    "ROI",
    "__version__",
]
