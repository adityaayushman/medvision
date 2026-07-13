"""Backend settings, configured via environment variables with sane defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# repo root = two levels up from this file (backend/app/config.py)
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    # SQLite lives under backend/ by default; override with DATABASE_URL
    database_url: str = os.getenv("DATABASE_URL", f"sqlite:///{REPO_ROOT / 'backend' / 'medchron.db'}")
    # trained checkpoint; if missing, the API runs in preprocess-only mode
    model_checkpoint: str = os.getenv("MODEL_CHECKPOINT", str(REPO_ROOT / "ml" / "artifacts" / "model_vgg16.pt"))
    modality: str = os.getenv("MEDCHRON_MODALITY", "chest_xray")
    # where uploaded images + Grad-CAM overlays are written (served statically)
    storage_dir: Path = field(default_factory=lambda: Path(os.getenv("STORAGE_DIR", str(REPO_ROOT / "backend" / "storage"))))
    cors_origins: List[str] = field(default_factory=lambda: os.getenv(
        "CORS_ORIGINS", "http://localhost:3000"
    ).split(","))

    def __post_init__(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        (self.storage_dir / "uploads").mkdir(exist_ok=True)
        (self.storage_dir / "overlays").mkdir(exist_ok=True)


settings = Settings()
