
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    database_url: str = os.getenv("DATABASE_URL", f"sqlite:///{REPO_ROOT / 'backend' / 'medchron.db'}")
    model_checkpoint: str = os.getenv("MODEL_CHECKPOINT", str(REPO_ROOT / "ml" / "artifacts" / "model_vgg16.pt"))
    modality: str = os.getenv("MEDCHRON_MODALITY", "chest_xray")
    model_checkpoint_brain_mri: str = os.getenv(
        "MODEL_CHECKPOINT_BRAIN_MRI",
        str(REPO_ROOT / "ml" / "artifacts" / "brain_mri" / "model_efficientnet_b0.pt"),
    )
    model_checkpoint_mammography: str = os.getenv(
        "MODEL_CHECKPOINT_MAMMOGRAPHY",
        str(REPO_ROOT / "ml" / "artifacts" / "mammography" / "model_efficientnet_b0.pt"),
    )
    storage_dir: Path = field(default_factory=lambda: Path(os.getenv("STORAGE_DIR", str(REPO_ROOT / "backend" / "storage"))))
    cors_origins: List[str] = field(default_factory=lambda: os.getenv(
        "CORS_ORIGINS", "http://localhost:3000"
    ).split(","))
    jwt_secret: str = os.getenv("JWT_SECRET", "dev-insecure-secret-change-me-in-production-please")
    jwt_expires_minutes: int = int(os.getenv("JWT_EXPIRES_MINUTES", "480"))

    def __post_init__(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        (self.storage_dir / "uploads").mkdir(exist_ok=True)
        (self.storage_dir / "overlays").mkdir(exist_ok=True)


settings = Settings()
