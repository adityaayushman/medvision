"""MedChron AI backend — FastAPI application entrypoint.

Run:  uvicorn app.main:app --reload   (from the backend/ directory)
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import settings
from .db import init_db
from .ml import get_analyzer
from .routers import datasets, images, inference, patients, studies


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    get_analyzer()  # warm the pipeline / load the model once
    yield


app = FastAPI(
    title="MedChron AI",
    description="Medical imaging intelligence API. Research/educational — not for clinical use.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve uploaded images + Grad-CAM overlays
app.mount("/static", StaticFiles(directory=str(settings.storage_dir)), name="static")

app.include_router(inference.router)
app.include_router(patients.router)
app.include_router(datasets.router)
app.include_router(studies.router)
app.include_router(images.router)


@app.get("/health", tags=["meta"])
def health() -> dict:
    analyzer = get_analyzer()
    return {
        "status": "ok",
        "model_loaded": analyzer.model_loaded,
        "modality": settings.modality,
        "modalities": analyzer.available_modalities(),
        "disclaimer": "Research/educational software. Not for clinical use.",
    }
