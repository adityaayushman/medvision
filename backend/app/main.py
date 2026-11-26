
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import OperationalError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

from .config import settings
from .db import db_ready, init_db, retry_init_in_background
from .ml import get_analyzer
from .routers import auth, dashboard, datasets, images, inference, patients, research, studies


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not init_db():
        retry_init_in_background()
    get_analyzer()
    yield


app = FastAPI(
    title="MedChron AI",
    description="Medical imaging intelligence API. Research/educational — not for clinical use.",
    version="0.1.0",
    lifespan=lifespan,
)

@app.exception_handler(OperationalError)
@app.exception_handler(PoolTimeoutError)
async def database_unavailable(request: Request, exc: Exception) -> JSONResponse:
    """A database that cannot be reached is a 503 the caller can act on, not a hang or a bare 500."""
    return JSONResponse(status_code=503, content={"detail": "database unavailable, try again shortly"})


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(settings.storage_dir)), name="static")

app.include_router(inference.router)
app.include_router(patients.router)
app.include_router(datasets.router)
app.include_router(studies.router)
app.include_router(images.router)
app.include_router(auth.router)
app.include_router(dashboard.router)
app.include_router(research.router)


@app.get("/health", tags=["meta"])
def health() -> dict:
    analyzer = get_analyzer()
    database_ok = db_ready()
    return {
        "status": "ok" if database_ok else "degraded",
        "database": "ok" if database_ok else "unavailable",
        "model_loaded": analyzer.model_loaded,
        "modality": settings.modality,
        "modalities": analyzer.available_modalities(),
        "disclaimer": "Research/educational software. Not for clinical use.",
    }
