
from __future__ import annotations

from dataclasses import asdict
from typing import List

from fastapi import APIRouter
from medchron.data import REGISTRY, recommended_v1

from ..schemas import DatasetSpecRead

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.get("", response_model=List[DatasetSpecRead])
def list_datasets():
    return [asdict(spec) for spec in REGISTRY.values()]


@router.get("/recommended", response_model=List[DatasetSpecRead])
def recommended():
    return [asdict(spec) for spec in recommended_v1()]
