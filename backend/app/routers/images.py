"""Serve pipeline-stage images stored in the database (survives ephemeral disks)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlmodel import Session

from ..db import get_session
from ..models_db import StudyImage

router = APIRouter(prefix="/api/image", tags=["images"])


@router.get("/{image_id}")
def get_image(image_id: int, session: Session = Depends(get_session)) -> Response:
    img = session.get(StudyImage, image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    return Response(content=img.data, media_type="image/png", headers={"Cache-Control": "public, max-age=31536000"})
