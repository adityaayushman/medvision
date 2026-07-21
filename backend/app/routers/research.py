
from __future__ import annotations

import json
from typing import List, Optional

from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from ..audit import log_action
from ..auth import get_current_user, require_role
from ..db import get_session
from ..models_db import ExperimentRun, User
from ..schemas import ExperimentRunCreate, ExperimentRunRead

router = APIRouter(prefix="/api/research", tags=["research"])


def _run_to_read(run: ExperimentRun, session: Session) -> ExperimentRunRead:
    creator = session.get(User, run.created_by_user_id)
    return ExperimentRunRead(
        id=run.id,
        kind=run.kind,
        modality=run.modality,
        backbone=run.backbone,
        label=run.label,
        metrics=json.loads(run.metrics) if run.metrics else {},
        notes=run.notes,
        created_by_user_id=run.created_by_user_id,
        created_by_email=creator.email if creator else None,
        created_at=run.created_at,
    )


@router.post("/runs", response_model=ExperimentRunRead)
def create_run(
    body: ExperimentRunCreate,
    user: User = Depends(require_role("admin", "researcher")),
    session: Session = Depends(get_session),
):
    run = ExperimentRun(
        kind=body.kind,
        modality=body.modality,
        backbone=body.backbone,
        label=body.label,
        metrics=json.dumps(body.metrics),
        notes=body.notes,
        created_by_user_id=user.id,
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    log_action(session, user.org_id, user.id, "experiment_run.created", "experiment_run", run.id)
    return _run_to_read(run, session)


@router.get("/runs", response_model=List[ExperimentRunRead])
def list_runs(
    modality: Optional[str] = None,
    kind: Optional[str] = None,
    user: User = Depends(require_role("admin", "researcher")),
    session: Session = Depends(get_session),
):
    stmt = select(ExperimentRun)
    if modality:
        stmt = stmt.where(ExperimentRun.modality == modality)
    if kind:
        stmt = stmt.where(ExperimentRun.kind == kind)
    runs = session.exec(stmt.order_by(ExperimentRun.created_at.desc())).all()
    return [_run_to_read(r, session) for r in runs]
