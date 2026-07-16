
from __future__ import annotations

import json
from typing import Optional

from sqlmodel import Session

from .models_db import AuditLog


def log_action(
    session: Session,
    org_id: int,
    actor_user_id: int,
    action: str,
    target_type: str,
    target_id: int,
    meta: Optional[dict] = None,
) -> None:
    session.add(AuditLog(
        org_id=org_id,
        actor_user_id=actor_user_id,
        action=action,
        target_type=target_type,
        target_id=target_id,
        meta=json.dumps(meta) if meta is not None else None,
    ))
    session.commit()
