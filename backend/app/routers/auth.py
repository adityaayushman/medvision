
from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from ..audit import log_action
from ..auth import create_access_token, get_current_user, hash_password, require_role, verify_password
from ..db import get_session
from ..models_db import Organization, User
from ..schemas import LoginRequest, OrgSignup, TokenResponse, UserCreate, UserRead

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _user_to_read(user: User) -> UserRead:
    return UserRead(
        id=user.id,
        org_id=user.org_id,
        email=user.email,
        role=user.role,
        name=user.name,
        created_at=user.created_at,
    )


@router.post("/signup", response_model=TokenResponse)
def signup(body: OrgSignup, session: Session = Depends(get_session)):
    existing = session.exec(select(User).where(User.email == body.email)).first()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    org = Organization(name=body.org_name)
    session.add(org)
    session.commit()
    session.refresh(org)

    user = User(
        org_id=org.id,
        email=body.email,
        password_hash=hash_password(body.password),
        role="admin",
        name=body.name,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    log_action(session, org.id, user.id, "user.created", "user", user.id)

    return TokenResponse(access_token=create_access_token(user), user=_user_to_read(user))


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, session: Session = Depends(get_session)):
    user = session.exec(select(User).where(User.email == body.email)).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return TokenResponse(access_token=create_access_token(user), user=_user_to_read(user))


@router.get("/me", response_model=UserRead)
def me(user: User = Depends(get_current_user)):
    return _user_to_read(user)


@router.post("/users", response_model=UserRead)
def create_teammate(
    body: UserCreate,
    admin: User = Depends(require_role("admin")),
    session: Session = Depends(get_session),
):
    if body.role not in ("admin", "radiologist"):
        raise HTTPException(status_code=422, detail="role must be 'admin' or 'radiologist'")
    existing = session.exec(select(User).where(User.email == body.email)).first()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(
        org_id=admin.org_id,
        email=body.email,
        password_hash=hash_password(body.password),
        role=body.role,
        name=body.name,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    log_action(session, admin.org_id, admin.id, "user.created", "user", user.id)
    return _user_to_read(user)


@router.get("/users", response_model=List[UserRead])
def list_teammates(admin: User = Depends(require_role("admin")), session: Session = Depends(get_session)):
    users = session.exec(select(User).where(User.org_id == admin.org_id).order_by(User.created_at)).all()
    return [_user_to_read(u) for u in users]
