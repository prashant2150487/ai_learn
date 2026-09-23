import secrets
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.auth.constants import SESSION_MAX_AGE_SECONDS
from app.auth.security import hash_password, verify_password
from app.core.config import settings
from app.db.models import Session as UserSession
from app.db.models import User


def create_user(db: Session, *, name: str, email: str, password: str) -> User:
    existing = db.query(User).filter(User.email == email).one_or_none()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )
    user = User(name=name, email=email, password_hash=hash_password(password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, email: str, password: str) -> User:
    user = db.query(User).filter(User.email == email).one_or_none()
    if user is None or not verify_password(password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    return user


def create_session(db: Session, user_id: int) -> UserSession:
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=SESSION_MAX_AGE_SECONDS)
    session = UserSession(token=token, user_id=user_id, expires_at=expires_at)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_user_by_session_token(db: Session, token: str | None) -> User | None:
    if not token:
        return None
    now = datetime.now(timezone.utc)
    row = (
        db.query(UserSession)
        .filter(UserSession.token == token, UserSession.expires_at > now)
        .one_or_none()
    )
    if row is None:
        return None
    return db.query(User).filter(User.id == row.user_id).one_or_none()


def revoke_session(db: Session, token: str | None) -> None:
    if not token:
        return
    db.query(UserSession).filter(UserSession.token == token).delete()
    db.commit()


def ensure_bootstrap_user(db: Session) -> None:
    if db.query(User).count() > 0:
        return
    create_user(
        db,
        name=settings.bootstrap_user_name,
        email=settings.bootstrap_user_email,
        password=settings.bootstrap_user_password,
    )
