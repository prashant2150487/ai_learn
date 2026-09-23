from typing import Annotated

from fastapi import Cookie, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth.constants import SESSION_COOKIE_NAME
from app.auth.schemas import UserPublic
from app.auth.service import get_user_by_session_token
from app.db.database import get_db
from app.db.models import User

DbSession = Annotated[Session, Depends(get_db)]


def get_current_user(
    db: DbSession,
    session_token: Annotated[str | None, Cookie(alias=SESSION_COOKIE_NAME)] = None,
) -> User:
    user = get_user_by_session_token(db, session_token)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def user_to_public(user: User) -> UserPublic:
    return UserPublic.model_validate(user)
