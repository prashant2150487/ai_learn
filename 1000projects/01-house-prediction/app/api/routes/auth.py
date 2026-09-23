from typing import Annotated

from fastapi import APIRouter, Cookie, Depends, Response, status
from sqlalchemy.orm import Session

from app.auth.constants import SESSION_COOKIE_NAME, SESSION_MAX_AGE_SECONDS
from app.auth.dependencies import CurrentUser, DbSession, user_to_public
from app.auth.schemas import AuthResponse, LoginRequest, RegisterRequest, UserPublic
from app.auth.service import authenticate_user, create_session, create_user, revoke_session
from app.core.config import settings

router = APIRouter(prefix="/auth", tags=["Auth"])


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=SESSION_MAX_AGE_SECONDS,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        path="/",
    )


def _clear_session_cookie(response: Response) -> None:
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
    )


@router.post("/register", response_model=UserPublic, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest, db: DbSession):
    user = create_user(db, name=body.name, email=body.email, password=body.password)
    return user_to_public(user)


@router.post("/login", response_model=AuthResponse)
def login(body: LoginRequest, response: Response, db: DbSession):
    user = authenticate_user(db, body.email, body.password)
    session = create_session(db, user.id)
    _set_session_cookie(response, session.token)
    return AuthResponse(user=user_to_public(user), message="Login successful")


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    response: Response,
    db: DbSession,
    session_token: Annotated[str | None, Cookie(alias=SESSION_COOKIE_NAME)] = None,
):
    revoke_session(db, session_token)
    _clear_session_cookie(response)
    return None


@router.get("/me", response_model=UserPublic)
def me(user: CurrentUser):
    return user_to_public(user)
