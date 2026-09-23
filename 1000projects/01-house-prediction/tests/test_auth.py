from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app


def test_login_logout_me():
    with TestClient(app) as client:
        login = client.post(
            "/auth/login",
            json={
                "email": settings.bootstrap_user_email,
                "password": settings.bootstrap_user_password,
            },
        )
        assert login.status_code == 200
        assert login.json()["user"]["email"] == settings.bootstrap_user_email

        me = client.get("/auth/me")
        assert me.status_code == 200
        assert me.json()["email"] == settings.bootstrap_user_email

        logout = client.post("/auth/logout")
        assert logout.status_code == 204

        me_after = client.get("/auth/me")
        assert me_after.status_code == 401


def test_register_and_login():
    with TestClient(app) as client:
        email = "newuser@example.com"
        register = client.post(
            "/auth/register",
            json={"name": "New User", "email": email, "password": "secure-pass-1"},
        )
        assert register.status_code == 201

        login = client.post(
            "/auth/login",
            json={"email": email, "password": "secure-pass-1"},
        )
        assert login.status_code == 200
