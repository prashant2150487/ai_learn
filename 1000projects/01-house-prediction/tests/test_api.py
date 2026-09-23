from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app

SAMPLE = {
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.984,
    "AveBedrms": 1.024,
    "Population": 322.0,
    "AveOccup": 2.556,
    "Latitude": 37.88,
    "Longitude": -122.23,
}


def _login(client: TestClient) -> None:
    response = client.post(
        "/auth/login",
        json={
            "email": settings.bootstrap_user_email,
            "password": settings.bootstrap_user_password,
        },
    )
    assert response.status_code == 200


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is True


def test_predict():
    with TestClient(app) as client:
        _login(client)
        response = client.post("/predict", json=SAMPLE)
        assert response.status_code == 200
        assert response.json()["predicted_price"] > 0


def test_predict_batch():
    with TestClient(app) as client:
        _login(client)
        response = client.post("/predict/batch", json={"houses": [SAMPLE, SAMPLE]})
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 2


def test_invalid_input():
    with TestClient(app) as client:
        _login(client)
        response = client.post("/predict", json={**SAMPLE, "MedInc": -1})
        assert response.status_code == 422


def test_predict_requires_auth():
    with TestClient(app) as client:
        response = client.post("/predict", json=SAMPLE)
        assert response.status_code == 401
