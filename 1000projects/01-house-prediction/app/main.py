from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.api.routes.auth import router as auth_router
from app.api.routes.health import router as health_router
from app.api.routes.predictions import router as predictions_router
from app.auth.service import ensure_bootstrap_user
from app.db.database import SessionLocal
from app.core.config import settings
from app.db import models
from app.db.database import Base, engine
from app.ml.house_price.service import registry


def _ensure_sqlite_parent() -> None:
    url = settings.database_url
    if url.startswith("sqlite:///"):
        db_path = Path(url.removeprefix("sqlite:///"))
        db_path.parent.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _ensure_sqlite_parent()
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        ensure_bootstrap_user(db)
    finally:
        db.close()
    registry.load()
    yield
    registry.clear()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Linear regression model for California Housing–style features.",
    lifespan=lifespan,
)

app.include_router(auth_router)
app.include_router(health_router)
app.include_router(predictions_router)


@app.get("/")
def root():
    return {
        "message": "House price prediction API",
        "docs": "/docs",
        "model_loaded": registry.is_loaded,
    }
