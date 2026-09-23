# Architecture

## Overview

```text
Client → FastAPI (app.main) → ModelRegistry → joblib pipeline
                ↓
         SQLAlchemy (optional persistence / health)
```

| Layer | Location | Role |
|-------|----------|------|
| HTTP | `app/main.py`, `app/api/routes/` | Routing, lifespan, OpenAPI |
| Config | `app/core/config.py` | Env-based settings (DB, model paths) |
| ML serving | `app/ml/house_price/service.py` | Load model once; batch inference |
| ML training | `app/ml/house_price/train.py` | Offline training CLI |
| DB | `app/db/` | SQLAlchemy engine, sample `User` model, health probe |

## Request flow (prediction)

1. Startup: `registry.load()` reads `models/house_price_model.joblib` and `metadata.json`.  
2. `POST /predict`: Pydantic validates `HouseFeatures`.  
3. Features are ordered per `metadata["features"]`, scaled and predicted in the pipeline.  
4. Response price is scaled to USD and clamped to ≥ 0.

## Configuration

Environment variables (see `.env.example`):

| Variable | Default | Notes |
|----------|---------|--------|
| `DATABASE_URL` | `sqlite:///./data/app.db` | Use PostgreSQL in Docker/production |
| `DEBUG` | `false` | Reserved for future logging levels |

Model paths default to `models/` at the project root; override by extending `Settings` if you add env fields later.

## Deployment

- **Local**: `uvicorn app.main:app --reload`  
- **Docker**: `docker build -t house-api .` (train on host first so `models/` exists)  
- **Compose**: `docker compose up --build` (API + PostgreSQL; mount trained `models/`)

## Backward compatibility

- `uvicorn src.api:app` re-exports `app.main:app`.  
- `python src/train.py` delegates to `app.ml.house_price.train`.
