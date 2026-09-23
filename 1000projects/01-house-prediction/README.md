# House Price Prediction (Project 01)

Predict median house prices from census-style features using **linear regression**, expose predictions through a **FastAPI** service, and keep a clear path for **retraining** and production deployment.

## What this project does

- Trains a scikit-learn pipeline (`StandardScaler` → `LinearRegression`) on the California Housing dataset (or your CSV).
- Saves artifacts to `models/` (`house_price_model.joblib`, `metadata.json`).
- Serves single and batch predictions with Pydantic validation.
- Uses a layered `app/` package (config, routes, ML, optional SQLAlchemy DB) suitable for growing into more models later.

**Dataset features:** `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`.

Further detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · Training: [docs/TRAINING.md](docs/TRAINING.md) · Auth: [docs/AUTH.md](docs/AUTH.md)

## Quick start (Windows)

```powershell
cd 01-house-prediction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

copy .env.example .env

python -m app.ml.house_price.train
uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for Swagger UI.

### Example prediction

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" `
  -Body '{"MedInc":8.3252,"HouseAge":41,"AveRooms":6.984,"AveBedrms":1.024,"Population":322,"AveOccup":2.556,"Latitude":37.88,"Longitude":-122.23}'
```

### Tests

Train first (tests expect a loaded model), then:

```powershell
pytest -q
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/auth/login` | Cookie session login |
| POST | `/auth/register` | New user |
| GET | `/auth/me` | Current user (cookie) |
| POST | `/auth/logout` | End session |
| GET | `/health` | API, DB, and model status |
| GET | `/model-info` | Training metadata and metrics |
| POST | `/predict` | Single prediction (**login required**) |
| POST | `/predict/batch` | Up to 1000 houses (**login required**) |

Default demo user (auto-created): `demo@example.com` / `demo-pass-123`

Legacy entrypoints still work: `uvicorn src.api:app`, `python src/train.py`.

## Folder structure

```text
01-house-prediction/
├── app/
│   ├── main.py                 # FastAPI app + lifespan
│   ├── core/config.py          # Settings from .env
│   ├── auth/                   # Cookie sessions, bcrypt, dependencies
│   ├── api/routes/             # auth, health, predictions
│   ├── ml/house_price/         # train, schemas, inference
│   └── db/                     # User, Session models
├── models/                     # Trained artifacts (gitignored binaries)
├── data/                       # Optional custom CSVs
├── tests/
├── docs/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

## Docker

Train locally so `models/` contains weights, then:

```powershell
docker build -t house-api .
docker run --rm -p 8000:8000 house-api
```

PostgreSQL stack:

```powershell
docker compose up --build
```

Set `DATABASE_URL` in `.env` for production PostgreSQL; SQLite is the default for local development.

## Learning goals

- Train/test split, cross-validation, regression metrics (RMSE, MAE, R²)
- Model persistence with `joblib`
- Production-style layout: config, services, training CLI, health checks
- Path to future models: extend `app/ml/house_price/train.py` and keep the same API schemas

## License

Part of the `1000projects` learning series.
