# Model training

This project separates **training** (offline batch job) from **serving** (FastAPI). Artifacts are written to `models/` and loaded at API startup.

## Default dataset

Uses scikit-learn [California Housing](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) (20,640 rows, 8 numeric features). Target values are in units of **$100,000**; the API multiplies predictions by `price_scale` from metadata so clients see USD.

## Train commands

From the project root with the virtual environment active:

```powershell
# Recommended
python -m app.ml.house_price.train

# Legacy shim (same behavior)
python src/train.py

# Console script after pip install -e .
train-house-model
```

### Custom CSV

Place your file under `data/` (numeric feature columns + one target column):

```powershell
python -m app.ml.house_price.train --csv data/houses.csv --target price
```

Optional flags:

| Flag | Default | Purpose |
|------|---------|---------|
| `--test-size` | `0.2` | Hold-out fraction |
| `--seed` | `42` | Reproducible split |

## Outputs

| File | Description |
|------|-------------|
| `models/house_price_model.joblib` | `StandardScaler` + `LinearRegression` pipeline |
| `models/metadata.json` | Features, metrics, coefficients, training timestamp |

After training, restart the API (or redeploy the container) so the new weights are loaded.

## Pipeline (current)

1. `StandardScaler` on features  
2. `LinearRegression`  
3. 5-fold cross-validation R² on the training split  
4. Test-set RMSE, MAE, R²  

## Future extensions

The layout under `app/ml/house_price/` is meant to grow without breaking the API:

- Swap `build_pipeline()` for Ridge, RandomForest, or gradient boosting  
- Add experiment tracking (MLflow, W&B) in `train.py`  
- Version models (`models/v2/...`) and select via `MODEL_DIR` env  
- Scheduled retraining (cron, GitHub Actions, Airflow) calling the same module  

Keep **schemas** (`schemas.py`) stable for clients; put new algorithms behind the same `registry.predict()` interface.
