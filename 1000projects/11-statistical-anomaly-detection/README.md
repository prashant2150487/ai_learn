# Project 11 — Anomaly Detection Using Statistics

## 1. Idea

Detect abnormal **machine sensor readings** (temperature, pressure, vibration, power) using only statistics. No labels are needed during training. The model learns what "normal" looks like and flags readings that are too far from it. The API tells you whether a reading is anomalous, **which sensor** caused it, and **how extreme** it is.

**What you learn**

- Z-score and its weakness (outliers inflate the mean and std)
- Robust statistics: median and MAD (modified z-score)
- IQR / Tukey fences (the box-plot rule)
- Multivariate anomalies with Mahalanobis distance and the chi-square distribution
- Robust covariance estimation (Minimum Covariance Determinant)
- Evaluating detectors with precision, recall and F1 on injected anomalies

**Dataset:** synthetic correlated sensor data generated in `train.py`, with injected spikes, drifts and "impossible combinations" (each value normal on its own, but the combination is wrong). You can also point it at your own CSV.

## 2. Methods

| Method | Rule | Catches |
|---|---|---|
| Z-score | \|x − mean\| / std > 3 | Large single-sensor spikes |
| Modified z-score | 0.6745 · \|x − median\| / MAD > 3.5 | Spikes, robust to dirty training data |
| IQR | x < Q1 − 1.5·IQR or x > Q3 + 1.5·IQR | Spikes, no normality assumption |
| Mahalanobis | d² > chi²(0.999, df = n_features) | Unusual combinations across sensors |

## 3. requirements.txt

```text
numpy>=2.1
pandas>=2.2
scipy>=1.14
scikit-learn>=1.7
joblib>=1.4
fastapi>=0.115
uvicorn[standard]>=0.32
pydantic>=2.9
```

## 4. Training Code — `src/train.py`

```python
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.covariance import MinCovDet
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
FEATURES = ["temperature", "pressure", "vibration", "power"]


def generate_data(n_normal=5000, n_anomalies=150, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mean = [70, 30, 0.5, 220]
    cov = np.array(
        [
            [4.0, 1.2, 0.05, 6.0],
            [1.2, 1.0, 0.02, 2.0],
            [0.05, 0.02, 0.01, 0.1],
            [6.0, 2.0, 0.1, 25.0],
        ]
    )
    normal = rng.multivariate_normal(mean, cov, n_normal)

    anomalies = rng.multivariate_normal(mean, cov, n_anomalies)
    kinds = rng.integers(0, 3, n_anomalies)
    for i, kind in enumerate(kinds):
        if kind == 0:  # spike on one sensor
            j = rng.integers(0, 4)
            anomalies[i, j] += rng.choice([-1, 1]) * rng.uniform(5, 8) * np.sqrt(cov[j, j])
        elif kind == 1:  # drift on all sensors
            anomalies[i] += rng.uniform(2.5, 4) * np.sqrt(np.diag(cov))
        else:  # broken correlation: high temperature with low power
            anomalies[i, 0] += 2.5 * np.sqrt(cov[0, 0])
            anomalies[i, 3] -= 2.5 * np.sqrt(cov[3, 3])

    df = pd.DataFrame(np.vstack([normal, anomalies]), columns=FEATURES)
    df["is_anomaly"] = np.r_[np.zeros(n_normal), np.ones(n_anomalies)].astype(int)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def fit_stats(X: pd.DataFrame) -> dict:
    q1, q3 = X.quantile(0.25), X.quantile(0.75)
    median = X.median()
    mad = (X - median).abs().median()
    mcd = MinCovDet(random_state=42).fit(X.values)
    return {
        "features": list(X.columns),
        "mean": X.mean().tolist(),
        "std": X.std().tolist(),
        "median": median.tolist(),
        "mad": mad.replace(0, 1e-9).tolist(),
        "q1": q1.tolist(),
        "q3": q3.tolist(),
        "robust_location": mcd.location_.tolist(),
        "robust_precision": mcd.precision_.tolist(),
        "thresholds": {
            "z": 3.0,
            "modified_z": 3.5,
            "iqr_k": 1.5,
            "mahalanobis_sq": float(chi2.ppf(0.999, df=X.shape[1])),
        },
    }


def score(stats: dict, X: np.ndarray) -> dict[str, np.ndarray]:
    mean, std = np.array(stats["mean"]), np.array(stats["std"])
    median, mad = np.array(stats["median"]), np.array(stats["mad"])
    q1, q3 = np.array(stats["q1"]), np.array(stats["q3"])
    t = stats["thresholds"]

    z = np.abs(X - mean) / std
    mz = 0.6745 * np.abs(X - median) / mad
    iqr = q3 - q1
    iqr_flag = (X < q1 - t["iqr_k"] * iqr) | (X > q3 + t["iqr_k"] * iqr)
    diff = X - np.array(stats["robust_location"])
    maha_sq = np.einsum("ij,jk,ik->i", diff, np.array(stats["robust_precision"]), diff)

    return {
        "z": z,
        "modified_z": mz,
        "iqr_flag": iqr_flag,
        "mahalanobis_sq": maha_sq,
        "pred_z": (z > t["z"]).any(axis=1),
        "pred_modified_z": (mz > t["modified_z"]).any(axis=1),
        "pred_iqr": iqr_flag.any(axis=1),
        "pred_mahalanobis": maha_sq > t["mahalanobis_sq"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="Optional CSV of normal readings (numeric columns only)")
    args = parser.parse_args()

    if args.csv:
        train = pd.read_csv(args.csv).select_dtypes("number").dropna()
        stats = fit_stats(train)
        evaluation = None
    else:
        df = generate_data()
        split = int(len(df) * 0.7)
        # Train on a slice that still contains some anomalies, like real-world dirty data.
        train, test = df.iloc[:split], df.iloc[split:]
        stats = fit_stats(train[FEATURES])

        scores = score(stats, test[FEATURES].values)
        y = test["is_anomaly"].values
        evaluation = {}
        for method in ["z", "modified_z", "iqr", "mahalanobis"]:
            pred = scores[f"pred_{method}"]
            evaluation[method] = {
                "precision": round(float(precision_score(y, pred)), 4),
                "recall": round(float(recall_score(y, pred)), 4),
                "f1": round(float(f1_score(y, pred)), 4),
            }
        combined = scores["pred_modified_z"] | scores["pred_mahalanobis"]
        evaluation["modified_z_or_mahalanobis"] = {
            "precision": round(float(precision_score(y, combined)), 4),
            "recall": round(float(recall_score(y, combined)), 4),
            "f1": round(float(f1_score(y, combined)), 4),
        }
        print(json.dumps(evaluation, indent=2))

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(stats, MODELS_DIR / "model.joblib")
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps({"model_type": "Statistical anomaly detector", "stats": stats, "evaluation": evaluation}, indent=2)
    )
    print("Saved models/model.joblib and models/metadata.json")


if __name__ == "__main__":
    main()
```

## 5. API Code — `src/api.py`

```python
import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"
state: dict = {}


class Reading(BaseModel):
    values: dict[str, float] = Field(
        ..., examples=[{"temperature": 85.0, "pressure": 30.5, "vibration": 0.52, "power": 221.0}]
    )


class ReadingBatch(BaseModel):
    readings: list[Reading] = Field(..., min_length=1, max_length=10_000)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if MODEL_PATH.exists():
        state["stats"] = joblib.load(MODEL_PATH)
        state["metadata"] = json.loads(METADATA_PATH.read_text())
    yield
    state.clear()


app = FastAPI(title="Statistical Anomaly Detection API", lifespan=lifespan)


def detect(readings: list[Reading]) -> list[dict]:
    if "stats" not in state:
        raise HTTPException(503, "Model not loaded. Run `python src/train.py` first.")
    stats = state["stats"]
    features = stats["features"]
    for r in readings:
        missing = [f for f in features if f not in r.values]
        if missing:
            raise HTTPException(422, f"Missing features: {missing}")

    X = np.array([[r.values[f] for f in features] for r in readings])
    median, mad = np.array(stats["median"]), np.array(stats["mad"])
    mz = 0.6745 * np.abs(X - median) / mad
    diff = X - np.array(stats["robust_location"])
    maha_sq = np.einsum("ij,jk,ik->i", diff, np.array(stats["robust_precision"]), diff)
    t = stats["thresholds"]

    results = []
    for i in range(len(X)):
        sensor_flags = {f: round(float(mz[i, j]), 3) for j, f in enumerate(features) if mz[i, j] > t["modified_z"]}
        combo_flag = bool(maha_sq[i] > t["mahalanobis_sq"])
        results.append(
            {
                "is_anomaly": bool(sensor_flags) or combo_flag,
                "anomalous_sensors": sensor_flags,
                "unusual_combination": combo_flag,
                "mahalanobis_sq": round(float(maha_sq[i]), 3),
                "mahalanobis_threshold": round(t["mahalanobis_sq"], 3),
                "worst_sensor": features[int(mz[i].argmax())],
            }
        )
    return results


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "stats" in state}


@app.get("/model-info")
def model_info():
    if "metadata" not in state:
        raise HTTPException(503, "Model not loaded.")
    return state["metadata"]


@app.post("/detect")
def detect_one(reading: Reading):
    return detect([reading])[0]


@app.post("/detect/batch")
def detect_batch(batch: ReadingBatch):
    results = detect(batch.readings)
    return {"n_anomalies": sum(r["is_anomaly"] for r in results), "results": results}
```

## 6. How to Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/train.py
uvicorn src.api:app --reload
```

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/detect `
  -ContentType "application/json" `
  -Body '{"values":{"temperature":76,"pressure":30,"vibration":0.5,"power":208}}'
```

That reading has a temperature and a power value that are each almost normal alone, but high temperature with low power is an unusual combination. Mahalanobis catches it, while the per-sensor rules do not.

Expected results: modified z-score has high precision on spikes, Mahalanobis catches the broken-correlation cases, and combining both gives the best F1.

## 7. Extension Ideas

1. Add Grubbs' test and the generalized ESD test for small samples.
2. Add a rolling-window z-score for streaming time series data.
3. Compare with Isolation Forest and Local Outlier Factor.
4. Add a `/fit` endpoint that recalculates statistics from a new batch of normal data.
5. Build a Streamlit dashboard that plots incoming readings and flags anomalies live.
