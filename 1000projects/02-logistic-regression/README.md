# Project 02 — Logistic Regression for Binary Classification

## 1. Idea

Predict whether a breast tumor is **malignant or benign** from 30 measurements of cell nuclei. Logistic regression outputs a **probability**, so the API returns both the predicted class and how confident the model is.

**What you learn**

- The sigmoid function and log-loss
- Class imbalance and stratified splitting
- Classification metrics: accuracy, precision, recall, F1, ROC-AUC
- Confusion matrix
- Choosing a decision threshold (for medical screening, recall matters more than precision)
- Regularization strength `C`

**Dataset:** [Breast Cancer Wisconsin](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset) (built into scikit-learn, 569 rows, 30 features, target `0 = malignant`, `1 = benign`).

## 2. Workflow

1. Load data and look at class balance
2. Stratified train/test split
3. Pipeline: `StandardScaler` then `LogisticRegression`
4. Tune `C` with 5-fold `GridSearchCV` (scoring = ROC-AUC)
5. Evaluate on test data, print confusion matrix
6. Pick a threshold that gives at least 98% recall on malignant cases
7. Save model, threshold and metadata
8. Serve with FastAPI

## 3. requirements.txt

```text
numpy>=2.1
pandas>=2.2
scikit-learn>=1.7
joblib>=1.4
fastapi>=0.115
uvicorn[standard]>=0.32
pydantic>=2.9
```

## 4. Training Code — `src/train.py`

```python
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def pick_threshold(y_true, malignant_proba, min_recall=0.98) -> float:
    """Highest threshold on P(malignant) that still catches `min_recall` of malignant cases."""
    precision, recall, thresholds = precision_recall_curve(y_true, malignant_proba)
    valid = np.where(recall[:-1] >= min_recall)[0]
    return float(thresholds[valid[-1]]) if len(valid) else 0.5


def main() -> None:
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    # Treat malignant as the positive class (1) because that is what we want to catch.
    y = 1 - y
    print(f"Rows: {len(X)}, malignant: {y.sum()}, benign: {(y == 0).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000)),
        ]
    )
    search = GridSearchCV(
        pipeline,
        param_grid={"model__C": [0.01, 0.1, 1, 10, 100]},
        cv=5,
        scoring="roc_auc",
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    print("Best params:", search.best_params_, f"CV AUC={search.best_score_:.4f}")

    train_proba = model.predict_proba(X_train)[:, 1]
    threshold = pick_threshold(y_train, train_proba)
    print(f"Chosen threshold on P(malignant): {threshold:.3f}")

    test_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (test_proba >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, test_proba),
    }
    metrics = {k: round(float(v), 4) for k, v in metrics.items()}
    print(json.dumps(metrics, indent=2))
    print("Confusion matrix [[TN, FP], [FN, TP]]:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["benign", "malignant"]))

    coefs = model.named_steps["model"].coef_[0]
    top = sorted(zip(X.columns, coefs), key=lambda t: abs(t[1]), reverse=True)[:10]
    print("Top 10 features:", [(n, round(float(c), 3)) for n, c in top])

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "model.joblib")
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "model_type": "LogisticRegression",
                "features": list(X.columns),
                "positive_class": "malignant",
                "threshold": threshold,
                "best_params": search.best_params_,
                "test_metrics": metrics,
            },
            indent=2,
        )
    )
    print("Saved models/model.joblib and models/metadata.json")


if __name__ == "__main__":
    main()
```

## 5. API Code — `src/api.py`

The dataset has 30 features, so the API accepts a dictionary of `feature_name: value` and checks that every feature is present.

```python
import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"
state: dict = {}


class TumorFeatures(BaseModel):
    features: dict[str, float] = Field(..., description="All 30 feature names mapped to values")


class TumorPrediction(BaseModel):
    label: str
    malignant_probability: float
    threshold: float


@asynccontextmanager
async def lifespan(_: FastAPI):
    if MODEL_PATH.exists():
        state["model"] = joblib.load(MODEL_PATH)
        state["metadata"] = json.loads(METADATA_PATH.read_text())
    yield
    state.clear()


app = FastAPI(title="Breast Cancer Classifier API", lifespan=lifespan)


def get_model():
    if "model" not in state:
        raise HTTPException(503, "Model not loaded. Run `python src/train.py` first.")
    return state["model"]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/model-info")
def model_info():
    get_model()
    return state["metadata"]


@app.get("/features")
def features():
    get_model()
    return {"features": state["metadata"]["features"]}


@app.post("/predict", response_model=TumorPrediction)
def predict(item: TumorFeatures):
    model = get_model()
    expected = state["metadata"]["features"]
    missing = [f for f in expected if f not in item.features]
    if missing:
        raise HTTPException(422, f"Missing features: {missing}")

    df = pd.DataFrame([item.features])[expected]
    proba = float(model.predict_proba(df)[0, 1])
    threshold = state["metadata"]["threshold"]
    return TumorPrediction(
        label="malignant" if proba >= threshold else "benign",
        malignant_probability=round(proba, 4),
        threshold=round(threshold, 4),
    )
```

## 6. How to Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/train.py
uvicorn src.api:app --reload
```

Get a valid request body from a real row:

```python
from sklearn.datasets import load_breast_cancer
import json
row = load_breast_cancer(as_frame=True).data.iloc[0].to_dict()
print(json.dumps({"features": row}))
```

Send it to `POST http://127.0.0.1:8000/predict`. Example response:

```json
{ "label": "malignant", "malignant_probability": 0.9998, "threshold": 0.21 }
```

## 7. Extension Ideas

1. Implement logistic regression from scratch with NumPy gradient descent and compare.
2. Plot the ROC and precision-recall curves.
3. Compare L1 vs L2 regularization and see which features L1 removes.
4. Calibrate probabilities with `CalibratedClassifierCV`.
5. Add SHAP explanations to the `/predict` response.
6. Try the [Pima Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) dataset with the same code.
