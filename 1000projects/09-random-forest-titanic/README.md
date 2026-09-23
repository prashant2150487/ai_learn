# Project 09 — Random Forest Classifier

## 1. Idea

Predict whether a **Titanic passenger survived** from ticket class, sex, age, family size, fare and port of embarkation. The dataset is real and messy (missing ages, text names), so it is a good introduction to a **full preprocessing pipeline** feeding a random forest.

**What you learn**

- Bagging and random feature subsets: why a forest beats a single tree
- Out-of-bag (OOB) score as free validation
- Handling missing values and categorical columns with `ColumnTransformer`
- Feature engineering (title from name, family size, is-alone)
- Avoiding data leakage (dropping `boat` and `body`, which reveal the answer)
- Randomized hyperparameter search
- Impurity vs permutation feature importance

**Dataset:** [Titanic on OpenML](https://www.openml.org/d/40945) (1,309 passengers). `fetch_openml` downloads and caches it.

## 2. Folder Structure

```text
09-random-forest-titanic/
├── src/
│   ├── __init__.py
│   ├── features.py   # feature engineering shared by train and api
│   ├── train.py
│   └── api.py
└── models/
```

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

## 4. Feature Code — `src/features.py`

```python
import numpy as np
import pandas as pd

NUMERIC = ["pclass", "age", "sibsp", "parch", "fare", "family_size", "is_alone"]
CATEGORICAL = ["sex", "embarked", "title"]

TITLE_MAP = {
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Lady": "Rare", "Countess": "Rare", "Capt": "Rare", "Col": "Rare", "Don": "Rare",
    "Dr": "Rare", "Major": "Rare", "Rev": "Rare", "Sir": "Rare", "Jonkheer": "Rare", "Dona": "Rare",
}


def extract_title(name: str | None, sex: str) -> str:
    if isinstance(name, str) and "," in name and "." in name:
        raw = name.split(",", 1)[1].split(".", 1)[0].strip()
        return TITLE_MAP.get(raw, raw if raw in {"Mr", "Mrs", "Miss", "Master"} else "Rare")
    return "Mr" if sex == "male" else "Miss"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["pclass", "age", "sibsp", "parch", "fare"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ["sex", "embarked"]:
        out[col] = out[col].astype(object).where(out[col].notna(), np.nan)
    out["family_size"] = out["sibsp"] + out["parch"] + 1
    out["is_alone"] = (out["family_size"] == 1).astype(int)
    names = out["name"] if "name" in out else pd.Series([None] * len(out), index=out.index)
    out["title"] = [extract_title(n, s) for n, s in zip(names, out["sex"])]
    return out[NUMERIC + CATEGORICAL]
```

## 5. Training Code — `src/train.py`

```python
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.features import CATEGORICAL, NUMERIC, build_features

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def main() -> None:
    titanic = fetch_openml("titanic", version=1, as_frame=True)
    raw = titanic.frame
    y = raw["survived"].astype(str).astype(int)
    # `boat` and `body` leak the outcome; they are never passed to build_features.
    X = build_features(raw[["pclass", "name", "sex", "age", "sibsp", "parch", "fare", "embarked"]])
    print(f"Passengers: {len(X)}, survival rate: {y.mean():.3f}")
    print("Missing values:\n", X.isna().sum()[X.isna().sum() > 0])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocess = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), NUMERIC),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL,
            ),
        ]
    )
    pipeline = Pipeline(
        [
            ("prep", preprocess),
            ("rf", RandomForestClassifier(oob_score=True, random_state=42, n_jobs=-1)),
        ]
    )

    search = RandomizedSearchCV(
        pipeline,
        param_distributions={
            "rf__n_estimators": [200, 300, 500, 800],
            "rf__max_depth": [4, 6, 8, 10, None],
            "rf__min_samples_leaf": [1, 2, 4, 8],
            "rf__max_features": ["sqrt", "log2", 0.5],
        },
        n_iter=30,
        cv=5,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    print("Best params:", search.best_params_, f"CV AUC={search.best_score_:.4f}")
    print(f"OOB score: {model.named_steps['rf'].oob_score_:.4f}")

    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
    }
    print(metrics)
    print(classification_report(y_test, y_pred, target_names=["died", "survived"]))

    perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, scoring="roc_auc")
    importance = dict(
        sorted(
            zip(X.columns, perm.importances_mean.round(4).tolist()),
            key=lambda t: t[1],
            reverse=True,
        )
    )
    print("Permutation importance:", importance)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "model.joblib")
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "model_type": "RandomForestClassifier",
                "numeric_features": NUMERIC,
                "categorical_features": CATEGORICAL,
                "best_params": {k: str(v) for k, v in search.best_params_.items()},
                "oob_score": float(model.named_steps["rf"].oob_score_),
                "test_metrics": metrics,
                "permutation_importance": importance,
            },
            indent=2,
        )
    )
    print("Saved models/model.joblib and models/metadata.json")


if __name__ == "__main__":
    main()
```

## 6. API Code — `src/api.py`

```python
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features import build_features

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"
state: dict = {}


class Passenger(BaseModel):
    pclass: Literal[1, 2, 3]
    sex: Literal["male", "female"]
    age: float | None = Field(None, ge=0, le=100)
    sibsp: int = Field(0, ge=0, description="Siblings/spouses aboard")
    parch: int = Field(0, ge=0, description="Parents/children aboard")
    fare: float | None = Field(None, ge=0)
    embarked: Literal["S", "C", "Q"] | None = None
    name: str | None = Field(None, examples=["Smith, Mrs. Jane"])


class SurvivalPrediction(BaseModel):
    survived: bool
    survival_probability: float


@asynccontextmanager
async def lifespan(_: FastAPI):
    if MODEL_PATH.exists():
        state["model"] = joblib.load(MODEL_PATH)
        state["metadata"] = json.loads(METADATA_PATH.read_text())
    yield
    state.clear()


app = FastAPI(title="Titanic Survival API", lifespan=lifespan)


def get_model():
    if "model" not in state:
        raise HTTPException(503, "Model not loaded. Run `python -m src.train` first.")
    return state["model"]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/model-info")
def model_info():
    get_model()
    return state["metadata"]


@app.post("/predict", response_model=SurvivalPrediction)
def predict(passenger: Passenger):
    model = get_model()
    X = build_features(pd.DataFrame([passenger.model_dump()]))
    proba = float(model.predict_proba(X)[0, 1])
    return SurvivalPrediction(survived=proba >= 0.5, survival_probability=round(proba, 4))


@app.post("/predict/batch", response_model=list[SurvivalPrediction])
def predict_batch(passengers: list[Passenger]):
    model = get_model()
    X = build_features(pd.DataFrame([p.model_dump() for p in passengers]))
    return [
        SurvivalPrediction(survived=p >= 0.5, survival_probability=round(float(p), 4))
        for p in model.predict_proba(X)[:, 1]
    ]
```

## 7. How to Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
New-Item -ItemType File src\__init__.py -Force
python -m src.train
uvicorn src.api:app --reload
```

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" `
  -Body '{"pclass":1,"sex":"female","age":29,"sibsp":0,"parch":0,"fare":211.3,"embarked":"S","name":"Allen, Miss. Elisabeth"}'
```

Expected results: test accuracy about 80 to 83%, ROC-AUC about 0.85 to 0.88.

## 8. Extension Ideas

1. Plot OOB error vs number of trees.
2. Compare impurity importance vs permutation importance and explain the difference.
3. Add `cabin` deck letter as a feature.
4. Compare with `ExtraTreesClassifier` and gradient boosting.
5. Add SHAP values to explain each passenger's prediction.
