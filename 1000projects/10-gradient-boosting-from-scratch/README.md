# Project 10 — Gradient Boosting Algorithm Implementation

## 1. Idea

Implement **gradient boosting for regression from scratch**. Each new small tree is fitted to the residuals (negative gradients) of the current ensemble. Use it to predict **diabetes disease progression one year ahead** from 10 patient measurements, and compare accuracy and speed with scikit-learn's `GradientBoostingRegressor`.

**What you learn**

- Boosting vs bagging
- Gradient descent in function space: residuals as negative gradients of squared loss
- Learning rate (shrinkage) and number of estimators trade-off
- Stochastic gradient boosting (row subsampling)
- Early stopping with a validation set
- Staged predictions and learning curves

**Dataset:** [Diabetes](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) (built into scikit-learn, 442 patients, 10 features). `scaled=False` gives raw units (age in years, BMI, blood pressure, blood serum values).

## 2. Algorithm

```text
F0(x) = mean(y)
for m = 1..M:
    r_i = y_i - F_{m-1}(x_i)                   # negative gradient of squared loss
    fit tree h_m on (x_i, r_i) using a random subsample of rows
    F_m(x) = F_{m-1}(x) + learning_rate * h_m(x)
stop early if validation loss has not improved for `patience` rounds
```

## 3. Folder Structure

```text
10-gradient-boosting-from-scratch/
├── src/
│   ├── __init__.py
│   ├── gbm.py      # GradientBoostingScratch
│   ├── train.py
│   └── api.py
└── models/
```

## 4. requirements.txt

```text
numpy>=2.1
scikit-learn>=1.7    # DecisionTreeRegressor as weak learner + comparison
joblib>=1.4
fastapi>=0.115
uvicorn[standard]>=0.32
pydantic>=2.9
```

## 5. Gradient Boosting Code — `src/gbm.py`

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor


class GradientBoostingScratch:
    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        min_samples_leaf: int = 5,
        subsample: float = 0.8,
        patience: int = 30,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.patience = patience
        self.random_state = random_state

    def fit(self, X, y, X_val=None, y_val=None):
        rng = np.random.default_rng(self.random_state)
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.init_ = float(y.mean())
        self.trees_ = []
        self.train_loss_, self.val_loss_ = [], []

        pred = np.full(len(y), self.init_)
        val_pred = None if X_val is None else np.full(len(y_val), self.init_)
        best_val, best_iter, rounds_without_gain = np.inf, 0, 0

        for m in range(self.n_estimators):
            residuals = y - pred
            n_sub = max(1, int(self.subsample * len(y)))
            idx = rng.choice(len(y), size=n_sub, replace=False)

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=int(rng.integers(1_000_000)),
            ).fit(X[idx], residuals[idx])
            self.trees_.append(tree)

            pred += self.learning_rate * tree.predict(X)
            self.train_loss_.append(float(np.mean((y - pred) ** 2)))

            if val_pred is not None:
                val_pred += self.learning_rate * tree.predict(X_val)
                val_loss = float(np.mean((y_val - val_pred) ** 2))
                self.val_loss_.append(val_loss)
                if val_loss < best_val - 1e-9:
                    best_val, best_iter, rounds_without_gain = val_loss, m + 1, 0
                else:
                    rounds_without_gain += 1
                    if rounds_without_gain >= self.patience:
                        break

        if val_pred is not None:
            self.trees_ = self.trees_[:best_iter]
        self.n_estimators_ = len(self.trees_)
        return self

    def staged_predict(self, X):
        X = np.asarray(X, float)
        pred = np.full(len(X), self.init_)
        for tree in self.trees_:
            pred = pred + self.learning_rate * tree.predict(X)
            yield pred

    def predict(self, X):
        X = np.asarray(X, float)
        pred = np.full(len(X), self.init_)
        for tree in self.trees_:
            pred += self.learning_rate * tree.predict(X)
        return pred

    @property
    def feature_importances_(self):
        total = np.sum([t.feature_importances_ for t in self.trees_], axis=0)
        return total / total.sum()
```

## 6. Training Code — `src/train.py`

```python
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.gbm import GradientBoostingScratch

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
FEATURES = ["age", "sex", "bmi", "bp", "s1_tc", "s2_ldl", "s3_hdl", "s4_tch", "s5_ltg", "s6_glu"]


def metrics(y_true, y_pred) -> dict:
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def main() -> None:
    X, y = load_diabetes(return_X_y=True, scaled=False)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

    start = time.perf_counter()
    ours = GradientBoostingScratch(n_estimators=1000, learning_rate=0.03, max_depth=2)
    ours.fit(X_train, y_train, X_val, y_val)
    our_time = time.perf_counter() - start
    our_metrics = metrics(y_test, ours.predict(X_test))
    print(f"Scratch GBM: {ours.n_estimators_} trees, {our_time:.2f}s, test {our_metrics}")

    start = time.perf_counter()
    sk = GradientBoostingRegressor(
        n_estimators=ours.n_estimators_, learning_rate=0.03, max_depth=2,
        min_samples_leaf=5, subsample=0.8, random_state=42,
    ).fit(X_train, y_train)
    sk_time = time.perf_counter() - start
    sk_metrics = metrics(y_test, sk.predict(X_test))
    print(f"sklearn GBM: {sk.n_estimators_} trees, {sk_time:.2f}s, test {sk_metrics}")

    baseline = metrics(y_test, np.full(len(y_test), y_train.mean()))
    print(f"Mean baseline: {baseline}")

    importances = dict(zip(FEATURES, ours.feature_importances_.round(4).tolist()))
    print("Feature importances:", importances)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(ours, MODELS_DIR / "model.joblib")
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "model_type": "GradientBoostingScratch",
                "features": FEATURES,
                "target": "Disease progression one year after baseline",
                "n_trees": ours.n_estimators_,
                "test_metrics": our_metrics,
                "sklearn_test_metrics": sk_metrics,
                "baseline_test_metrics": baseline,
                "train_time_seconds": {"scratch": round(our_time, 3), "sklearn": round(sk_time, 3)},
                "feature_importances": importances,
                "val_loss_curve": ours.val_loss_[::10],
            },
            indent=2,
        )
    )
    print("Saved models/model.joblib and models/metadata.json")


if __name__ == "__main__":
    main()
```

## 7. API Code — `src/api.py`

```python
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import src.gbm  # noqa: F401  (needed so joblib can unpickle GradientBoostingScratch)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"
state: dict = {}


class Patient(BaseModel):
    age: float = Field(..., ge=0, le=120, examples=[50])
    sex: Literal[1, 2] = Field(..., description="Encoded as in the dataset: 1 or 2")
    bmi: float = Field(..., gt=10, lt=60, examples=[27.5])
    bp: float = Field(..., gt=40, lt=200, description="Average blood pressure", examples=[95])
    s1_tc: float = Field(..., description="Total serum cholesterol", examples=[190])
    s2_ldl: float = Field(..., description="Low-density lipoproteins", examples=[115])
    s3_hdl: float = Field(..., description="High-density lipoproteins", examples=[48])
    s4_tch: float = Field(..., description="Total cholesterol / HDL", examples=[4.0])
    s5_ltg: float = Field(..., description="Log of serum triglycerides", examples=[4.6])
    s6_glu: float = Field(..., description="Blood sugar level", examples=[91])


@asynccontextmanager
async def lifespan(_: FastAPI):
    if MODEL_PATH.exists():
        state["model"] = joblib.load(MODEL_PATH)
        state["metadata"] = json.loads(METADATA_PATH.read_text())
    yield
    state.clear()


app = FastAPI(title="Diabetes Progression GBM API", lifespan=lifespan)


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


@app.post("/predict")
def predict(patient: Patient):
    model = get_model()
    x = np.array([[getattr(patient, f) for f in state["metadata"]["features"]]])
    staged = [float(p[0]) for p in model.staged_predict(x)]
    step = max(1, len(staged) // 10)
    return {
        "predicted_progression": round(staged[-1], 2),
        "n_trees": len(staged),
        "prediction_after_trees": {i + 1: round(staged[i], 2) for i in range(0, len(staged), step)},
    }
```

## 8. How to Run

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
  -Body '{"age":50,"sex":1,"bmi":32.1,"bp":101,"s1_tc":157,"s2_ldl":93.2,"s3_hdl":38,"s4_tch":4,"s5_ltg":4.86,"s6_glu":87}'
```

Expected results: test R² about 0.40 to 0.50, close to scikit-learn's result. The mean baseline is about 0.

## 9. Extension Ideas

1. Add absolute-error and Huber loss (change how residuals are computed).
2. Extend to binary classification with log-loss (fit trees to `y - sigmoid(F)`).
3. Write your own regression tree instead of using `DecisionTreeRegressor`.
4. Plot train vs validation loss per iteration for different learning rates.
5. Compare with XGBoost and LightGBM (Projects 73 and 74).
