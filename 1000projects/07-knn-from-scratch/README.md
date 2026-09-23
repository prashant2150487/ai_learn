# Project 07 — KNN Algorithm Implementation

## 1. Idea

Implement **K-Nearest Neighbors from scratch** with NumPy, choose the best `k` and distance metric with hand-written cross-validation, and check that it matches scikit-learn's `KNeighborsClassifier`. Use it to classify **wines into 3 cultivars** from their chemical analysis. The API returns the predicted class and the actual nearest neighbors, which makes the prediction easy to explain.

**What you learn**

- Lazy learning: KNN has no training step, it memorizes the data
- Euclidean vs Manhattan distance
- Uniform vs distance-weighted voting
- Why feature scaling is essential for KNN
- Writing k-fold cross-validation by hand
- The bias-variance effect of small vs large `k`

**Dataset:** [Wine](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-recognition-dataset) (built into scikit-learn, 178 rows, 13 features, 3 classes).

## 2. Folder Structure

```text
07-knn-from-scratch/
├── src/
│   ├── __init__.py
│   ├── knn.py       # KNNClassifier and StandardScaler from scratch
│   ├── train.py
│   └── api.py
└── models/
```

## 3. requirements.txt

```text
numpy>=2.1
scikit-learn>=1.7    # only for dataset and comparison
joblib>=1.4
fastapi>=0.115
uvicorn[standard]>=0.32
pydantic>=2.9
```

## 4. KNN Code — `src/knn.py`

```python
import numpy as np


class Scaler:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_


class KNNClassifier:
    def __init__(self, k: int = 5, metric: str = "euclidean", weights: str = "uniform"):
        self.k = k
        self.metric = metric
        self.weights = weights

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y)
        self.classes_ = np.unique(self.y_)
        return self

    def _distances(self, X: np.ndarray) -> np.ndarray:
        diff = X[:, None, :] - self.X_[None, :, :]
        if self.metric == "manhattan":
            return np.abs(diff).sum(axis=2)
        return np.sqrt((diff ** 2).sum(axis=2))

    def kneighbors(self, X: np.ndarray):
        d = self._distances(np.asarray(X, dtype=float))
        idx = np.argsort(d, axis=1)[:, : self.k]
        return np.take_along_axis(d, idx, axis=1), idx

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        dist, idx = self.kneighbors(X)
        if self.weights == "distance":
            w = 1.0 / (dist + 1e-9)
        else:
            w = np.ones_like(dist)
        proba = np.zeros((len(idx), len(self.classes_)))
        for c_i, c in enumerate(self.classes_):
            proba[:, c_i] = (w * (self.y_[idx] == c)).sum(axis=1)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def k_fold_score(model_factory, X, y, n_splits=5, seed=42) -> float:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    folds = np.array_split(order, n_splits)
    scores = []
    for i in range(n_splits):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        scaler = Scaler().fit(X[train_idx])
        model = model_factory().fit(scaler.transform(X[train_idx]), y[train_idx])
        preds = model.predict(scaler.transform(X[val_idx]))
        scores.append((preds == y[val_idx]).mean())
    return float(np.mean(scores))
```

## 5. Training Code — `src/train.py`

```python
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from src.knn import KNNClassifier, Scaler, k_fold_score

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def main() -> None:
    wine = load_wine()
    X, y = wine.data, wine.target
    features = list(wine.feature_names)
    classes = wine.target_names.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    results = []
    for k in [1, 3, 5, 7, 9, 11, 15, 21]:
        for metric in ["euclidean", "manhattan"]:
            for weights in ["uniform", "distance"]:
                score = k_fold_score(lambda: KNNClassifier(k, metric, weights), X_train, y_train)
                results.append({"k": k, "metric": metric, "weights": weights, "cv_accuracy": score})
    results.sort(key=lambda r: r["cv_accuracy"], reverse=True)
    best = results[0]
    print("Top 5 configs:")
    for r in results[:5]:
        print(r)

    scaler = Scaler().fit(X_train)
    model = KNNClassifier(best["k"], best["metric"], best["weights"]).fit(
        scaler.transform(X_train), y_train
    )
    ours = model.predict(scaler.transform(X_test))
    our_acc = float((ours == y_test).mean())

    sk = KNeighborsClassifier(
        n_neighbors=best["k"], metric=best["metric"], weights=best["weights"]
    ).fit(scaler.transform(X_train), y_train)
    sk_preds = sk.predict(scaler.transform(X_test))
    agreement = float((ours == sk_preds).mean())

    print(f"Our KNN test accuracy:     {our_acc:.4f}")
    print(f"sklearn KNN test accuracy: {(sk_preds == y_test).mean():.4f}")
    print(f"Prediction agreement:      {agreement:.4f}")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"scaler": scaler, "model": model}, MODELS_DIR / "model.joblib")
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "model_type": "KNN (from scratch)",
                "features": features,
                "classes": classes,
                "best_config": best,
                "test_accuracy": our_acc,
                "agreement_with_sklearn": agreement,
                "search_results": results,
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

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import src.knn  # noqa: F401  (needed so joblib can unpickle KNNClassifier and Scaler)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"
state: dict = {}


class Wine(BaseModel):
    features: dict[str, float] = Field(..., description="All 13 wine feature names mapped to values")


@asynccontextmanager
async def lifespan(_: FastAPI):
    if MODEL_PATH.exists():
        state.update(joblib.load(MODEL_PATH))
        state["metadata"] = json.loads(METADATA_PATH.read_text())
    yield
    state.clear()


app = FastAPI(title="Wine KNN API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/model-info")
def model_info():
    if "metadata" not in state:
        raise HTTPException(503, "Model not loaded.")
    meta = state["metadata"]
    return {k: v for k, v in meta.items() if k != "search_results"}


@app.post("/predict")
def predict(wine: Wine):
    if "model" not in state:
        raise HTTPException(503, "Model not loaded. Run `python -m src.train` first.")
    meta = state["metadata"]
    missing = [f for f in meta["features"] if f not in wine.features]
    if missing:
        raise HTTPException(422, f"Missing features: {missing}")

    x = np.array([[wine.features[f] for f in meta["features"]]])
    x_s = state["scaler"].transform(x)
    model = state["model"]
    proba = model.predict_proba(x_s)[0]
    dist, idx = model.kneighbors(x_s)

    return {
        "cultivar": meta["classes"][int(proba.argmax())],
        "probabilities": {c: round(float(p), 4) for c, p in zip(meta["classes"], proba)},
        "neighbors": [
            {"train_index": int(i), "distance": round(float(d), 4), "cultivar": meta["classes"][int(model.y_[i])]}
            for d, i in zip(dist[0], idx[0])
        ],
    }
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

Get a request body from a real row:

```python
from sklearn.datasets import load_wine
import json
w = load_wine()
print(json.dumps({"features": dict(zip(w.feature_names, w.data[0].tolist()))}))
```

Expected test accuracy: about 95 to 100%, with 100% agreement with scikit-learn.

## 8. Extension Ideas

1. Add a KD-tree or ball tree to speed up neighbor search, and time it vs brute force.
2. Implement KNN regression (average of neighbor targets).
3. Plot accuracy vs `k` for train and validation to see over- and under-fitting.
4. Add cosine distance and try it on text TF-IDF vectors.
5. Use approximate nearest neighbors (FAISS) for a million-point dataset.
