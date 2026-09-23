# Project 04 — Decision Tree Classifier for the Iris Dataset

## 1. Idea

Classify iris flowers into **setosa, versicolor or virginica** from petal and sepal measurements using a decision tree. Decision trees are easy to explain, so the API also returns the **exact rules** the tree used to make its decision.

**What you learn**

- How a tree splits data (Gini impurity, entropy)
- Overfitting and how `max_depth` and `min_samples_leaf` control it
- Hyperparameter tuning with `GridSearchCV`
- Feature importance
- Extracting readable rules and the decision path for one prediction

**Dataset:** [Iris](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset) (built into scikit-learn, 150 rows, 4 features, 3 classes).

## 2. Workflow

1. Load Iris
2. Stratified train/test split
3. Grid search over `criterion`, `max_depth`, `min_samples_leaf`
4. Evaluate with accuracy, classification report and confusion matrix
5. Print the tree as text rules and feature importances
6. Save model + rules
7. API returns class, probabilities and the decision path

## 3. requirements.txt

```text
numpy>=2.1
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
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def main() -> None:
    iris = load_iris()
    X, y = iris.data, iris.target
    classes = iris.target_names.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    search = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid={
            "criterion": ["gini", "entropy"],
            "max_depth": [2, 3, 4, 5, None],
            "min_samples_leaf": [1, 2, 4],
        },
        cv=5,
        scoring="accuracy",
    )
    search.fit(X_train, y_train)
    tree = search.best_estimator_
    print("Best params:", search.best_params_, f"CV accuracy={search.best_score_:.4f}")

    y_pred = tree.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    print(f"Test accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=classes))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    rules = export_text(tree, feature_names=FEATURES)
    print("Tree rules:\n", rules)

    importances = dict(zip(FEATURES, tree.feature_importances_.round(4).tolist()))
    print("Feature importances:", importances)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(tree, MODELS_DIR / "model.joblib")
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "model_type": "DecisionTreeClassifier",
                "features": FEATURES,
                "classes": classes,
                "best_params": search.best_params_,
                "test_accuracy": accuracy,
                "feature_importances": importances,
                "depth": int(tree.get_depth()),
                "n_leaves": int(tree.get_n_leaves()),
                "rules": rules,
            },
            indent=2,
        )
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


class Flower(BaseModel):
    sepal_length: float = Field(..., gt=0, le=10, examples=[5.1])
    sepal_width: float = Field(..., gt=0, le=10, examples=[3.5])
    petal_length: float = Field(..., gt=0, le=10, examples=[1.4])
    petal_width: float = Field(..., gt=0, le=10, examples=[0.2])


class FlowerPrediction(BaseModel):
    species: str
    probabilities: dict[str, float]
    decision_path: list[str]


@asynccontextmanager
async def lifespan(_: FastAPI):
    if MODEL_PATH.exists():
        state["model"] = joblib.load(MODEL_PATH)
        state["metadata"] = json.loads(METADATA_PATH.read_text())
    yield
    state.clear()


app = FastAPI(title="Iris Decision Tree API", lifespan=lifespan)


def get_model():
    if "model" not in state:
        raise HTTPException(503, "Model not loaded. Run `python src/train.py` first.")
    return state["model"]


def explain(tree, x: np.ndarray, features: list[str]) -> list[str]:
    node_ids = tree.decision_path(x).indices
    leaf = tree.apply(x)[0]
    steps = []
    for node in node_ids:
        if node == leaf:
            break
        feat = tree.tree_.feature[node]
        threshold = tree.tree_.threshold[node]
        value = x[0, feat]
        sign = "<=" if value <= threshold else ">"
        steps.append(f"{features[feat]} = {value:.2f} {sign} {threshold:.2f}")
    return steps


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/model-info")
def model_info():
    get_model()
    return state["metadata"]


@app.get("/rules", response_model=str)
def rules():
    get_model()
    return state["metadata"]["rules"]


@app.post("/predict", response_model=FlowerPrediction)
def predict(flower: Flower):
    tree = get_model()
    features = state["metadata"]["features"]
    classes = state["metadata"]["classes"]
    x = np.array([[getattr(flower, f) for f in features]])
    proba = tree.predict_proba(x)[0]
    return FlowerPrediction(
        species=classes[int(proba.argmax())],
        probabilities={c: round(float(p), 4) for c, p in zip(classes, proba)},
        decision_path=explain(tree, x, features),
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

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" `
  -Body '{"sepal_length":6.3,"sepal_width":2.8,"petal_length":5.1,"petal_width":1.5}'
```

Example response:

```json
{
  "species": "virginica",
  "probabilities": { "setosa": 0.0, "versicolor": 0.33, "virginica": 0.67 },
  "decision_path": ["petal_length = 5.10 > 2.45", "petal_width = 1.50 <= 1.75", "petal_length = 5.10 > 4.95"]
}
```

## 7. Extension Ideas

1. Visualize the tree with `sklearn.tree.plot_tree` and save it as PNG.
2. Implement a decision tree from scratch (recursive Gini splitting).
3. Show how accuracy changes with `max_depth` (plot train vs test).
4. Try cost-complexity pruning with `ccp_alpha`.
5. Compare with a random forest (see Project 09).
