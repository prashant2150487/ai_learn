# Project 05 — Simple Neural Network from Scratch

## 1. Idea

Build a small **multi-layer neural network using only NumPy**: no PyTorch, no TensorFlow. Train it to separate two interleaving half-moons, a shape that a linear model cannot learn. Save the learned weights and serve predictions through an API.

**What you learn**

- Forward pass: matrix multiply, bias, activation
- ReLU and sigmoid activations
- Binary cross-entropy loss
- Backpropagation (chain rule) written by hand
- Mini-batch gradient descent, learning rate, epochs
- He weight initialization
- Why hidden layers let a model learn non-linear boundaries

**Dataset:** `sklearn.datasets.make_moons` (synthetic, 2 features, 2 classes).

## 2. Architecture

```text
input (2) → Dense(16) → ReLU → Dense(16) → ReLU → Dense(1) → Sigmoid → P(class = 1)
```

## 3. Folder Structure

```text
05-neural-network-from-scratch/
├── src/
│   ├── __init__.py
│   ├── nn.py        # the neural network class (shared by train and api)
│   ├── train.py
│   └── api.py
└── models/
    ├── weights.npz
    └── metadata.json
```

## 4. requirements.txt

```text
numpy>=2.1
scikit-learn>=1.7    # only for the dataset and train/test split
fastapi>=0.115
uvicorn[standard]>=0.32
pydantic>=2.9
```

## 5. Neural Network Code — `src/nn.py`

```python
import numpy as np


def relu(z):
    return np.maximum(0, z)


def relu_grad(z):
    return (z > 0).astype(z.dtype)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class NeuralNetwork:
    def __init__(self, layer_sizes: list[int], seed: int = 42):
        rng = np.random.default_rng(seed)
        self.layer_sizes = layer_sizes
        self.weights = [
            rng.normal(0, np.sqrt(2.0 / n_in), size=(n_in, n_out))
            for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
        self.biases = [np.zeros((1, n_out)) for n_out in layer_sizes[1:]]

    def forward(self, X: np.ndarray):
        activations, pre_activations = [X], []
        a = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            pre_activations.append(z)
            a = sigmoid(z) if i == len(self.weights) - 1 else relu(z)
            activations.append(a)
        return activations, pre_activations

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)[0][-1].ravel()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    @staticmethod
    def loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        eps = 1e-12
        y_prob = np.clip(y_prob, eps, 1 - eps)
        return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))

    def backward(self, activations, pre_activations, y: np.ndarray):
        m = y.shape[0]
        grads_W = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        # Sigmoid + binary cross-entropy simplifies to (prediction - target).
        delta = activations[-1] - y.reshape(-1, 1)
        for i in reversed(range(len(self.weights))):
            grads_W[i] = activations[i].T @ delta / m
            grads_b[i] = delta.mean(axis=0, keepdims=True)
            if i > 0:
                delta = (delta @ self.weights[i].T) * relu_grad(pre_activations[i - 1])
        return grads_W, grads_b

    def fit(self, X, y, epochs=500, lr=0.05, batch_size=32, X_val=None, y_val=None, seed=42):
        rng = np.random.default_rng(seed)
        history = []
        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(X))
            for start in range(0, len(X), batch_size):
                idx = order[start : start + batch_size]
                acts, pres = self.forward(X[idx])
                gW, gb = self.backward(acts, pres, y[idx])
                for i in range(len(self.weights)):
                    self.weights[i] -= lr * gW[i]
                    self.biases[i] -= lr * gb[i]

            if epoch % 50 == 0 or epoch == 1:
                record = {"epoch": epoch, "train_loss": self.loss(y, self.predict_proba(X))}
                if X_val is not None:
                    record["val_loss"] = self.loss(y_val, self.predict_proba(X_val))
                    record["val_acc"] = float((self.predict(X_val) == y_val).mean())
                history.append(record)
                print(record)
        return history

    def save(self, path):
        arrays = {f"W{i}": W for i, W in enumerate(self.weights)}
        arrays.update({f"b{i}": b for i, b in enumerate(self.biases)})
        np.savez(path, layer_sizes=np.array(self.layer_sizes), **arrays)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        net = cls(data["layer_sizes"].tolist())
        net.weights = [data[f"W{i}"] for i in range(len(net.weights))]
        net.biases = [data[f"b{i}"] for i in range(len(net.biases))]
        return net
```

## 6. Training Code — `src/train.py`

```python
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from src.nn import NeuralNetwork

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def main() -> None:
    X, y = make_moons(n_samples=2000, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Standardize using training statistics only.
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train_s = (X_train - mean) / std
    X_test_s = (X_test - mean) / std

    net = NeuralNetwork([2, 16, 16, 1])
    history = net.fit(
        X_train_s, y_train, epochs=500, lr=0.05, batch_size=32,
        X_val=X_test_s, y_val=y_test,
    )

    test_acc = float((net.predict(X_test_s) == y_test).mean())
    print(f"Test accuracy: {test_acc:.4f}")

    MODELS_DIR.mkdir(exist_ok=True)
    net.save(MODELS_DIR / "weights.npz")
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "model_type": "NumPy MLP",
                "layers": net.layer_sizes,
                "features": ["x1", "x2"],
                "scaler_mean": mean.tolist(),
                "scaler_std": std.tolist(),
                "test_accuracy": test_acc,
                "history": history,
            },
            indent=2,
        )
    )
    print("Saved models/weights.npz and models/metadata.json")


if __name__ == "__main__":
    main()
```

## 7. API Code — `src/api.py`

```python
import json
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.nn import NeuralNetwork

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = ROOT / "models" / "weights.npz"
METADATA_PATH = ROOT / "models" / "metadata.json"
state: dict = {}


class Point(BaseModel):
    x1: float = Field(..., examples=[0.5])
    x2: float = Field(..., examples=[-0.3])


class PointBatch(BaseModel):
    points: list[Point] = Field(..., min_length=1, max_length=10_000)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if WEIGHTS_PATH.exists():
        state["model"] = NeuralNetwork.load(WEIGHTS_PATH)
        state["metadata"] = json.loads(METADATA_PATH.read_text())
    yield
    state.clear()


app = FastAPI(title="NumPy Neural Network API", lifespan=lifespan)


def scaled(points: list[Point]) -> np.ndarray:
    if "model" not in state:
        raise HTTPException(503, "Model not loaded. Run `python -m src.train` first.")
    meta = state["metadata"]
    X = np.array([[p.x1, p.x2] for p in points])
    return (X - np.array(meta["scaler_mean"])) / np.array(meta["scaler_std"])


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/model-info")
def model_info():
    if "metadata" not in state:
        raise HTTPException(503, "Model not loaded.")
    return {k: v for k, v in state["metadata"].items() if k != "history"}


@app.post("/predict")
def predict(point: Point):
    X = scaled([point])
    proba = float(state["model"].predict_proba(X)[0])
    return {"class": int(proba >= 0.5), "probability_class_1": round(proba, 4)}


@app.post("/predict/batch")
def predict_batch(batch: PointBatch):
    X = scaled(batch.points)
    probas = state["model"].predict_proba(X)
    return {
        "predictions": [
            {"class": int(p >= 0.5), "probability_class_1": round(float(p), 4)} for p in probas
        ]
    }
```

## 8. How to Run

Because `train.py` and `api.py` both import `src.nn`, run them as modules from the project folder:

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
  -ContentType "application/json" -Body '{"x1":1.0,"x2":-0.4}'
```

Expected test accuracy: about 97%.

## 9. Extension Ideas

1. Add a gradient check: compare backprop gradients to numerical gradients.
2. Add momentum and the Adam optimizer.
3. Add L2 regularization and dropout.
4. Extend to softmax output for multi-class (try `make_blobs` or the digits dataset).
5. Plot the decision boundary after every 50 epochs and make a GIF.
6. Rebuild the same network in PyTorch and compare the code.
