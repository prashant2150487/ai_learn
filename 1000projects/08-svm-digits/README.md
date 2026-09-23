# Project 08 — Support Vector Machine for Classification

## 1. Idea

Recognize **handwritten digits (0–9)** with a Support Vector Machine. The API accepts either the 64 raw pixel values or an **uploaded image**. Uploaded images are resized to 8×8 grayscale so you can draw a digit in Paint and test the model.

**What you learn**

- Maximum-margin classifiers and support vectors
- The kernel trick (linear, polynomial, RBF)
- Hyperparameters `C` (margin softness) and `gamma` (kernel width)
- One-vs-rest vs one-vs-one multi-class strategy
- Probability estimates from SVMs (Platt scaling)
- Image preprocessing for a small-image model

**Dataset:** [Digits](https://scikit-learn.org/stable/datasets/toy_dataset.html#optical-recognition-of-handwritten-digits-dataset) (built into scikit-learn, 1,797 images of 8×8 pixels, values 0–16).

## 2. Workflow

1. Load digits and flatten to 64 features
2. Stratified train/test split
3. Pipeline: `StandardScaler` then `SVC`
4. Grid search kernels, `C` and `gamma`
5. Evaluate: accuracy, per-class report, confusion matrix
6. Save the pipeline
7. API: `/predict` (64 pixels) and `/predict-image` (file upload)

## 3. requirements.txt

```text
numpy>=2.1
scikit-learn>=1.7
joblib>=1.4
pillow>=11.0
fastapi>=0.115
uvicorn[standard]>=0.32
pydantic>=2.9
python-multipart>=0.0.12
```

## 4. Training Code — `src/train.py`

```python
import json
from pathlib import Path

import joblib
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def main() -> None:
    digits = load_digits()
    X, y = digits.data, digits.target
    print(f"Images: {len(X)}, pixels per image: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(probability=True, random_state=42)),
        ]
    )
    search = GridSearchCV(
        pipeline,
        param_grid=[
            {"svc__kernel": ["linear"], "svc__C": [0.01, 0.1, 1]},
            {"svc__kernel": ["rbf"], "svc__C": [1, 10, 100], "svc__gamma": ["scale", 0.001, 0.01]},
            {"svc__kernel": ["poly"], "svc__C": [1, 10], "svc__degree": [2, 3]},
        ],
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    print("Best params:", search.best_params_, f"CV accuracy={search.best_score_:.4f}")

    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    print(f"Test accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    svc = model.named_steps["svc"]
    print(f"Support vectors per class: {svc.n_support_.tolist()}")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "model.joblib")
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "model_type": "SVC",
                "input": "64 pixel values (8x8), each 0-16, row-major, dark ink = high value",
                "best_params": {k: str(v) for k, v in search.best_params_.items()},
                "test_accuracy": accuracy,
                "support_vectors_per_class": svc.n_support_.tolist(),
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
import io
import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"
state: dict = {}


class Pixels(BaseModel):
    pixels: list[float] = Field(..., min_length=64, max_length=64, description="64 values in 0-16")


class DigitPrediction(BaseModel):
    digit: int
    confidence: float
    top3: list[dict]


@asynccontextmanager
async def lifespan(_: FastAPI):
    if MODEL_PATH.exists():
        state["model"] = joblib.load(MODEL_PATH)
        state["metadata"] = json.loads(METADATA_PATH.read_text())
    yield
    state.clear()


app = FastAPI(title="SVM Digit Recognition API", lifespan=lifespan)


def classify(x: np.ndarray) -> DigitPrediction:
    if "model" not in state:
        raise HTTPException(503, "Model not loaded. Run `python src/train.py` first.")
    proba = state["model"].predict_proba(x.reshape(1, -1))[0]
    order = np.argsort(proba)[::-1]
    return DigitPrediction(
        digit=int(order[0]),
        confidence=round(float(proba[order[0]]), 4),
        top3=[{"digit": int(i), "probability": round(float(proba[i]), 4)} for i in order[:3]],
    )


def image_to_pixels(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("L")
    # Training digits are light ink (high values) on dark background; typical drawings are the opposite.
    if np.asarray(img).mean() > 127:
        img = ImageOps.invert(img)
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    img = ImageOps.pad(img, (max(img.size),) * 2, color=0)
    img = img.resize((8, 8), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=float)
    return (arr / 255.0 * 16.0).ravel()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/model-info")
def model_info():
    if "metadata" not in state:
        raise HTTPException(503, "Model not loaded.")
    return state["metadata"]


@app.post("/predict", response_model=DigitPrediction)
def predict(item: Pixels):
    return classify(np.array(item.pixels))


@app.post("/predict-image", response_model=DigitPrediction)
async def predict_image(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(415, "Upload a PNG or JPEG image.")
    return classify(image_to_pixels(await file.read()))
```

## 6. How to Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/train.py
uvicorn src.api:app --reload
```

Draw a digit in Paint (black on white), save as `digit.png`, then:

```bash
curl -X POST http://127.0.0.1:8000/predict-image -F "file=@digit.png"
```

Or open `http://127.0.0.1:8000/docs` and use the upload button on `/predict-image`.

Expected test accuracy: about 98 to 99% with the RBF kernel.

## 7. Extension Ideas

1. Visualize the support vectors as 8×8 images.
2. Compare kernels by plotting accuracy vs `C` and `gamma` as a heatmap.
3. Train on MNIST (28×28) with `LinearSVC` and PCA to keep it fast.
4. Build a small HTML canvas page that posts drawings to `/predict-image`.
5. Implement a linear SVM from scratch with hinge loss and sub-gradient descent.
