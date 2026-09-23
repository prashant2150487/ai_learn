# Project 03 — K-Means Clustering for Customer Segmentation

## 1. Idea

Group customers into segments based on **age, annual income and spending score**, then give each segment a business-friendly name (for example "High income, low spenders"). The API assigns a new customer to a segment so marketing can target them.

**What you learn**

- Unsupervised learning (no labels)
- Why scaling matters for distance-based algorithms
- Choosing `k` with the elbow method (inertia) and silhouette score
- Interpreting cluster centers
- Turning clusters into business personas

**Dataset:** the [Mall Customers dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) (`Mall_Customers.csv`). If you don't have the file, `train.py` generates a realistic synthetic version so you can start immediately.

## 2. Workflow

1. Load `data/Mall_Customers.csv` or generate synthetic customers
2. Scale features with `StandardScaler`
3. Try `k = 2..10`, record inertia and silhouette score
4. Pick `k` with the best silhouette score
5. Fit final K-Means, describe each cluster by its average age, income and spending
6. Auto-name each segment
7. Save pipeline + segment profiles
8. API: `/segment` for one customer, `/segments` to list all personas

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
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Mall_Customers.csv"
MODELS_DIR = ROOT / "models"
FEATURES = ["age", "annual_income_k", "spending_score"]


def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        return df.rename(
            columns={
                "Age": "age",
                "Annual Income (k$)": "annual_income_k",
                "Spending Score (1-100)": "spending_score",
            }
        )[FEATURES]

    print("Mall_Customers.csv not found, generating synthetic customers")
    rng = np.random.default_rng(42)
    groups = [  # (age_mean, income_mean, spend_mean, count)
        (25, 25, 80, 40),
        (45, 25, 20, 40),
        (33, 85, 85, 40),
        (42, 88, 15, 40),
        (40, 55, 50, 60),
    ]
    frames = []
    for age, income, spend, n in groups:
        frames.append(
            pd.DataFrame(
                {
                    "age": rng.normal(age, 6, n).clip(18, 70).round(),
                    "annual_income_k": rng.normal(income, 8, n).clip(10, 140).round(),
                    "spending_score": rng.normal(spend, 8, n).clip(1, 100).round(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def name_segment(center: pd.Series, overall: pd.Series) -> str:
    income = "High income" if center["annual_income_k"] > overall["annual_income_k"] * 1.15 else (
        "Low income" if center["annual_income_k"] < overall["annual_income_k"] * 0.85 else "Mid income"
    )
    spend = "high spenders" if center["spending_score"] > overall["spending_score"] * 1.15 else (
        "low spenders" if center["spending_score"] < overall["spending_score"] * 0.85 else "average spenders"
    )
    age = "young" if center["age"] < 32 else ("senior" if center["age"] > 50 else "middle-aged")
    return f"{income}, {spend} ({age})"


def main() -> None:
    df = load_data()
    print(f"Customers: {len(df)}")
    X = df[FEATURES]

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    scores = {}
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
        scores[k] = {
            "inertia": float(km.inertia_),
            "silhouette": float(silhouette_score(X_scaled, km.labels_)),
        }
        print(f"k={k:2d}  inertia={km.inertia_:8.1f}  silhouette={scores[k]['silhouette']:.3f}")

    best_k = max(scores, key=lambda k: scores[k]["silhouette"])
    print(f"Best k by silhouette: {best_k}")

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=best_k, n_init=10, random_state=42)),
        ]
    ).fit(X)

    df["segment"] = pipeline.predict(X)
    overall = X.mean()
    profiles = {}
    for seg, group in df.groupby("segment"):
        center = group[FEATURES].mean()
        profiles[int(seg)] = {
            "name": name_segment(center, overall),
            "size": int(len(group)),
            "avg_age": round(float(center["age"]), 1),
            "avg_income_k": round(float(center["annual_income_k"]), 1),
            "avg_spending_score": round(float(center["spending_score"]), 1),
        }
    print(json.dumps(profiles, indent=2))

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / "model.joblib")
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "model_type": "KMeans",
                "features": FEATURES,
                "k": best_k,
                "k_search": scores,
                "segments": profiles,
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
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"
state: dict = {}


class Customer(BaseModel):
    age: float = Field(..., ge=18, le=100)
    annual_income_k: float = Field(..., ge=0, description="Annual income in thousands of USD")
    spending_score: float = Field(..., ge=1, le=100)


class SegmentResult(BaseModel):
    segment_id: int
    segment_name: str
    distance_to_center: float
    profile: dict


@asynccontextmanager
async def lifespan(_: FastAPI):
    if MODEL_PATH.exists():
        state["model"] = joblib.load(MODEL_PATH)
        state["metadata"] = json.loads(METADATA_PATH.read_text())
    yield
    state.clear()


app = FastAPI(title="Customer Segmentation API", lifespan=lifespan)


def get_model():
    if "model" not in state:
        raise HTTPException(503, "Model not loaded. Run `python src/train.py` first.")
    return state["model"]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/segments")
def segments():
    get_model()
    return state["metadata"]["segments"]


@app.post("/segment", response_model=SegmentResult)
def segment(customer: Customer):
    model = get_model()
    df = pd.DataFrame([customer.model_dump()])[state["metadata"]["features"]]
    seg = int(model.predict(df)[0])
    scaled = model.named_steps["scaler"].transform(df)
    center = model.named_steps["kmeans"].cluster_centers_[seg]
    profile = state["metadata"]["segments"][str(seg)]
    return SegmentResult(
        segment_id=seg,
        segment_name=profile["name"],
        distance_to_center=round(float(np.linalg.norm(scaled[0] - center)), 4),
        profile=profile,
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
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/segment `
  -ContentType "application/json" `
  -Body '{"age":28,"annual_income_k":90,"spending_score":88}'
```

Example response:

```json
{
  "segment_id": 2,
  "segment_name": "High income, high spenders (middle-aged)",
  "distance_to_center": 0.41,
  "profile": { "size": 40, "avg_age": 33.1, "avg_income_k": 85.4, "avg_spending_score": 84.9 }
}
```

## 7. Extension Ideas

1. Plot clusters in 2D (income vs spending) and 3D with matplotlib.
2. Compare with DBSCAN, Gaussian Mixture and hierarchical clustering.
3. Add RFM features (recency, frequency, monetary) from a transaction dataset.
4. Build a Streamlit page where marketers explore each segment.
5. Add a `/segment/batch` endpoint that accepts a CSV upload.
