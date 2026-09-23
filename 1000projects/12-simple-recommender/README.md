# Project 12 — Simple Recommendation System

## 1. Idea

Build a **movie recommender** with item-based collaborative filtering: two movies are similar if the same people rate them similarly. The API can find movies similar to a given movie, recommend movies for an existing user, and recommend for a **new user** who just picks a few movies they like. A popularity fallback handles the cold-start case.

**What you learn**

- The user-item rating matrix and sparsity
- Mean-centering ratings to remove user bias
- Cosine similarity between items
- Item-based vs user-based collaborative filtering
- Popularity baseline with a Bayesian average (so a movie with one 5-star rating does not top the list)
- Offline evaluation with hold-out data: precision@10 and hit rate@10
- The cold-start problem

**Dataset:** [MovieLens Latest Small](https://grouplens.org/datasets/movielens/latest/) (100k ratings, 610 users, 9.7k movies). `train.py` downloads it automatically.

## 2. Workflow

1. Download ratings and movies
2. Keep movies with at least 10 ratings (makes similarities reliable and memory small)
3. Hold out 20% of each user's liked movies (rating ≥ 4) as a test set
4. Build the mean-centered user-item sparse matrix from training ratings
5. Compute item-item cosine similarity
6. Score candidates for each user: weighted sum of similarities to the movies they rated
7. Evaluate precision@10 and hit rate@10 vs the popularity baseline
8. Rebuild on all data and save
9. API: `/similar/{movie_id}`, `/recommend/user/{user_id}`, `/recommend` (new user), `/search`

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
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
MIN_RATINGS = 10
TOP_K = 10


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings_path = DATA_DIR / "ml-latest-small" / "ratings.csv"
    if not ratings_path.exists():
        print(f"Downloading {URL}")
        DATA_DIR.mkdir(exist_ok=True)
        with urllib.request.urlopen(URL) as response:
            zipfile.ZipFile(io.BytesIO(response.read())).extractall(DATA_DIR)
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(DATA_DIR / "ml-latest-small" / "movies.csv")
    return ratings, movies


def build_model(ratings: pd.DataFrame, movie_ids: np.ndarray) -> dict:
    user_ids = np.sort(ratings["userId"].unique())
    u_index = {u: i for i, u in enumerate(user_ids)}
    m_index = {m: i for i, m in enumerate(movie_ids)}

    user_mean = ratings.groupby("userId")["rating"].mean()
    centered = ratings["rating"].values - ratings["userId"].map(user_mean).values

    matrix = csr_matrix(
        (centered, (ratings["userId"].map(u_index), ratings["movieId"].map(m_index))),
        shape=(len(user_ids), len(movie_ids)),
    )
    similarity = cosine_similarity(matrix.T, dense_output=True).astype(np.float32)
    np.fill_diagonal(similarity, 0.0)

    stats = ratings.groupby("movieId")["rating"].agg(["count", "mean"]).reindex(movie_ids)
    global_mean, prior = ratings["rating"].mean(), 20
    popularity = ((stats["count"] * stats["mean"] + prior * global_mean) / (stats["count"] + prior)).values

    return {
        "movie_ids": movie_ids,
        "movie_index": m_index,
        "user_ids": user_ids,
        "user_index": u_index,
        "similarity": similarity,
        "popularity": popularity.astype(np.float32),
        "user_ratings": {
            int(u): dict(zip(g["movieId"].astype(int), g["rating"] - user_mean[u]))
            for u, g in ratings.groupby("userId")
        },
    }


def score_items(model: dict, rated: dict[int, float]) -> np.ndarray:
    """Weighted sum of similarities to rated movies, weighted by (mean-centered) rating."""
    idx = [model["movie_index"][m] for m in rated if m in model["movie_index"]]
    if not idx:
        return model["popularity"].copy()
    weights = np.array([rated[m] for m in rated if m in model["movie_index"]], dtype=np.float32)
    sims = model["similarity"][idx]
    scores = weights @ sims / (np.abs(sims).sum(axis=0) + 1e-6)
    scores[idx] = -np.inf
    return scores


def evaluate(train: pd.DataFrame, test: pd.DataFrame, movie_ids: np.ndarray) -> dict:
    model = build_model(train, movie_ids)
    test_sets = test.groupby("userId")["movieId"].apply(set)
    hits = precisions = 0.0
    pop_hits = pop_precisions = 0.0

    for user, relevant in test_sets.items():
        rated = model["user_ratings"].get(int(user), {})
        scores = score_items(model, rated)
        top = set(movie_ids[np.argsort(scores)[::-1][:TOP_K]].tolist())

        pop = model["popularity"].copy()
        pop[[model["movie_index"][m] for m in rated]] = -np.inf
        pop_top = set(movie_ids[np.argsort(pop)[::-1][:TOP_K]].tolist())

        precisions += len(top & relevant) / TOP_K
        hits += bool(top & relevant)
        pop_precisions += len(pop_top & relevant) / TOP_K
        pop_hits += bool(pop_top & relevant)

    n = len(test_sets)
    return {
        "users_evaluated": n,
        f"item_cf_precision@{TOP_K}": round(precisions / n, 4),
        f"item_cf_hit_rate@{TOP_K}": round(hits / n, 4),
        f"popularity_precision@{TOP_K}": round(pop_precisions / n, 4),
        f"popularity_hit_rate@{TOP_K}": round(pop_hits / n, 4),
    }


def main() -> None:
    ratings, movies = load_data()
    counts = ratings["movieId"].value_counts()
    keep = counts[counts >= MIN_RATINGS].index
    ratings = ratings[ratings["movieId"].isin(keep)].reset_index(drop=True)
    movie_ids = np.sort(ratings["movieId"].unique())
    print(f"Ratings: {len(ratings)}, users: {ratings['userId'].nunique()}, movies: {len(movie_ids)}")

    liked = ratings[ratings["rating"] >= 4]
    test = liked.groupby("userId", group_keys=False).sample(frac=0.2, random_state=42)
    train = ratings.drop(test.index)
    metrics = evaluate(train, test, movie_ids)
    print(json.dumps(metrics, indent=2))

    model = build_model(ratings, movie_ids)
    info = movies.set_index("movieId").reindex(movie_ids)
    model["titles"] = info["title"].tolist()
    model["genres"] = info["genres"].tolist()

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "model.joblib", compress=3)
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "model_type": "Item-based collaborative filtering (cosine, mean-centered)",
                "n_users": int(len(model["user_ids"])),
                "n_movies": int(len(movie_ids)),
                "min_ratings_per_movie": MIN_RATINGS,
                "evaluation": metrics,
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
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"
state: dict = {}


class NewUserRequest(BaseModel):
    ratings: dict[int, float] = Field(
        ..., description="movieId mapped to rating 0.5-5", examples=[{"1": 5, "260": 4.5, "1196": 5}]
    )
    top_n: int = Field(10, ge=1, le=50)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if MODEL_PATH.exists():
        state["model"] = joblib.load(MODEL_PATH)
        state["metadata"] = json.loads(METADATA_PATH.read_text())
    yield
    state.clear()


app = FastAPI(title="Movie Recommender API", lifespan=lifespan)


def get_model() -> dict:
    if "model" not in state:
        raise HTTPException(503, "Model not loaded. Run `python src/train.py` first.")
    return state["model"]


def movie_card(model: dict, i: int, score: float | None = None) -> dict:
    card = {"movie_id": int(model["movie_ids"][i]), "title": model["titles"][i], "genres": model["genres"][i]}
    if score is not None:
        card["score"] = round(float(score), 4)
    return card


def score_items(model: dict, rated: dict[int, float]) -> np.ndarray:
    known = [m for m in rated if m in model["movie_index"]]
    if not known:
        return model["popularity"].copy()
    idx = [model["movie_index"][m] for m in known]
    weights = np.array([rated[m] for m in known], dtype=np.float32)
    sims = model["similarity"][idx]
    scores = weights @ sims / (np.abs(sims).sum(axis=0) + 1e-6)
    scores[idx] = -np.inf
    return scores


def top_n(model: dict, scores: np.ndarray, n: int) -> list[dict]:
    order = np.argsort(scores)[::-1][:n]
    return [movie_card(model, int(i), scores[i]) for i in order]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/model-info")
def model_info():
    get_model()
    return state["metadata"]


@app.get("/search")
def search(q: str = Query(..., min_length=2), limit: int = Query(10, le=50)):
    model = get_model()
    q = q.lower()
    matches = [i for i, t in enumerate(model["titles"]) if isinstance(t, str) and q in t.lower()]
    return [movie_card(model, i) for i in matches[:limit]]


@app.get("/similar/{movie_id}")
def similar(movie_id: int, n: int = Query(10, ge=1, le=50)):
    model = get_model()
    if movie_id not in model["movie_index"]:
        raise HTTPException(404, "Unknown movie (or fewer than 10 ratings).")
    i = model["movie_index"][movie_id]
    return {"movie": movie_card(model, i), "similar": top_n(model, model["similarity"][i].copy(), n)}


@app.get("/recommend/user/{user_id}")
def recommend_user(user_id: int, n: int = Query(10, ge=1, le=50)):
    model = get_model()
    rated = model["user_ratings"].get(user_id)
    if rated is None:
        raise HTTPException(404, "Unknown user. Use POST /recommend for new users.")
    return {"user_id": user_id, "recommendations": top_n(model, score_items(model, rated), n)}


@app.post("/recommend")
def recommend_new_user(request: NewUserRequest):
    model = get_model()
    ratings = {m: r for m, r in request.ratings.items() if 0.5 <= r <= 5}
    if not ratings:
        return {"strategy": "popularity", "recommendations": top_n(model, model["popularity"].copy(), request.top_n)}
    mean = float(np.mean(list(ratings.values())))
    centered = {m: r - mean for m, r in ratings.items()}
    # If all ratings are equal, centering gives zeros; treat every rated movie as a "like".
    if all(abs(v) < 1e-9 for v in centered.values()):
        centered = {m: 1.0 for m in ratings}
    return {"strategy": "item_cf", "recommendations": top_n(model, score_items(model, centered), request.top_n)}
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
# Find a movie ID
Invoke-RestMethod "http://127.0.0.1:8000/search?q=toy story"

# Movies similar to Toy Story (movieId 1)
Invoke-RestMethod "http://127.0.0.1:8000/similar/1?n=5"

# Recommendations for existing user 1
Invoke-RestMethod "http://127.0.0.1:8000/recommend/user/1"

# New user who likes Star Wars and The Matrix, dislikes Titanic
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/recommend `
  -ContentType "application/json" `
  -Body '{"ratings":{"260":5,"2571":5,"1721":1},"top_n":10}'
```

Expected results: item-based CF should beat the popularity baseline on hit rate@10. Exact numbers depend on the split.

## 7. Extension Ideas

1. Add user-based collaborative filtering and compare.
2. Add content-based similarity from genres (TF-IDF on the genre string) and blend it with CF (hybrid).
3. Implement matrix factorization with ALS or SGD (Projects 324 and 345).
4. Add diversity: penalize recommending many movies from the same genre.
5. Add a feedback endpoint and retrain nightly.
