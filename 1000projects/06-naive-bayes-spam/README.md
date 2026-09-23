# Project 06 — Naive Bayes Classifier for Spam Detection

## 1. Idea

Classify SMS messages as **spam or ham** (not spam) using text features and a Multinomial Naive Bayes model. The API takes raw text and returns the label, the spam probability, and the words that pushed the decision toward spam.

**What you learn**

- Bayes' theorem and the "naive" independence assumption
- Turning text into numbers: bag-of-words and TF-IDF
- N-grams, stop words, vocabulary size
- Laplace smoothing (`alpha`)
- Precision vs recall trade-off for spam (a false positive hides a real message)
- Explaining predictions with per-word log-probabilities

**Dataset:** [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (5,574 messages, about 13% spam). `train.py` downloads it automatically.

## 2. Workflow

1. Download and unzip the dataset into `data/`
2. Stratified train/test split
3. Pipeline: `TfidfVectorizer` then `MultinomialNB`
4. Grid search `ngram_range`, `min_df` and `alpha` (scoring = F1)
5. Evaluate: precision, recall, F1, confusion matrix
6. Print the most "spammy" words
7. Save the pipeline
8. API: `/predict` for one message, `/predict/batch` for many

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
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_FILE = DATA_DIR / "SMSSpamCollection"
MODELS_DIR = ROOT / "models"
URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        print(f"Downloading {URL}")
        DATA_DIR.mkdir(exist_ok=True)
        with urllib.request.urlopen(URL) as response:
            zipfile.ZipFile(io.BytesIO(response.read())).extract("SMSSpamCollection", DATA_DIR)
    df = pd.read_csv(DATA_FILE, sep="\t", header=None, names=["label", "text"], quoting=3)
    df["y"] = (df["label"] == "spam").astype(int)
    return df


def main() -> None:
    df = load_data()
    print(f"Messages: {len(df)}, spam ratio: {df['y'].mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["y"], test_size=0.2, stratify=df["y"], random_state=42
    )

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", sublinear_tf=True)),
            ("nb", MultinomialNB()),
        ]
    )
    search = GridSearchCV(
        pipeline,
        param_grid={
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__min_df": [1, 2],
            "nb__alpha": [0.01, 0.05, 0.1, 0.5, 1.0],
        },
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    print("Best params:", search.best_params_, f"CV F1={search.best_score_:.4f}")

    y_pred = model.predict(X_test)
    f1 = float(f1_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, target_names=["ham", "spam"], output_dict=True)
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
    print("Confusion matrix [[TN, FP], [FN, TP]]:\n", confusion_matrix(y_test, y_pred))

    vocab = np.array(model.named_steps["tfidf"].get_feature_names_out())
    log_prob = model.named_steps["nb"].feature_log_prob_
    spam_score = log_prob[1] - log_prob[0]
    top_spam = vocab[np.argsort(spam_score)[-20:][::-1]].tolist()
    print("Top spam words:", top_spam)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "model.joblib")
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "model_type": "TF-IDF + MultinomialNB",
                "best_params": {k: str(v) for k, v in search.best_params_.items()},
                "test_f1": f1,
                "test_report": report,
                "vocabulary_size": int(len(vocab)),
                "top_spam_words": top_spam,
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


class Message(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, examples=["WINNER!! Claim your free prize now, call 0800 123"])


class MessageBatch(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=1000)


class SpamPrediction(BaseModel):
    label: str
    spam_probability: float
    top_spam_terms: list[str]


@asynccontextmanager
async def lifespan(_: FastAPI):
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        state["model"] = model
        state["metadata"] = json.loads(METADATA_PATH.read_text())
        log_prob = model.named_steps["nb"].feature_log_prob_
        state["spam_score"] = log_prob[1] - log_prob[0]
        state["vocab"] = model.named_steps["tfidf"].get_feature_names_out()
    yield
    state.clear()


app = FastAPI(title="Spam Detection API", lifespan=lifespan)


def get_model():
    if "model" not in state:
        raise HTTPException(503, "Model not loaded. Run `python src/train.py` first.")
    return state["model"]


def classify(texts: list[str]) -> list[SpamPrediction]:
    model = get_model()
    probas = model.predict_proba(texts)[:, 1]
    tfidf = model.named_steps["tfidf"].transform(texts)
    results = []
    for i, proba in enumerate(probas):
        row = tfidf.getrow(i)
        contributions = row.data * state["spam_score"][row.indices]
        order = np.argsort(contributions)[::-1]
        terms = [str(state["vocab"][row.indices[j]]) for j in order[:5] if contributions[j] > 0]
        results.append(
            SpamPrediction(
                label="spam" if proba >= 0.5 else "ham",
                spam_probability=round(float(proba), 4),
                top_spam_terms=terms,
            )
        )
    return results


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/model-info")
def model_info():
    get_model()
    return state["metadata"]


@app.post("/predict", response_model=SpamPrediction)
def predict(message: Message):
    return classify([message.text])[0]


@app.post("/predict/batch", response_model=list[SpamPrediction])
def predict_batch(batch: MessageBatch):
    return classify(batch.texts)
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
  -Body '{"text":"Congratulations! You won a FREE iPhone. Text WIN to 80082 now"}'
```

Example response:

```json
{ "label": "spam", "spam_probability": 0.9987, "top_spam_terms": ["free", "won", "text", "win", "congratulations"] }
```

Expected test results: F1 on spam about 0.93 to 0.96.

## 7. Extension Ideas

1. Write Multinomial Naive Bayes from scratch (word counts + Laplace smoothing).
2. Compare with `ComplementNB`, logistic regression and linear SVM.
3. Add hand-crafted features: number of digits, URLs, capital letters ratio.
4. Apply the same pipeline to email spam (Enron dataset).
5. Add a feedback endpoint that stores user corrections for retraining.
