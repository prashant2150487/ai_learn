"""Train and persist the house price regression model.

Usage:
    python -m app.ml.house_price.train
    python -m app.ml.house_price.train --csv data/houses.csv --target price
"""

import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.core.config import settings

CALIFORNIA_PRICE_SCALE = 100_000


def load_data(csv_path: str | None, target: str) -> tuple[pd.DataFrame, pd.Series, float]:
    if csv_path:
        df = pd.read_csv(csv_path).dropna()
        X = df.drop(columns=[target]).select_dtypes(include="number")
        y = df[target]
        return X, y, 1.0

    data = fetch_california_housing(as_frame=True)
    return data.data, data.target, CALIFORNIA_PRICE_SCALE


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train house price model")
    parser.add_argument("--csv", help="Optional CSV file with your own data")
    parser.add_argument("--target", default="price", help="Target column name in the CSV")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X, y, price_scale = load_data(args.csv, args.target)
    print(f"Loaded {len(X)} rows, {X.shape[1]} features: {list(X.columns)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    pipeline = build_pipeline()

    cv_r2 = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2")
    print(f"5-fold CV R2: {cv_r2.mean():.4f} (+/- {cv_r2.std():.4f})")

    pipeline.fit(X_train, y_train)

    train_metrics = evaluate(y_train, pipeline.predict(X_train))
    test_metrics = evaluate(y_test, pipeline.predict(X_test))
    print("Train metrics:", json.dumps(train_metrics, indent=2))
    print("Test metrics: ", json.dumps(test_metrics, indent=2))

    coefficients = dict(
        zip(X.columns, pipeline.named_steps["model"].coef_.round(4).tolist())
    )
    print("Coefficients (on standardized features):", json.dumps(coefficients, indent=2))

    settings.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, settings.model_path)

    metadata = {
        "model_type": "LinearRegression",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.csv or "sklearn.california_housing",
        "features": list(X.columns),
        "price_scale": price_scale,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "cv_r2_mean": float(cv_r2.mean()),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "coefficients": coefficients,
    }
    settings.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved model to {settings.model_path}")
    print(f"Saved metadata to {settings.metadata_path}")


if __name__ == "__main__":
    main()
