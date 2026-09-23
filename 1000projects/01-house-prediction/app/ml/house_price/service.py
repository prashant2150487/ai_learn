import json
from typing import Any

import joblib
import pandas as pd
from fastapi import HTTPException

from app.core.config import settings
from app.ml.house_price.schemas import HouseFeatures, Prediction


class ModelRegistry:
    """In-memory model store loaded once at application startup."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._metadata: dict[str, Any] | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._metadata is not None

    def load(self) -> None:
        if not settings.model_path.exists():
            return
        self._model = joblib.load(settings.model_path)
        self._metadata = json.loads(settings.metadata_path.read_text(encoding="utf-8"))

    def clear(self) -> None:
        self._model = None
        self._metadata = None

    @property
    def metadata(self) -> dict[str, Any]:
        if self._metadata is None:
            raise HTTPException(status_code=503, detail="Model not loaded.")
        return self._metadata

    def predict(self, houses: list[HouseFeatures]) -> list[Prediction]:
        if not self.is_loaded:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Run training first: `python -m app.ml.house_price.train`.",
            )
        metadata = self._metadata
        assert metadata is not None
        df = pd.DataFrame([h.model_dump() for h in houses])[metadata["features"]]
        raw = self._model.predict(df) * metadata["price_scale"]
        return [
            Prediction(predicted_price=round(max(float(p), 0.0), 2))
            for p in raw
        ]


registry = ModelRegistry()
