from fastapi import APIRouter

from app.auth.dependencies import CurrentUser
from app.ml.house_price.schemas import (
    BatchPrediction,
    BatchRequest,
    HouseFeatures,
    Prediction,
)
from app.ml.house_price.service import registry

router = APIRouter(tags=["Predictions"])


@router.get("/model-info")
def model_info():
    return registry.metadata


@router.post("/predict", response_model=Prediction)
def predict(house: HouseFeatures, _user: CurrentUser):
    return registry.predict([house])[0]


@router.post("/predict/batch", response_model=BatchPrediction)
def predict_batch(request: BatchRequest, _user: CurrentUser):
    return BatchPrediction(predictions=registry.predict(request.houses))
