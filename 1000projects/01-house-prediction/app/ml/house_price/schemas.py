from pydantic import BaseModel, Field


class HouseFeatures(BaseModel):
    MedInc: float = Field(..., gt=0, description="Median income in block group (tens of thousands USD)")
    HouseAge: float = Field(..., ge=0, description="Median house age in block group (years)")
    AveRooms: float = Field(..., gt=0, description="Average rooms per household")
    AveBedrms: float = Field(..., gt=0, description="Average bedrooms per household")
    Population: float = Field(..., ge=0, description="Block group population")
    AveOccup: float = Field(..., gt=0, description="Average household members")
    Latitude: float = Field(..., ge=32, le=42)
    Longitude: float = Field(..., ge=-125, le=-114)

    model_config = {
        "json_schema_extra": {
            "example": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.984,
                "AveBedrms": 1.024,
                "Population": 322.0,
                "AveOccup": 2.556,
                "Latitude": 37.88,
                "Longitude": -122.23,
            }
        }
    }


class Prediction(BaseModel):
    predicted_price: float
    currency: str = "USD"


class BatchRequest(BaseModel):
    houses: list[HouseFeatures] = Field(..., min_length=1, max_length=1000)


class BatchPrediction(BaseModel):
    predictions: list[Prediction]
