"""Minimal FastAPI serving stub for extraction inference."""

from fastapi import FastAPI
from pydantic import BaseModel

from .predict import predict_single

app = FastAPI(title="docextract-lora API")


class PredictRequest(BaseModel):
    """Input schema for prediction requests."""

    prompt: str
    model_name_or_path: str = "checkpoints/docextract-lora"


class PredictResponse(BaseModel):
    """Output schema for prediction responses."""

    prediction: str


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health status."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Run prediction for one request payload."""
    prediction: str = predict_single(
        prompt=request.prompt,
        model_name_or_path=request.model_name_or_path,
    )
    return PredictResponse(prediction=prediction)
