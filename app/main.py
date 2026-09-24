"""
FastAPI app: health check, model info, single predict, batch predict.
"""

import logging
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from src.config import load_config
from src.logging_config import setup_logging
from src.models.predictor import ArtifactLoadError, get_predictor
from src.pipeline import PredictionPipelineError, predict
from src.validation.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    OrderRequest,
    PredictionResponse,
)

setup_logging()
logger = logging.getLogger(__name__)

api_config = load_config()["api"]

app = FastAPI(title=api_config["title"], version="0.1.0")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "request | method=%s | path=%s | status=%s | latency_ms=%.1f",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
    )
    return response


_startup_error = None


@app.on_event("startup")
def load_model_on_startup():
    global _startup_error
    try:
        get_predictor()
        logger.info("model loaded successfully, service ready")
    except ArtifactLoadError as exc:
        _startup_error = str(exc)
        logger.error("failed to load model artifacts at startup: %s", exc)


@app.exception_handler(PredictionPipelineError)
def handle_pipeline_error(request: Request, exc: PredictionPipelineError):
    return JSONResponse(
        status_code=422, content={"detail": f"could not generate a prediction: {exc}"}
    )


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    if _startup_error is not None:
        return HealthResponse(
            status="unhealthy", model_loaded=False, model_name="unknown", model_version="unknown"
        )
    predictor = get_predictor()
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_name=predictor.model_name,
        model_version=predictor.model_version,
    )


@app.get("/model/info", tags=["ops"])
def model_info():
    if _startup_error is not None:
        raise HTTPException(status_code=503, detail=_startup_error)
    predictor = get_predictor()
    return {
        "model_name": predictor.model_name,
        "model_version": predictor.model_version,
        "decision_threshold": predictor.threshold,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict_single(order: OrderRequest):
    if _startup_error is not None:
        raise HTTPException(status_code=503, detail="model not loaded: " + _startup_error)
    return predict(order)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"])
def predict_batch(batch: BatchPredictionRequest):
    if _startup_error is not None:
        raise HTTPException(status_code=503, detail="model not loaded: " + _startup_error)
    return BatchPredictionResponse(predictions=[predict(order) for order in batch.orders])
