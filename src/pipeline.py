"""
Glues feature engineering and the predictor together, one function in, one
object out.
"""

import json
import logging
import time
from datetime import datetime, timezone

from src.config import PROJECT_ROOT, load_config
from src.features.engineer import build_features
from src.models.predictor import get_predictor
from src.monitoring.metrics import record_prediction
from src.validation.schemas import OrderRequest, PredictionResponse

logger = logging.getLogger(__name__)

PREDICTIONS_LOG_PATH = PROJECT_ROOT / "logs" / "predictions.jsonl"


def _log_prediction_record(order: OrderRequest, result: dict) -> None:
    PREDICTIONS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "order_id": order.order_id,
        "is_late": result["is_late"],
        "probability": result["probability"],
        "threshold_used": result["threshold_used"],
        "model_name": result["model_name"],
        "model_version": result["model_version"],
    }
    with open(PREDICTIONS_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


class PredictionPipelineError(Exception):
    """Raised for problems with a single request. Caught by the API layer,
    returned as a clean error, never crashes the service."""


def predict(order: OrderRequest) -> PredictionResponse:
    start = time.perf_counter()
    features_config = load_config()["features"]

    try:
        raw = order.model_dump()
        features_df = build_features(
            raw_order=raw,
            top_states=get_predictor().top_states,
            top_seller_states=get_predictor().top_seller_states,
            high_risk_months=features_config["high_risk_months"],
        )
        result = get_predictor().predict_one(features_df)
        record_prediction(result["is_late"])
        _log_prediction_record(order, result)

    except Exception as exc:
        logger.exception("prediction failed for order_id=%s: %s", order.order_id, exc)
        raise PredictionPipelineError(str(exc)) from exc

    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "prediction served | order_id=%s | is_late=%s | probability=%.4f | "
        "model=%s v%s | latency_ms=%.1f",
        order.order_id,
        result["is_late"],
        result["probability"],
        result["model_name"],
        result["model_version"],
        latency_ms,
    )
    return PredictionResponse(order_id=order.order_id, **result)
