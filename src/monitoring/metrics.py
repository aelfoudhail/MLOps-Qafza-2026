"""
Simple in-process metrics counters. Not persisted across restarts, and not
shared across multiple worker processes, good enough for this project's
scope (a single-process service). A production setup at real scale would
export these to Prometheus/Datadog instead of keeping them in memory.
"""

import threading

_lock = threading.Lock()
_state = {
    "request_count": 0,
    "error_count": 0,
    "total_latency_ms": 0.0,
    "predictions_late": 0,
    "predictions_on_time": 0,
}


def record_request(status_code: int, latency_ms: float) -> None:
    with _lock:
        _state["request_count"] += 1
        _state["total_latency_ms"] += latency_ms
        if status_code >= 400:
            _state["error_count"] += 1


def record_prediction(is_late: bool) -> None:
    with _lock:
        if is_late:
            _state["predictions_late"] += 1
        else:
            _state["predictions_on_time"] += 1


def snapshot() -> dict:
    with _lock:
        request_count = _state["request_count"]
        error_count = _state["error_count"]
        avg_latency = _state["total_latency_ms"] / request_count if request_count else 0.0
        total_predictions = _state["predictions_late"] + _state["predictions_on_time"]
        late_rate = _state["predictions_late"] / total_predictions if total_predictions else None
        return {
            "request_count": request_count,
            "error_count": error_count,
            "error_rate": round(error_count / request_count, 4) if request_count else 0.0,
            "avg_latency_ms": round(avg_latency, 2),
            "predictions_served": total_predictions,
            "late_rate_this_process": round(late_rate, 4) if late_rate is not None else None,
        }
