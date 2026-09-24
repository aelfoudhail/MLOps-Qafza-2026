from fastapi.testclient import TestClient

from app.main import app

VALID_ORDER = {
    "order_id": "integration-test-1",
    "order_purchase_timestamp": "2018-02-10T14:30:00",
    "order_estimated_delivery_date": "2018-02-25T00:00:00",
    "num_items": 2, "num_distinct_products": 2, "num_distinct_sellers": 1,
    "total_price": 150.0, "total_freight_value": 20.0, "avg_item_price": 75.0,
    "num_payments": 1, "total_payment_value": 170.0, "max_installments": 3,
    "main_payment_type": "credit_card", "customer_state": "SP", "main_seller_state": "RJ",
}


def test_health_route_reports_model_loaded():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is True


def test_predict_route_returns_valid_prediction():
    with TestClient(app) as client:
        response = client.post("/predict", json=VALID_ORDER)
        assert response.status_code == 200
        body = response.json()
        assert body["order_id"] == "integration-test-1"
        assert isinstance(body["is_late"], bool)
        assert 0.0 <= body["probability"] <= 1.0


def test_predict_route_rejects_bad_payload_cleanly():
    with TestClient(app) as client:
        response = client.post("/predict", json={**VALID_ORDER, "total_price": -1})
        assert response.status_code == 422


def test_batch_predict_route_works():
    with TestClient(app) as client:
        response = client.post("/predict/batch", json={"orders": [VALID_ORDER, VALID_ORDER]})
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 2