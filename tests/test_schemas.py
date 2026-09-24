import pytest
from pydantic import ValidationError

from src.validation.schemas import OrderRequest

VALID_ORDER = {
    "order_purchase_timestamp": "2018-02-10T14:30:00",
    "order_estimated_delivery_date": "2018-02-25T00:00:00",
    "num_items": 2,
    "num_distinct_products": 2,
    "num_distinct_sellers": 1,
    "total_price": 150.0,
    "total_freight_value": 20.0,
    "avg_item_price": 75.0,
    "num_payments": 1,
    "total_payment_value": 170.0,
    "max_installments": 3,
    "main_payment_type": "credit_card",
    "customer_state": "SP",
    "main_seller_state": "RJ",
}


def test_valid_order_is_accepted():
    order = OrderRequest(**VALID_ORDER)
    assert order.customer_state == "SP"


def test_negative_price_is_rejected():
    bad = {**VALID_ORDER, "total_price": -50.0}
    with pytest.raises(ValidationError):
        OrderRequest(**bad)


def test_estimated_before_purchase_is_rejected():
    bad = {
        **VALID_ORDER,
        "order_purchase_timestamp": "2018-02-25T00:00:00",
        "order_estimated_delivery_date": "2018-02-10T14:30:00",
    }
    with pytest.raises(ValidationError):
        OrderRequest(**bad)


def test_lowercase_state_gets_uppercased():
    order = OrderRequest(**{**VALID_ORDER, "customer_state": "sp"})
    assert order.customer_state == "SP"


def test_no_post_delivery_fields_exist_on_the_schema():
    """Data leakage check: fields that only exist after delivery must never
    be requestable, since a real prediction happens before delivery."""
    leaky_fields = {
        "order_delivered_customer_date",
        "order_delivered_carrier_date",
        "avg_review_score",
        "num_reviews",
        "delivery_gap_days",
        "order_status",
    }
    assert leaky_fields.isdisjoint(OrderRequest.model_fields.keys())
