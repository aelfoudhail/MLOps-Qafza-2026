from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class OrderRequest(BaseModel):
    """Raw fields notebook 5's engineer_dates/bucket functions need.
    Engineered fields (estimated_delivery_days, purchase_month,
    high_risk_month, customer_state_grouped, etc.) are NOT requested here,
    they get computed by src/features/engineer.py from these raw values.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "order_id": "example-order-1",
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
                "avg_distance_km": 500.0,
                "total_product_weight_g": 1200.0,
                "avg_product_weight_g": 600.0,
                "avg_product_length_cm": 20.0,
                "avg_product_height_cm": 10.0,
                "avg_product_width_cm": 15.0,
                "customer_state": "SP",
                "main_seller_state": "RJ",
            }
        }
    )

    order_id: Optional[str] = None

    order_purchase_timestamp: datetime
    order_estimated_delivery_date: datetime

    num_items: int = Field(ge=1, le=50)
    num_distinct_products: int = Field(ge=1, le=50)
    num_distinct_sellers: int = Field(ge=1, le=50)
    total_price: float = Field(gt=0)
    total_freight_value: float = Field(ge=0)
    avg_item_price: float = Field(gt=0)

    num_payments: int = Field(ge=1, le=20)
    total_payment_value: float = Field(gt=0)
    max_installments: int = Field(ge=1, le=24)
    main_payment_type: str

    avg_distance_km: Optional[float] = Field(default=None, ge=0)
    total_product_weight_g: Optional[float] = Field(default=None, ge=0)
    avg_product_weight_g: Optional[float] = Field(default=None, ge=0)
    avg_product_length_cm: Optional[float] = Field(default=None, ge=0)
    avg_product_height_cm: Optional[float] = Field(default=None, ge=0)
    avg_product_width_cm: Optional[float] = Field(default=None, ge=0)

    customer_state: str = Field(min_length=2, max_length=2)
    main_seller_state: str = Field(min_length=2, max_length=2)

    @field_validator("order_estimated_delivery_date")
    @classmethod
    def estimated_after_purchase(cls, v, info):
        purchase = info.data.get("order_purchase_timestamp")
        if purchase is not None and v <= purchase:
            raise ValueError("order_estimated_delivery_date must be after order_purchase_timestamp")
        return v

    @field_validator("customer_state", "main_seller_state")
    @classmethod
    def upper_case_state(cls, v: str) -> str:
        return v.upper()


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    order_id: Optional[str] = None
    is_late: bool
    probability: float = Field(ge=0, le=1)
    threshold_used: float
    model_name: str
    model_version: str


class BatchPredictionRequest(BaseModel):
    orders: list[OrderRequest]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_name: str
    model_version: str