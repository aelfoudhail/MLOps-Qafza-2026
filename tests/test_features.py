import pandas as pd
import pytest

from src.features.engineer import (
    add_high_risk_month_flag,
    bucket_rare_categories,
    build_features,
    engineer_dates,
)


def _sample_order(purchase="2018-02-10 14:30:00", estimated="2018-02-25 00:00:00"):
    return {
        "order_purchase_timestamp": purchase,
        "order_estimated_delivery_date": estimated,
        "num_items": 2,
        "customer_state": "SP",
        "main_seller_state": "RJ",
    }


def test_engineer_dates_computes_expected_day_count():
    df = pd.DataFrame([_sample_order()])
    result = engineer_dates(df)
    assert result.loc[0, "estimated_delivery_days"] == pytest.approx(14.396, abs=0.01)
    assert result.loc[0, "purchase_month"] == 2
    assert result.loc[0, "purchase_weekday"] == 5  # Saturday
    assert "order_purchase_timestamp" not in result.columns
    assert "order_estimated_delivery_date" not in result.columns


@pytest.mark.parametrize("month,expected", [(11, 1), (2, 1), (3, 1), (7, 0), (1, 0)])
def test_high_risk_month_flag(month, expected):
    df = pd.DataFrame([{"purchase_month": month}])
    result = add_high_risk_month_flag(df, high_risk_months=[11, 2, 3])
    assert result.loc[0, "high_risk_month"] == expected


def test_bucket_rare_categories_keeps_allowed_value():
    df = pd.DataFrame([{"customer_state": "SP"}])
    result = bucket_rare_categories(df, "customer_state", ["SP", "RJ"], "customer_state_grouped")
    assert result.loc[0, "customer_state_grouped"] == "SP"
    assert "customer_state" not in result.columns


def test_bucket_rare_categories_buckets_unseen_value_as_other():
    df = pd.DataFrame([{"customer_state": "AC"}])
    result = bucket_rare_categories(df, "customer_state", ["SP", "RJ"], "customer_state_grouped")
    assert result.loc[0, "customer_state_grouped"] == "other"


def test_build_features_end_to_end():
    result = build_features(
        raw_order=_sample_order(),
        top_states=["SP"],
        top_seller_states=["RJ"],
        high_risk_months=[11, 2, 3],
    )
    assert result.loc[0, "customer_state_grouped"] == "SP"
    assert result.loc[0, "main_seller_state_grouped"] == "RJ"
    assert result.loc[0, "high_risk_month"] == 1
