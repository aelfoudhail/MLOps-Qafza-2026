"""
Same feature engineering logic as notebook 5, as plain functions instead of
notebook cells. Verified line-by-line against the actual engineer_dates,
bucket_states, and bucket_seller_states functions in the real notebook.
"""

import pandas as pd


def engineer_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["order_estimated_delivery_date"] = pd.to_datetime(df["order_estimated_delivery_date"])

    df["estimated_delivery_days"] = (
        df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_weekday"] = df["order_purchase_timestamp"].dt.dayofweek

    return df.drop(columns=["order_purchase_timestamp", "order_estimated_delivery_date"])


def add_high_risk_month_flag(df: pd.DataFrame, high_risk_months: list) -> pd.DataFrame:
    df = df.copy()
    df["high_risk_month"] = df["purchase_month"].isin(high_risk_months).astype(int)
    return df


def bucket_rare_categories(
    df: pd.DataFrame,
    column: str,
    allowed_values: list,
    new_column: str,
    other_label: str = "other",
) -> pd.DataFrame:
    df = df.copy()
    df[new_column] = df[column].where(df[column].isin(allowed_values), other_label)
    return df.drop(columns=[column])


def build_features(
    raw_order: dict,
    top_states: list,
    top_seller_states: list,
    high_risk_months: list,
) -> pd.DataFrame:
    df = pd.DataFrame([raw_order])
    df = engineer_dates(df)
    df = add_high_risk_month_flag(df, high_risk_months)
    df = bucket_rare_categories(df, "customer_state", top_states, "customer_state_grouped")
    df = bucket_rare_categories(df, "main_seller_state", top_seller_states, "main_seller_state_grouped")
    return df