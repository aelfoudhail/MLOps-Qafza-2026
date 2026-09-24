import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.models.predictor import ArtifactLoadError, Predictor

NUMERIC_FEATURES = ["num_items", "total_price"]
CATEGORICAL_FEATURES = ["customer_state_grouped"]


def _fake_predictor():
    predictor = object.__new__(Predictor)
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    fake_train = pd.DataFrame(
        {
            "num_items": [1, 2, 3, 1, 5],
            "total_price": [50.0, 120.0, 30.0, 80.0, 200.0],
            "customer_state_grouped": ["SP", "RJ", "SP", "other", "RJ"],
        }
    )
    X = preprocessor.fit_transform(fake_train)
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, [0, 1, 0, 0, 1])

    predictor.preprocessor = preprocessor
    predictor.model = model
    predictor.threshold = 0.5
    predictor.scaler = None
    predictor.model_name = "RandomForestClassifier"
    predictor.model_version = "1"
    predictor.config = {
        "features": {
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
        }
    }
    return predictor


def test_predict_one_returns_expected_keys_and_shape():
    predictor = _fake_predictor()
    features_df = pd.DataFrame(
        [{"num_items": 2, "total_price": 100.0, "customer_state_grouped": "SP"}]
    )
    result = predictor.predict_one(features_df)
    assert set(result.keys()) == {
        "is_late",
        "probability",
        "threshold_used",
        "model_name",
        "model_version",
    }
    assert 0.0 <= result["probability"] <= 1.0
    assert isinstance(result["is_late"], bool)


def test_predict_one_threshold_logic_is_consistent():
    predictor = _fake_predictor()
    features_df = pd.DataFrame(
        [{"num_items": 2, "total_price": 100.0, "customer_state_grouped": "SP"}]
    )
    result = predictor.predict_one(features_df)
    assert result["is_late"] == (result["probability"] >= predictor.threshold)


def test_predict_one_rejects_missing_column():
    predictor = _fake_predictor()
    with pytest.raises(ValueError, match="missing expected columns"):
        predictor.predict_one(pd.DataFrame([{"num_items": 2}]))


def test_predictor_raises_when_mlflow_registry_unreachable(monkeypatch):
    import mlflow.sklearn
    from mlflow.exceptions import MlflowException

    def fake_load_model(uri):
        raise MlflowException("model not found")

    monkeypatch.setattr(mlflow.sklearn, "load_model", fake_load_model)

    with pytest.raises(ArtifactLoadError):
        Predictor()
