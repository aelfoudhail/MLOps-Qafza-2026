"""
Shared test fixtures. The key one, ci_artifacts, builds a throwaway model,
preprocessor, and MLflow registry entry in a temp directory, then points
the app at those instead of your real production artifacts. This means
tests never depend on your real trained model existing on the machine
running them, exactly what's needed for a clean CI run.
"""

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import src.models.predictor as predictor_module
from src.config import load_config


@pytest.fixture(scope="session")
def ci_artifacts(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("ci_artifacts")
    real_config = load_config()
    numeric_features = real_config["features"]["numeric_features"]
    categorical_features = real_config["features"]["categorical_features"]

    # build a tiny fake preprocessor + model over the real feature names
    fake_rows = 30
    fake_df = pd.DataFrame({f: range(fake_rows) for f in numeric_features})
    for c in categorical_features:
        fake_df[c] = (["SP", "RJ", "other"] * fake_rows)[:fake_rows]
    fake_labels = [i % 2 for i in range(fake_rows)]

    preprocessor = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    X = preprocessor.fit_transform(fake_df)
    model = RandomForestClassifier(n_estimators=5, random_state=42).fit(X, fake_labels)

    artifacts_dir = tmp_dir / "artifacts"
    artifacts_dir.mkdir()
    joblib.dump(preprocessor, artifacts_dir / "preprocessor.pkl")
    joblib.dump(0.5, artifacts_dir / "decision_threshold.pkl")
    joblib.dump(["SP", "RJ"], artifacts_dir / "top_states.pkl")
    joblib.dump(["SP", "RJ"], artifacts_dir / "top_seller_states.pkl")
    with open(artifacts_dir / "feature_list.txt", "w") as f:
        f.write("\n".join(numeric_features + categorical_features))

    mlflow_db = tmp_dir / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    mlflow.set_experiment("ci_test_experiment")
    model_name = real_config["mlflow"]["model_name"]
    alias = real_config["mlflow"]["model_alias"]

    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=model_name,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
    mlflow.tracking.MlflowClient().set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=model_info.registered_model_version,
    )

    fake_config = {**real_config}
    fake_config["artifacts"] = {**real_config["artifacts"], "dir": str(artifacts_dir)}
    fake_config["mlflow"] = {**real_config["mlflow"], "db_file": str(mlflow_db)}

    return fake_config


@pytest.fixture(autouse=True)
def reset_predictor_singleton():
    """The Predictor is a module-level singleton, reset it between test
    modules so one test's fake model doesn't leak into another's."""
    predictor_module._predictor_instance = None
    yield
    predictor_module._predictor_instance = None


@pytest.fixture
def patched_config(ci_artifacts, monkeypatch):
    monkeypatch.setattr("src.config.load_config", lambda: ci_artifacts)
    monkeypatch.setattr(predictor_module, "load_config", lambda: ci_artifacts)
    yield ci_artifacts
