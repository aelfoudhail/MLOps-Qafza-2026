"""
Loads the fitted artifacts notebooks 5/6 produced. Fails loudly at startup
if anything's missing, rather than serving predictions from a half-loaded
model.
"""
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

import logging

import joblib
import pandas as pd

from src.config import PROJECT_ROOT, get_artifact_path, load_config

logger = logging.getLogger(__name__)


class ArtifactLoadError(RuntimeError):
    """Raised when a required model artifact can't be loaded."""


class Predictor:
    def __init__(self):
        self.config = load_config()
        self._load_artifacts()

    def _load_artifacts(self):
        mlflow_cfg = self.config["mlflow"]
        mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / mlflow_cfg['db_file']}")

        model_uri = f"models:/{mlflow_cfg['model_name']}@{mlflow_cfg['model_alias']}"
        try:
            self.model = mlflow.sklearn.load_model(model_uri)
            client = MlflowClient()
            model_version_info = client.get_model_version_by_alias(
                mlflow_cfg["model_name"], mlflow_cfg["model_alias"]
            )
        except MlflowException as exc:
            raise ArtifactLoadError(
                f"Could not load model from MLflow registry at {model_uri}: {exc}. "
                f"Did you run `python -m src.register_model` first?"
            ) from exc

        try:
            self.preprocessor = joblib.load(get_artifact_path("preprocessor_file"))
            self.threshold = joblib.load(get_artifact_path("threshold_file"))
            self.top_states = joblib.load(get_artifact_path("top_states_file"))
            self.top_seller_states = joblib.load(get_artifact_path("top_seller_states_file"))
            with open(get_artifact_path("feature_list_file"), "r") as f:
                self.feature_list = [line.strip() for line in f if line.strip()]
        except FileNotFoundError as exc:
            raise ArtifactLoadError(f"Missing a required model artifact: {exc}") from exc

        self.model_name = type(self.model).__name__  # e.g. "RandomForestClassifier"
        self.model_version = str(model_version_info.version)  # e.g. "1", the real registry version

        self.scaler = None
        if self.model_name == "LogisticRegression":
            scaler_path = get_artifact_path("logreg_scaler_file")
            if not scaler_path.exists():
                raise ArtifactLoadError("Model is LogisticRegression but logreg_scaler.pkl is missing")
            self.scaler = joblib.load(scaler_path)

        logger.info(
            "Loaded model=%s (registry version %s) threshold=%.3f",
            self.model_name, self.model_version, self.threshold,
        )

    def predict_one(self, features_df: pd.DataFrame) -> dict:
        numeric_features = self.config["features"]["numeric_features"]
        categorical_features = self.config["features"]["categorical_features"]

        missing_cols = [c for c in numeric_features + categorical_features if c not in features_df.columns]
        if missing_cols:
            raise ValueError(f"features_df is missing expected columns: {missing_cols}")

        X = self.preprocessor.transform(features_df[numeric_features + categorical_features])
        if self.scaler is not None:
            X = self.scaler.transform(X)

        probability = float(self.model.predict_proba(X)[0, 1])
        is_late = probability >= self.threshold

        return {
            "is_late": bool(is_late),
            "probability": probability,
            "threshold_used": float(self.threshold),
            "model_name": self.model_name,
            "model_version": self.model_version,
        }


_predictor_instance = None


def get_predictor() -> Predictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = Predictor()
    return _predictor_instance