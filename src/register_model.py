"""
Logs the already-trained model (from notebook 6) as a real MLflow run, then
registers it and marks it with a "champion" alias, the current replacement
for the deprecated Staging/Production "stage" concept.

Run once after training: python -m src.register_model
"""

import logging

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.config import PROJECT_ROOT, get_artifact_path

logger = logging.getLogger(__name__)

MODEL_NAME = "late_delivery_classifier"

mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT}/mlflow.db")
mlflow.set_experiment("late_delivery_prediction")


def register_current_model():
    model = joblib.load(get_artifact_path("model_file"))
    threshold = joblib.load(get_artifact_path("threshold_file"))

    with mlflow.start_run() as run:
        mlflow.log_params(model.get_params())
        mlflow.log_metric("decision_threshold", threshold)

        model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                    registered_model_name=MODEL_NAME,
                    skops_trusted_types=["sklearn.tree._tree.Tree"],
                )

        logger.info("logged run %s, registered as %s v%s",
                    run.info.run_id, MODEL_NAME, model_info.registered_model_version)

    client = MlflowClient()
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=model_info.registered_model_version,
    )
    logger.info("set alias 'champion' -> version %s", model_info.registered_model_version)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    register_current_model()