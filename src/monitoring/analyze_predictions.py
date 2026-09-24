"""
Reads the stored prediction log and checks for drift against the baseline
late rate from task 2's test set. Run manually or on a schedule:
python -m src.monitoring.analyze_predictions
"""

import json

import pandas as pd

from src.config import PROJECT_ROOT, load_config

PREDICTIONS_LOG_PATH = PROJECT_ROOT / "logs" / "predictions.jsonl"


def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_LOG_PATH.exists():
        raise FileNotFoundError(
            f"No predictions logged yet at {PREDICTIONS_LOG_PATH}, serve a few requests first."
        )
    records = [json.loads(line) for line in open(PREDICTIONS_LOG_PATH) if line.strip()]
    return pd.DataFrame(records)


def analyze():
    df = load_predictions()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    total = len(df)
    late_rate = df["is_late"].mean()

    config = load_config()
    baseline = config["monitoring"]["baseline_late_rate"]
    drift_threshold = config["monitoring"]["drift_threshold_points"]
    drift = abs(late_rate - baseline)

    print(f"predictions analyzed: {total}")
    print(f"observed late rate:   {late_rate:.4f}")
    print(f"baseline late rate:   {baseline:.4f}  (task 2 test set)")
    print(f"drift:                {drift:.4f}")

    if drift > drift_threshold:
        print(f"WARNING: drifted more than {drift_threshold:.2f} points from baseline")
    else:
        print("within expected range, no drift alert")


if __name__ == "__main__":
    analyze()
