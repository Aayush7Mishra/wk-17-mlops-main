"""Local defaults; use TELCO_STATE_DIR/MLFLOW_TRACKING_URI for another deployment."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE = Path(os.environ.get("TELCO_STATE_DIR", ROOT / ".state")).resolve()
REPORTS = ROOT / "reports"
DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
MODEL_NAME = "telco-churn"
SEED = 42


def tracking_uri() -> str:
    STATE.mkdir(parents=True, exist_ok=True)
    return os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{STATE / 'mlflow.db'}")


def configure_tracking() -> None:
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    name = "telco-churn-track-a"
    if client.get_experiment_by_name(name) is None:
        client.create_experiment(name, artifact_location=(STATE / "artifacts").as_uri())
    mlflow.set_experiment(name)
