"""Verify real MLflow artifacts and a live HTTP server after training/monitoring."""

import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from telco_mlops.config import MODEL_NAME, REPORTS, ROOT, STATE, tracking_uri
from telco_mlops.data import FEATURES


def request(url, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as response:
        return response.status, json.load(response)


def main():
    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    rows = pd.read_csv(REPORTS / "run_comparison.csv").to_dict(orient="records")
    evidence = {"candidate_runs": [], "monitoring_artifacts_verified": False}
    for row in rows:
        run = client.get_run(row["run_id"])
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            assert run.data.metrics[f"validation_{metric}"] == row[f"validation_{metric}"]
        model_files = [item.path for item in client.list_artifacts(row["run_id"], "model")]
        figures = [item.path for item in client.list_artifacts(row["run_id"], "figures")]
        assert "model/model.pkl" in model_files and "model/MLmodel" in model_files
        assert any("confusion" in path for path in figures) and any("roc" in path for path in figures)
        evidence["candidate_runs"].append({"run_id": row["run_id"], "status": run.info.status,
                                            "model_files": model_files, "figures": figures})
    monitoring = json.loads((REPORTS / "drift_summary.json").read_text())
    logged = {item.path for item in client.list_artifacts(monitoring["run_id"], "monitoring")}
    assert "monitoring/drift_report.html" in logged and "monitoring/drift_summary.json" in logged
    evidence["monitoring_artifacts_verified"] = True
    registered = client.get_model_version_by_alias(MODEL_NAME, "champion")
    assert registered.current_stage == "Production"

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with (STATE / "serving.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "telco_mlops.api:app", "--host", "127.0.0.1",
             "--port", str(port)], cwd=ROOT, stdout=log, stderr=log,
        )
        try:
            for attempt in range(100):
                if process.poll() is not None:
                    raise RuntimeError("API exited during startup; inspect .state/serving.log")
                try:
                    health_status, health = request(f"http://127.0.0.1:{port}/health")
                    break
                except (urllib.error.URLError, ConnectionError):
                    time.sleep(0.1)
            else:
                raise RuntimeError("API did not become ready")
            payload = json.loads((REPORTS / "sample_request.json").read_text())
            status, response = request(f"http://127.0.0.1:{port}/predict", payload)
            assert health_status == status == 200
            assert str(response["model_version"]) == str(registered.version)
            model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{registered.version}")
            expected = model.predict_proba(pd.DataFrame(payload["customers"])[FEATURES])[:, 1]
            np.testing.assert_allclose([p["probability"] for p in response["predictions"]], expected)
            try:
                request(f"http://127.0.0.1:{port}/predict", {"customers": []})
            except urllib.error.HTTPError as error:
                assert error.code == 422
            else:
                raise AssertionError("Empty batch should be rejected")
            evidence["http_api"] = {"health": health, "prediction_status": status,
                                     "response": response, "empty_batch_status": 422,
                                     "probabilities_match_registry_model": True}
        finally:
            process.terminate()
            process.wait(timeout=15)
    (REPORTS / "runtime_verification.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
