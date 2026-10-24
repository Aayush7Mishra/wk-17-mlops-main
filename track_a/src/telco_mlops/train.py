"""Train candidates; select by validation F1; test and register the selected pipeline."""

import json
import time
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from telco_mlops.config import MODEL_NAME, REPORTS, SEED, configure_tracking
from telco_mlops.data import CATEGORICAL, FEATURES, NUMERIC, load_data, main as data_main, partition


def candidates():
    return {
        "logistic_c0.1": LogisticRegression(C=0.1, max_iter=1500, random_state=SEED),
        "logistic_balanced_c1": LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1500, random_state=SEED
        ),
        "forest_depth8": RandomForestClassifier(
            n_estimators=180, max_depth=8, min_samples_leaf=5, class_weight="balanced",
            n_jobs=2, random_state=SEED,
        ),
    }


def pipeline(classifier) -> Pipeline:
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
        ]), NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    return Pipeline([("preprocessing", preprocessing), ("classifier", classifier)])


def classification_metrics(y, probability) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
    }


def select_winner(rows: list[dict]) -> dict:
    # Deliberately independent of all test-set metrics.
    return max(rows, key=lambda row: (row["validation_f1"], row["validation_roc_auc"]))


def save_plots(y, probability, name: str, split: str) -> None:
    path = REPORTS / "figures"
    path.mkdir(exist_ok=True)
    for kind, display in [
        ("confusion", lambda: ConfusionMatrixDisplay.from_predictions(
            y, probability >= 0.5, display_labels=["No churn", "Churn"], cmap="Blues"
        )),
        ("roc", lambda: RocCurveDisplay.from_predictions(y, probability, name=name)),
    ]:
        display()
        plt.title(f"{name} — {split} {kind}")
        plt.tight_layout()
        file = path / f"{name}_{split}_{kind}.png"
        plt.savefig(file, dpi=120)
        plt.close()
        mlflow.log_artifact(str(file), artifact_path="figures")


def main() -> None:
    configure_tracking()
    REPORTS.mkdir(exist_ok=True)
    data_main()
    frame = load_data()
    train, validation, test = partition(frame)
    split_manifest = {
        "seed": SEED,
        "method": "stratified random 60/20/20; validation F1 then validation AUC tie-break",
        "threshold": 0.5,
        **{name: {"rows": len(part), "churn_rate": float(part.Churn.mean()),
                  "customer_ids": sorted(part.customerID.tolist())}
           for name, part in [("train", train), ("validation", validation), ("test", test)]},
    }
    (REPORTS / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2) + "\n")
    rows, models = [], {}
    for name, classifier in candidates().items():
        with mlflow.start_run(run_name=name) as run:
            model = pipeline(classifier)
            params = classifier.get_params()
            mlflow.log_params(params)
            mlflow.log_params({"threshold": 0.5, "split_seed": SEED, "train_rows": len(train)})
            mlflow.set_tags({"track": "A", "role": "candidate", "selection_metric": "validation_f1"})
            started = time.perf_counter()
            model.fit(train[FEATURES], train.Churn)
            fit_seconds = time.perf_counter() - started
            probability = model.predict_proba(validation[FEATURES])[:, 1]
            metrics = {f"validation_{k}": v for k, v in
                       classification_metrics(validation.Churn, probability).items()}
            mlflow.log_metrics({**metrics, "fit_seconds": fit_seconds})
            save_plots(validation.Churn, probability, name, "validation")
            mlflow.sklearn.log_model(
                model, artifact_path="model",
                signature=infer_signature(train[FEATURES], model.predict(train[FEATURES])),
                input_example=train[FEATURES].head(3),
                pip_requirements=["scikit-learn==1.6.1", "pandas==2.2.3", "numpy==2.2.6"],
            )
            mlflow.log_artifact(str(REPORTS / "data_provenance.json"))
            mlflow.log_artifact(str(REPORTS / "split_manifest.json"))
            row = {"model": name, "run_id": run.info.run_id, **metrics, "fit_seconds": fit_seconds}
            rows.append(row)
            models[name] = model
            (REPORTS / f"{name}_run.json").write_text(json.dumps(
                {**row, "parameters": params, "artifact_uri": f"runs:/{run.info.run_id}/model"},
                indent=2,
            ) + "\n")

    selected = select_winner(rows)
    best = models[selected["model"]]
    test_metrics = classification_metrics(test.Churn, best.predict_proba(test[FEATURES])[:, 1])
    with mlflow.start_run(run_id=selected["run_id"]):
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.set_tag("selected", "true")
        save_plots(test.Churn, best.predict_proba(test[FEATURES])[:, 1], selected["model"], "test")
    # The test set is evaluated once, only for the already-selected pipeline.
    comparison = pd.DataFrame(rows).sort_values("validation_f1", ascending=False)
    comparison.to_csv(REPORTS / "run_comparison.csv", index=False)
    (REPORTS / "test_metrics.json").write_text(json.dumps({
        "model": selected["model"], "rows": len(test), "metrics": test_metrics,
        "note": "Untouched holdout; not used to choose hyperparameters, model or threshold",
    }, indent=2) + "\n")

    registered = mlflow.register_model(f"runs:/{selected['run_id']}/model", MODEL_NAME)
    client = MlflowClient()
    transitions = []
    for stage in ["Staging", "Production"]:
        client.transition_model_version_stage(
            MODEL_NAME, registered.version, stage, archive_existing_versions=True
        )
        actual = client.get_model_version(MODEL_NAME, registered.version)
        assert actual.current_stage == stage
        transitions.append({"stage": actual.current_stage,
                            "observed_at": datetime.now(timezone.utc).isoformat()})
    client.set_registered_model_alias(MODEL_NAME, "champion", registered.version)
    client.set_model_version_tag(MODEL_NAME, registered.version, "selection_metric", "validation_f1")
    # Loading via the registry is part of the experiment, not merely a configured URI.
    loaded = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
    sample = test[FEATURES].head(2)
    assert (loaded.predict(sample) == best.predict(sample)).all()
    evidence = {
        "model_name": MODEL_NAME, "version": registered.version, "run_id": selected["run_id"],
        "model": selected["model"], "model_uri": f"models:/{MODEL_NAME}@champion",
        "selection_rule": "highest validation F1; validation ROC-AUC breaks ties",
        "validation_f1": selected["validation_f1"], "transitions": transitions,
        "final_stage": client.get_model_version(MODEL_NAME, registered.version).current_stage,
        "registry_load_verified": True, "test_metrics": test_metrics,
    }
    (REPORTS / "registry_transitions.json").write_text(json.dumps(evidence, indent=2) + "\n")
    # Native types and null for missing numeric values make this a valid curl request.
    records = json.loads(sample.to_json(orient="records"))
    (REPORTS / "sample_request.json").write_text(json.dumps({"customers": records}, indent=2) + "\n")
    with mlflow.start_run(run_id=selected["run_id"]):
        for filename in ["registry_transitions.json", "run_comparison.csv", "test_metrics.json"]:
            mlflow.log_artifact(str(REPORTS / filename), artifact_path="export")
    print(comparison.to_string(index=False))
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
