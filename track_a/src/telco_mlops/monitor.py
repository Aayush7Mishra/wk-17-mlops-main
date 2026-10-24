"""Real Evidently data/target drift with a custom metric and explicit response policy."""

import json

import mlflow
import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.core.metric_types import SingleValueCalculation, SingleValueMetric
from evidently.metrics import ValueDrift
from evidently.presets import DataDriftPreset
from evidently.tests import lt
from sklearn.model_selection import train_test_split

from telco_mlops.config import REPORTS, SEED, configure_tracking
from telco_mlops.data import CATEGORICAL, FEATURES, NUMERIC, load_data


class AbsoluteMeanShift(SingleValueMetric):
    """Business-unit shift, unlike the built-in normalized drift distances."""

    column: str


class AbsoluteMeanShiftCalculation(SingleValueCalculation[AbsoluteMeanShift]):
    def calculate(self, context, current_data, reference_data):
        if reference_data is None:
            raise ValueError("AbsoluteMeanShift requires reference data")
        current_mean = current_data.column(self.metric.column).data.mean()
        reference_mean = reference_data.column(self.metric.column).data.mean()
        return self.result(float(abs(current_mean - reference_mean)))

    def display_name(self) -> str:
        return f"Absolute mean shift in {self.metric.column} (billing units)"


def inject_drift(current: pd.DataFrame) -> pd.DataFrame:
    changed = current.copy(deep=True)
    rng = np.random.default_rng(SEED)
    changed["MonthlyCharges"] += rng.normal(loc=30, scale=5, size=len(changed))
    changed.loc[rng.random(len(changed)) < 0.85, "Contract"] = "Month-to-month"
    # One-way flips intentionally increase churn prevalence; these are not observed outcomes.
    changed.loc[(changed.Churn == 0) & (rng.random(len(changed)) < 0.25), "Churn"] = 1
    return changed


def evaluate_drift(reference: pd.DataFrame, current: pd.DataFrame):
    definition = DataDefinition(
        numerical_columns=NUMERIC, categorical_columns=CATEGORICAL + ["Churn"]
    )
    target_metric = ValueDrift(column="Churn", method="jensenshannon", threshold=0.1)
    custom_metric = AbsoluteMeanShift(column="MonthlyCharges", tests=[lt(10)])
    snapshot = Report([
        DataDriftPreset(
            columns=FEATURES, num_method="wasserstein", cat_method="jensenshannon",
            num_threshold=0.1, cat_threshold=0.1, drift_share=0.2,
        ),
        target_metric,
        custom_metric,
    ], include_tests=True).run(
        current_data=Dataset.from_pandas(current[FEATURES + ["Churn"]], data_definition=definition),
        reference_data=Dataset.from_pandas(reference[FEATURES + ["Churn"]], data_definition=definition),
    )
    columns = {}
    for column in FEATURES + ["Churn"]:
        method = "wasserstein" if column in NUMERIC else "jensenshannon"
        metric = ValueDrift(column=column, method=method, threshold=0.1)
        result = snapshot.context.get_metric_result(metric)
        columns[column] = {
            "score": float(result.value), "method": method, "threshold": 0.1,
            "drift_detected": any(str(test.status).lower().endswith("fail") for test in result.tests),
        }
    custom = float(snapshot.context.get_metric_result(custom_metric).value)
    summary = {
        "reference_rows": len(reference), "current_rows": len(current),
        "columns": columns,
        "drifted_features": [column for column in FEATURES if columns[column]["drift_detected"]],
        "target_drift": columns["Churn"]["drift_detected"],
        "reference_churn_rate": float(reference.Churn.mean()),
        "current_churn_rate": float(current.Churn.mean()),
        "reference_monthly_charges_mean": float(reference.MonthlyCharges.mean()),
        "current_monthly_charges_mean": float(current.MonthlyCharges.mean()),
        "custom_absolute_mean_charge_shift": custom,
        "custom_test_passed": custom < 10,
        "reference_month_to_month_share": float((reference.Contract == "Month-to-month").mean()),
        "current_month_to_month_share": float((current.Contract == "Month-to-month").mean()),
    }
    summary["investigate_and_retrain_recommended"] = (
        len(summary["drifted_features"]) / len(FEATURES) >= 0.2
        or summary["target_drift"] or not summary["custom_test_passed"]
    )
    return snapshot, summary


def main() -> None:
    configure_tracking()
    frame = load_data()
    reference, incoming = train_test_split(
        frame, test_size=0.3, stratify=frame.Churn, random_state=SEED
    )
    # A no-injection control distinguishes sampling noise from engineered drift.
    baseline, baseline_summary = evaluate_drift(reference, incoming)
    current = inject_drift(incoming)
    snapshot, summary = evaluate_drift(reference, current)
    for column in ["MonthlyCharges", "Contract", "Churn"]:
        if not summary["columns"][column]["drift_detected"]:
            raise RuntimeError(f"Expected injected drift was not detected in {column}")
    REPORTS.mkdir(exist_ok=True)
    baseline.save_html(str(REPORTS / "drift_control.html"))
    snapshot.save_html(str(REPORTS / "drift_report.html"))
    snapshot.save_json(str(REPORTS / "drift_report.json"))
    with mlflow.start_run(run_name="drift-monitor") as run:
        summary["run_id"] = run.info.run_id
        summary["seed"] = SEED
        summary["perturbations"] = {
            "MonthlyCharges": "add Normal(mean=30, sd=5) to every incoming row",
            "Contract": "set 85% Bernoulli-selected rows to Month-to-month",
            "Churn": "flip Bernoulli(0.25)-selected non-churn labels to churn",
        }
        summary["control"] = baseline_summary
        summary["action"] = (
            "Investigate billing/input changes and label quality; collect real delayed outcomes, "
            "evaluate the deployed model on recent labeled data, and retrain a candidate only "
            "after validation. Drift alone does not prove performance degradation; do not "
            "automatically promote a model trained on synthetic labels."
        )
        (REPORTS / "drift_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        mlflow.log_params({"seed": SEED, "reference_fraction": 0.7,
                           "charge_shift_mean": 30, "charge_shift_sd": 5,
                           "month_to_month_injection_probability": 0.85,
                           "negative_label_flip_probability": 0.25})
        mlflow.log_metrics({
            "drifted_feature_count": len(summary["drifted_features"]),
            "target_drift_detected": int(summary["target_drift"]),
            "absolute_mean_charge_shift": summary["custom_absolute_mean_charge_shift"],
            "current_churn_rate": summary["current_churn_rate"],
            "retraining_review_recommended": int(summary["investigate_and_retrain_recommended"]),
        })
        for filename in ["drift_report.html", "drift_control.html", "drift_report.json", "drift_summary.json"]:
            mlflow.log_artifact(str(REPORTS / filename), artifact_path="monitoring")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
