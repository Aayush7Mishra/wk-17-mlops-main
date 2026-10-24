# Track A — Telco churn MLOps

An executed end-to-end workflow for the IBM Telco sample: verified data → training → MLflow tracking → registry promotion → HTTP inference → Evidently monitoring → a retraining review recommendation. This project has its own environment and runs independently of Track B.

## Environment and reproduction

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.12 (uv can install it). From this directory:

```sh
uv sync
uv run telco-train
uv run telco-monitor
uv run pytest -q
uv run python scripts/verify_runtime.py
```

`uv sync` is the one-command environment setup, including development/test dependencies. For strict use of the committed resolution, use `uv sync --frozen` and `uv run --frozen ...`. The first training run downloads the public CSV; subsequent runs verify the cached bytes. No account, API key, paid service, Docker, or separate MLflow server is required. An internet connection is needed for the first dependency/data download.

A clean-directory reproduction was executed with only source, tests, scripts and the environment files copied: fresh virtual environment, fresh CSV download and fresh registry. Setup, training, monitoring, all nine tests and live HTTP verification passed. Holdout metrics matched exactly; drift scores matched to numerical precision and all drift decisions matched. [Clean reproduction evidence](reports/clean_reproduction.json) records the commands and lockfile checksum.

`pyproject.toml`, `.python-version`, and `uv.lock` fix Python minor version, direct packages and all transitive packages. This is material here: the initial unbounded SQLAlchemy 2.1 resolution broke MLflow 2.22's `FallbackAsyncAdaptedQueuePool` import. SQLAlchemy is pinned to 2.0.41. MLflow 2.22.1 preserves the assignment's stage-transition API, scikit-learn 1.6.1 stabilizes preprocessing/model serialization, and Evidently 0.7.17 fixes the custom metric/report interfaces. Upgrades require re-running the tests and regenerating the model and reports.

The local SQLite registry and model artifacts are under `.state/`, which is ignored by Git because its artifact paths are machine-specific. Re-running training rebuilds these on a new machine. The committed `reports/` directory contains portable evidence from an actual execution, including every candidate's parameters, metrics, plots, registry transitions, drift reports, and a live HTTP smoke test. Run IDs/version numbers and measured training durations change on a rerun; deterministic classification and drift values should remain stable with this lockfile. Each training rerun creates a new model version, archives the preceding Production version, and moves `champion` to the new version.

## Dataset and leakage controls

Data comes directly from [IBM's archived source repository](https://github.com/IBM/telco-customer-churn-on-icp4d), not generated rows: 7,043 fictional sample customers, 21 raw columns and 1,869 churn labels (26.537%). `data.py` checks SHA-256 `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91` before reading the downloaded file. [Provenance](reports/data_provenance.json) includes the exact URL, size and checksum. A changed or corrupt download fails closed.

The training split is stratified 60/20/20, seed 42: 4,225 training, 1,409 validation and 1,409 test customers. [Split membership](reports/split_manifest.json) records all sample IDs. IDs and the target are excluded from model features. Eleven blank `TotalCharges` cells become missing numeric values; median imputation, scaling and one-hot category learning are fitted **only on training data** inside the saved sklearn pipeline. Unseen categories use `handle_unknown="ignore"`. We select with validation F1 and use validation ROC-AUC only to break an exact tie. The 0.5 decision threshold is fixed in advance. Only the already-selected model is evaluated on the untouched test set. No test metrics drive model/threshold selection, and no refitting changes the evaluated winner before registration.

This is a cross-sectional, fictional sample dataset. A stratified holdout measures performance on that sample, not future deployment performance or the causal effect of retention offers.

## Experiment tracking and selection

Three candidates compare regularization/class weighting and model families. MLflow logs all estimator parameters, split seed, threshold, fit duration, five validation metrics, each full preprocessing/model pipeline, an input example/signature, a confusion matrix and a ROC curve. Churn (`1`) is the positive class for precision, recall and F1.

Actual validation results from [the exported comparison](reports/run_comparison.csv):

| Model | Deliberate variation | Accuracy | Precision | Recall | F1 | ROC-AUC | Fit seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| `logistic_c0.1` | C=0.1; no class weighting | 0.803407 | 0.664407 | 0.524064 | 0.585949 | 0.835634 | 0.025978 |
| `logistic_balanced_c1` | C=1.0; balanced weights | 0.745919 | 0.513937 | 0.788770 | 0.622363 | 0.836259 | 0.025628 |
| `forest_depth8` | 180 trees; depth 8; leaf minimum 5; balanced weights | 0.753726 | 0.524771 | 0.764706 | **0.622416** | **0.838833** | 0.253436 |

`forest_depth8` wins the declared validation-F1 rule by just **0.000053** over balanced logistic regression. That difference is tiny; it is not evidence of statistically significant superiority. The forest has slightly higher AUC and precision, whereas balanced logistic has higher recall and trains about ten times faster in this execution. The unweighted logistic model's higher accuracy hides its much lower churn recall. A production decision would compare uncertainty, inference latency and retention costs before claiming the forest is operationally preferable.

The selected forest's untouched test results are accuracy **0.757275**, precision **0.529091**, recall **0.778075**, F1 **0.629870**, and ROC-AUC **0.843089** ([JSON](reports/test_metrics.json)). The [figures](reports/figures/) include validation confusion matrices/ROC curves for all candidates and test plots for the winner. These are real held-out predictions, not illustrative numbers.

To inspect the reconstructed tracking store:

```sh
uv run mlflow ui --backend-store-uri sqlite:///.state/mlflow.db --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000` and choose experiment `telco-churn-track-a`. The CSV also satisfies the side-by-side comparison requirement without running the UI.

## Registry and serving

The executed experiment registered `telco-churn`, version 1, and called the real registry API for **Staging → Production**, verifying each returned state. [Transition evidence](reports/registry_transitions.json) links the model version to its training run and timestamps. The `champion` alias points to that version. Stage transitions are deprecated in newer MLflow; this pinned version deliberately demonstrates the required stages while serving uses the alias. These immediate transitions demonstrate the lifecycle, not an assertion that a real deployment completed a staging soak period.

```sh
uv run uvicorn telco_mlops.api:app --host 127.0.0.1 --port 8000
```

In another terminal in this directory:

```sh
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @reports/sample_request.json
```

The API resolves `models:/telco-churn@champion` through MLflow at startup, then loads/pins the resolved registry version for the lifetime of that process. Restart after promotion to pick up the new alias. Startup fails if no registered champion exists; run training first. `/docs` describes all 19 required input fields. `SeniorCitizen` is the categorical string `"0"` or `"1"`; `TotalCharges` may be null and is imputed by the training-fitted pipeline. Empty/oversized batches, unknown fields, invalid categories and negative charges receive HTTP 422. `/predict` returns the exact version, probability and decision at threshold 0.5. The example's probabilities are **0.053798** and **0.778444**.

`scripts/verify_runtime.py` starts a temporary real Uvicorn HTTP server, calls `/health` and `/predict`, compares probabilities with an independently registry-loaded model, verifies HTTP 422 for an empty batch, and stops the server. It also checks that all three runs contain metrics, actual model files and plots, and that the monitoring HTML is present in MLflow. [Recorded verification](reports/runtime_verification.json) contains the observed results. The service is a localhost assignment API; authentication and public deployment are outside this implementation.

`TELCO_STATE_DIR` changes the local state directory; `MLFLOW_TRACKING_URI` can point to another configured registry. `TELCO_MODEL_ALIAS` defaults to `champion`. When moving to a remote tracking server, configure an artifact location reachable by that server/serving host; the provided default is a local filesystem workflow.

## Monitoring and drift response

Monitoring independently partitions the verified data into a stratified **70% reference (4,930)** and **30% incoming (2,113)** sample. This models training-time versus production-time distributions for demonstration; there is no real temporal ordering. The incoming sample is copied before injecting seeded drift:

- Add `Normal(30, 5)` billing units to each `MonthlyCharges` value.
- Set 85% of Bernoulli-selected rows' `Contract` to `Month-to-month`.
- Turn 25% of Bernoulli-selected non-churn labels into churn to deliberately simulate target/prevalence drift. These labels are synthetic and are never used to train or promote a model.

Evidently's `DataDriftPreset` checks all 19 features. Numeric features use normalized Wasserstein distance; categoricals use Jensen–Shannon distance. The explicit distance threshold is 0.1 (these distances are **not p-values**). A separate `ValueDrift` on categorical `Churn` provides the target-drift equivalent. `AbsoluteMeanShift` implements Evidently's `SingleValueMetric` / `SingleValueCalculation` API and returns the absolute `MonthlyCharges` mean difference in billing units. Its custom test is `lt(10)`, so both upward and downward shifts of 10 or more trigger investigation.

Actual observed results ([summary](reports/drift_summary.json), [full JSON](reports/drift_report.json)):

| Signal | Reference | Drifted current | Evidently result |
|---|---:|---:|---|
| Mean MonthlyCharges | 64.950081 | 94.052377 | Wasserstein 0.962358 > 0.1: drift |
| Month-to-month share | 54.8479% | 93.4690% | Jensen–Shannon 0.326369 > 0.1: drift |
| Churn rate | 26.5314% | 46.0956% | Jensen–Shannon 0.144506 > 0.1: target drift |
| Custom absolute mean shift | — | 29.102296 billing units | `< 10` test fails |

Exactly the two perturbed features (`MonthlyCharges`, `Contract`) were flagged. The unmodified control comparison flags **zero** features and no target drift; its mean charge difference is only **0.627933**, and the custom test passes. Both [drift report](reports/drift_report.html) and [control report](reports/drift_control.html) are genuine standalone Evidently HTML exports, saved as MLflow artifacts under the `drift-monitor` run's `monitoring/` directory. Download/open them locally; GitHub's source view does not render the dashboard.

The global feature-drift-share threshold is 20%. Only 2/19 features drifted, so that aggregate signal alone would miss this case; the target-drift and custom billing tests independently trigger the documented review recommendation. Investigate billing/input changes and label quality, collect real delayed outcomes, then evaluate performance on recent labeled data. A changed price distribution or contract mix may invalidate learned relationships or calibration; label prevalence drift does not by itself prove concept drift or degraded predictive performance. Retrain a candidate on verified real data if needed, compare on an appropriate holdout, and repeat registry validation before promotion. We log the recommendation, and deliberately do not train on the injected labels. Airflow is optional and not included.

## Verification and project layout

Nine tests cover corrupt data rejection, disjoint stratified partitions, train-only imputation, validation-only winner selection, imbalanced-class metrics, deterministic non-mutating drift injection, the actual Evidently custom-metric/test API, and HTTP payload/response validation. `uv run ruff check src tests scripts` checks source quality. Runtime verification exercises actual artifacts and HTTP inference in addition to those tests.

```text
src/telco_mlops/    ingestion, training, monitoring, registry-backed FastAPI
tests/             invariant and API tests
scripts/           live runtime verification
data/              download cache (CSV ignored)
reports/           committed results, figures, provenance and HTML
.state/            local MLflow SQLite database and artifacts (ignored)
pyproject.toml      direct dependencies and commands
uv.lock            exact reproducible dependency resolution
```
