# Week 17 — MLOps

Both required tracks are implemented as independent, reproducible projects:

| Track | Project | Workflow |
|---|---|---|
| A: predictive ML | [track_a](track_a/README.md) | IBM Telco data → three MLflow training runs → model registry → FastAPI serving → Evidently drift monitoring |
| B: agentic AI | [track_b](track_b/README.md) | W15/W16 assistant → three prompt/configuration experiments → full traces → two LLM judges → Evidently regression gate |

The `main` branch integrates both tracks. The `track-a` and `track-b` branches provide each track separately, as allowed by the assignment. Each track has its own `pyproject.toml` and committed `uv.lock`; run `uv sync --locked` in the desired track directory. Separate environments intentionally preserve the different Evidently APIs used by the assignment: current custom metrics for tabular monitoring and the explicit TestSuite/BinaryClassificationPromptTemplate API for LLM regression tests.

Start with the linked track READMEs for executed metric tables, environment decisions, model/prompt selection, monitoring interpretation, commands, and limitations. The original assignment PDF is retained here. Generated HTML reports, portable run exports, raw synthetic agent traces, and registry transition records are committed. Local MLflow databases, model binaries, caches, downloaded datasets, and secrets are ignored and rebuilt by the documented commands.

No Airflow bonus is claimed. Both workflows expose explicit command-line entry points suitable for a later scheduler. Model or prompt promotion depends on the recorded evaluation criteria; failures are retained as evidence.
