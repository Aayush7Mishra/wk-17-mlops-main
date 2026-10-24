# W16 — Agentify the Assistant

A self-contained extension of the W15 FastAPI/Streamlit assistant with model-directed, cross-source policy verification. The original RAG, document ingestion, calculator/time tools, batching, provider fallback, caching, rate limits, and UI remain available. Verification adds `/api/v1/verify` and a CLI. The checked-in corpus describes a **fictional company**, not real employment policy.

## Run and reproduce

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are required. Dependencies are pinned in `uv.lock`.

```bash
uv sync --frozen --group dev
uv run --frozen pytest
uv run --frozen ruff check app agentic tests
uv run --frozen python -m agentic verify 'How many annual leave days do employees receive?'
uv run --frozen python -m agentic evaluate --output evaluation/results
```

The default `offline` mode uses an explicitly labeled handwritten policy to exercise the loop without credentials; it is **not an LLM**, and its scores are regression results. Live mode performs a genuine model call on every decision:

```bash
# OpenAI-compatible local model; start Ollama and pull the model first.
export AGENT_BASE_URL=http://localhost:11434/v1
export AGENT_MODEL=qwen2.5:1.5b
export OPENAI_API_KEY=ollama
uv run --frozen python -m agentic verify 'How many annual leave days?' --mode live
uv run --frozen python -m agentic evaluate --mode live --output evaluation/live-results
# For a cloud model, set AGENT_BASE_URL, AGENT_MODEL and OPENAI_API_KEY accordingly.
```

Start the combined W15/W16 API without infrastructure for local smoke testing:

```bash
PRIMARY_PROVIDER=demo PRIMARY_MODEL=demo EMBEDDING_PROVIDER=hash VECTOR_STORE=memory \
  AGENT_MODE=offline uv run --frozen uvicorn app.main:app --host 127.0.0.1 --port 8000
curl -s http://localhost:8000/api/v1/verify -H 'Content-Type: application/json' \
  -d '{"query":"How many annual leave days do employees receive?"}'
uv run --frozen streamlit run ui/streamlit_app.py
```

The UI retains W15 chat; use the CLI or API `/docs` for verification. Set `APP_API_KEY` to protect the API and send `X-API-Key`. For live HTTP verification set `AGENT_MODE=live`, `AGENT_BASE_URL`, `AGENT_MODEL`, and `AGENT_API_KEY`; the CLI reads `OPENAI_API_KEY`. For the original infrastructure-backed RAG deployment, copy `.env.example` to `.env`, configure the W15 provider variables, and run `docker compose up --build`. Full inherited instructions are in [the W15 reference](docs/w15-readme.md).

## Design decisions (approximately one page)

**Why a loop.** A fixed pipeline is insufficient because verification must decide whether to corroborate agreement, resolve contradictory evidence using an authoritative source, search for missing information, or ask for clarification based on each tool result. This decision was recorded before implementation in [design-intent.md](docs/design-intent.md). `OpenAIPlanner` selects one action after each observation; the application enforces permissions and budgets without prescribing tool order. The offline policy deliberately simulates only a limited set of trajectories.

### a. Context Engineering Technique

`VerificationAgent.run` re-ranks retrieval lexically, caps each tool response to `top_k=2`, replaces repeated document IDs, and retains eight structured evidence notes with 600-character excerpts. The next prompt includes these notes and compact action outcomes, rather than replaying verbose tool messages. Raw source responses, prompt contexts, model outputs, concise decision reasons when returned, and per-step usage remain in an external trace. This prevents repeat searches from saturating the model context while retaining source/version provenance. Evicted notes can be retrieved again; the full evidence ledger remains available to the citation guard.

### b. Agentic Pattern

One agent chooses between two bounded source tools, clarification, and completion. Source collection access needs no distinct reasoning specialty, and these small searches do not justify parallel coordination. A second agent would add tokens without addressing a measured sequential bottleneck or skill dilution. Structured notes address context saturation. Deterministic evidence checks reduce the self-verification problem: a verified claim must match observed structured facts and cite two publishers; current authoritative entries resolve stale conflicts. These checks establish corpus support, not real-world truth or perfect relevance. Abstention and clarification use fixed safe text; unverified model prose stays in the trace. The primary planner fails explicitly when unavailable.

### c. Evaluation Harness

The custom [harness](agentic/evaluation.py) runs 11 fixed cases: agreement, stale conflict, remote work, ambiguity, employment scope, missing policy, one-source evidence, three injected tool failures, and an intentionally inadequate step budget. It measures exact task/status-and-fact completion, valid/appropriate tool names and arguments, trajectory length, and every unsuccessful case. Tool appropriateness uses case-specific query terms; it is a transparent lexical oracle, not a semantic judge. Hard failures are explicit outages; soft failures include invalid data/decisions or incomplete tasks; consecutive repeated rejected decisions are cascading soft failures. Unit tests force each taxonomy branch. Offline results are 10/11 completion, 22/22 correct calls, and 2.91 mean steps; the deliberately one-step case remains a logged failure. The real local Qwen 1.5B run completed 2/11 tasks with 95.2% tool correctness; its schema/citation mistakes were rejected rather than accepted as answers.

### Additional requirements

**Skill versus agent.** A Skill can describe verification policy, but cannot itself observe failures or acquire evidence; the single agent needs adaptive decisions and the two tools need executable source access. No extra agent is introduced.

**Tokens and cost.** Every iteration records input/output/total usage. Live calls use provider-reported counts; missing counts use labeled character-based estimates. Offline counts are simulated estimates, never claimed as model consumption. Failed provider calls mark accounting incomplete because billed tokens may be unknown. Optional CLI input/output prices produce a cost estimate; otherwise cost is null. A token threshold prevents another call after exhaustion but a final in-flight response can exceed it. There is no multi-agent coordination premium to compare.

**Failure injection.** Making the registry unavailable, returning malformed data, or forcing a timeout yields a visible failed observation. The offline policy seeks alternate evidence then abstains; it never verifies the stale handbook alone. Raw traces and the failure report retain these events even when safe abstention satisfies the scenario.

**Tool versus agent boundary.** The two source tools are bounded read-only calls over separate local collections, with capped results, input validation, and a timeout. Neither is itself stateful or multi-step, so no external agent-to-agent interaction exists. An external search service could replace a collection behind the same bounded contract; it would need a separate trust/provenance review before its output could populate the authoritative structured facts.

## Architecture and deliverables

See the [updated architecture diagram](docs/architecture.md). Source is split into inherited `app/`, `ui/`, and the new `agentic/` package. Evaluation fixtures are in `evaluation/cases.json`; [the regression report](evaluation/results/report.md), [the live model report](evaluation/live-results/report.md), [complete structured results](evaluation/results/results.json), [raw traces](evaluation/results/traces.jsonl), and [failure log](evaluation/results/failures.json) are checked in. [Validation notes](docs/validation.md) distinguish verified behavior from infrastructure/model limitations.
