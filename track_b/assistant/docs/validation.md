# Validation record

- `uv sync --frozen --group dev` installed the lockfile environment on Python 3.12.3.
- `pytest -q`: **75 tests passed**. Includes all 51 inherited W15 tests and 24 new parameterized verification test instances.
- `ruff check agentic app tests`: passed.
- `python -m agentic evaluate`: 11 offline regression cases, 10 complete; the intentionally one-step verification case is a soft failure. 22/22 valid and appropriate tool calls; 2.91 mean decisions; 23,180 simulated estimated tokens. See committed raw traces.
- Live local `qwen2.5:1.5b` evaluation: **2/11 tasks complete (18.2%)**, **40/42 correct tool calls (95.2%)**, **5.00 mean decisions**, and **38,031 provider-reported tokens** across 55 model calls. Results are in `evaluation/live-results`; no paid model service was used. Completion criteria deliberately require the expected terminal action, so asking for clarification when the policy is absent does not pass an abstention case.
- Live failures: seven soft failures and two hard failures. The small model often copied placeholder citation IDs and misplaced clarification fields, repeated failed tools, or reached the step budget. Guards rejected its unsupported claims. It completed the remote-work answer and ambiguous-policy clarification; the current small-model setup is not reliable enough for unattended use. Live trajectories can vary between runs even at temperature zero; these reports were regenerated against the final controller.

## What was and was not validated

HTTP auth, endpoint validation, rate-limit integration, retained W15 chat functionality, bounded iteration/token/time behavior, source authority resolution, unsupported citation rejection, alternate tool order, context trimming, failure recovery, safe terminal rendering, malformed-action scoring, consecutive failure classification, and complete offline usage accumulation are covered by automated tests. Live usage accounting is based on the provider's response fields; provider failures cannot guarantee billing completeness.

The source corpus is deliberately small and fictional. Structured source facts are trusted fixture metadata; the system is not a general arbitrary-document entailment verifier. The two publishers maintain separate collections but are not claimed to be organizationally independent. Citing two collection publishers does not prove global factual accuracy. Relevance evaluation uses a lexical oracle and exact expected fact values, so paraphrase quality is not scored.

No external production account, cloud model, real company corpus, Qdrant/Redis deployment, or Docker image build was needed for the offline validation. Inherited infrastructure configuration is provided but this session does not claim to have deployed it. Costs remain null without explicit model prices. Tool failures recovered by a correct abstention remain in component failure logs rather than being hidden as successes.
