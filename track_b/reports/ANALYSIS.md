# Executed prompt/configuration comparison

These are real local model calls with provider-reported usage, not replayed or simulated agent scores. The fixed judge is Qwen 2.5 3B for all three versions. Full prompts, traces, verdicts, run IDs, HTML test reports and per-case token totals sit in the version directories. Exact model digests are recorded in [model_inventory.json](model_inventory.json).

| Version | Agent | Max steps / top-k | Harness completion | Mean steps | Agent tokens | Judge tokens | LLM checks passing | Promoted |
|---|---|---|---:|---:|---:|---:|---:|---|
| v1 | qwen2.5:1.5b | 2 / 1 | 0.0% | 2.00 | 5,154 | 5,030 | 0.0% | No |
| v2 | qwen2.5:1.5b | 6 / 2 | 16.7% | 5.00 | 22,151 | 7,374 | 8.3% | No |
| v3 | qwen2.5:3b | 8 / 2 | 50.0% | 3.00 | 17,918 | 5,610 | 16.7% | No |

## What changed and what won

v1 could not finish any case within two steps. The traces show the budget consumed by retrieval or rejected incomplete claims. v2 allocated six steps and explicitly requested two relevant publishers and exact citations. It completed the remote-work case, but still invented citation suffixes, supplied malformed clarification fields, and tried to use carryover evidence for annual entitlement.

v3 responded with explicit action examples, an eight-step budget and the larger 3B actor. It stopped safely for missing policy and registry outage, and routed the contractor case to clarification. It still failed the answerable leave, carryover and remote-work queries. Thus v3 has the highest harness completion (3/6), and consumes 17,918 agent tokens versus v2's 22,151; however its mean agent latency is 2,345 ms versus 1,358 ms. Its gain is safer terminal behavior, not improved factual answering: v2 produced one verified factual answer while v3 produced none. There is no production winner; all promotion gates remain blocked.

The completion oracle checks exact required facts for verified cases and the expected safe terminal status for abstention/clarification cases. In particular, the contractor score does **not** prove that the generic safe clarification is maximally useful. The model's original proposed question is retained in its raw trace; the controller renders a fixed question to prevent unsupported assertions leaking through a clarification. This usability limitation helps explain why the correctness judge rejected that case.

## Regression failures and judge sanity check

All four Evidently checks (correctness, groundedness, completion, accounting) run for every version. Each suite passes only its complete-token-accounting check; the other three fail because they require every case to pass. `pct_tests_passed` is the percentage of the 12 individual LLM checks, not the percentage of four aggregate suite checks.

- v1: every case failed completion and both judges. Its budget messages give no answer, so correctness rejection is justified.
- v2: `remote` is canonically correct and its correctness judge agrees. The grounding judge rejects it despite both observed sources supporting the exact same 2-day full-time allowance. Other cases remain incomplete or request clarification where the evidence should answer the question.
- v3: correctness passes for `missing_policy` and `registry_outage`, both explicit abstentions matching their references. It rejects the unanswered factual questions and generic contractor clarification. The grounding judge rejects all six, including those safe abstentions.

The grounding verdicts do **not** line up with manual inspection in several cases. For v3 `registry_outage`, the judge treats the archived 20-day entitlement as the answer, despite the unavailable current registry and the instruction to respect authoritative-current evidence. For `missing_policy`, it cites an unrelated remote-work allowance when judging a meal-policy abstention. These are evaluator mistakes: the final candidate makes no policy assertion, and the unrelated source is not evidence about meal allowance. The report retains these raw errors rather than overwriting their labels or relaxing the gate until it passes. A failing or unreliable judge is a reason to block promotion and improve evaluation, not evidence that a grounded system should return the judge's invented answer.

Tool-call correctness is 100% under the explicit lexical/schema oracle across these final runs. This only establishes permitted tools, well-formed arguments and overlapping query terms; it cannot establish that retrieval found the right fact. Low task completion despite this metric illustrates that limitation. Invalid attempted actions are included in the denominator if they occur.

Decision-summary coverage is 0% for v1/v2 and 66.7% for v3. The model was instructed to provide a brief reason, but sometimes omitted it; raw actions, observations and contexts are still retained, and missing summaries are never fabricated. All provider token usage was returned; per-query totals include both judge calls. Dollar cost is unmeasured for local inference.

## Reproduction and limits

Use the three commands in the parent README with the recorded models. Small-model results can vary with runtime/quantization, and all six cases were used for development rather than held out. See [CALIBRATION.md](CALIBRATION.md) for observed failures that motivated the revisions and judge setup. No statistical significance or production-quality claim follows from this small synthetic corpus. Next steps are a separate held-out set, reviewed judge labels, a stronger independent evaluator, and improving factual completion before any release. The current safe-failure gates remain enabled.
