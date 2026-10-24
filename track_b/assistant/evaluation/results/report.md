# Verification evaluation

Mode: **offline_simulation**. Offline results test a handwritten policy, not model quality. Live results use a small synthetic policy corpus and lexical tool relevance oracle; semantic correctness is scoped to structured facts. Token estimates are labeled. The final step-budget case deliberately cannot complete.
Model/policy: `handwritten-offline-policy`.

Completion: **90.9%**; tool correctness: **100.0%**; mean steps: **2.91**.

Tokens include every iteration. Offline counts use ceil(JSON characters / 4) and are simulated estimates, not model consumption. Live counts use provider usage when available. Cost is null unless explicit input/output rates are configured.

| Query | Complete | Outcome | Steps | Correct tools | Tokens | Failure class |
|---|---|---|---:|---:|---:|---|
| agreement | True | verified | 3 | 2/2 | 2394 | - |
| stale_conflict | True | verified | 3 | 2/2 | 2404 | - |
| remote | True | verified | 3 | 2/2 | 2081 | - |
| ambiguous | True | needs_clarification | 1 | 0/0 | 549 | - |
| contractor_scope | True | needs_clarification | 1 | 0/0 | 558 | - |
| missing_policy | True | insufficient_evidence | 4 | 3/3 | 3011 | - |
| single_source | True | insufficient_evidence | 4 | 3/3 | 2771 | - |
| registry_outage | True | insufficient_evidence | 4 | 3/3 | 3021 | - |
| malformed_output | True | insufficient_evidence | 4 | 3/3 | 3031 | - |
| tool_timeout | True | insufficient_evidence | 4 | 3/3 | 2804 | - |
| step_budget | False | budget_exhausted | 1 | 1/1 | 556 | soft_failure |

## Failure observations

Hard failure means an explicit tool/provider outage or timeout; soft failure means a plausible invalid decision, malformed data, wrong answer, or incomplete task; cascading soft failure means repeating a rejected decision without using feedback. Recovered failures remain in traces even when safe abstention satisfies the task.

- registry_outage, step 2: hard_failure: search_policy_registry: ConnectionError.
- malformed_output, step 2: soft_failure: malformed source response: documents must be a list.
- tool_timeout, step 2: hard_failure: search_policy_registry: TimeoutError.

The step-budget case intentionally has one iteration: stopping safely does not count as answering its policy question. The harness records a soft task failure. Unit tests force repeated invalid actions and outages to check all taxonomy branches.
