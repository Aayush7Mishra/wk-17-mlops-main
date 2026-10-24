# Verification evaluation

Mode: **live**. Offline results test a handwritten policy, not model quality. Live results use a small synthetic policy corpus and lexical tool relevance oracle; semantic correctness is scoped to structured facts. Token estimates are labeled. The final step-budget case deliberately cannot complete.
Model/policy: `qwen2.5:1.5b`.

Completion: **18.2%**; tool correctness: **95.2%**; mean steps: **5.00**.

Tokens include every iteration. Offline counts use ceil(JSON characters / 4) and are simulated estimates, not model consumption. Live counts use provider usage when available. Cost is null unless explicit input/output rates are configured.

| Query | Complete | Outcome | Steps | Correct tools | Tokens | Failure class |
|---|---|---|---:|---:|---:|---|
| agreement | False | needs_clarification | 3 | 2/2 | 2069 | soft_failure |
| stale_conflict | False | needs_clarification | 3 | 2/2 | 2060 | soft_failure |
| remote | True | verified | 8 | 5/5 | 7108 | - |
| ambiguous | True | needs_clarification | 3 | 0/2 | 1277 | - |
| contractor_scope | False | budget_exhausted | 8 | 7/7 | 6387 | soft_failure |
| missing_policy | False | needs_clarification | 4 | 3/3 | 2353 | soft_failure |
| single_source | False | needs_clarification | 5 | 4/4 | 3074 | soft_failure |
| registry_outage | False | needs_clarification | 4 | 3/3 | 2565 | hard_failure |
| malformed_output | False | budget_exhausted | 8 | 7/7 | 5885 | soft_failure |
| tool_timeout | False | budget_exhausted | 8 | 6/6 | 4844 | hard_failure |
| step_budget | False | budget_exhausted | 1 | 1/1 | 409 | soft_failure |

## Failure observations

Hard failure means an explicit tool/provider outage or timeout; soft failure means a plausible invalid decision, malformed data, wrong answer, or incomplete task; cascading soft failure means repeating a rejected decision without using feedback. Recovered failures remain in traces even when safe abstention satisfies the task.

- remote, step 3: soft_failure: unobserved citation.
- remote, step 6: soft_failure: unobserved citation.
- contractor_scope, step 3: soft_failure: clarification requires a concise question.
- registry_outage, step 2: hard_failure: search_policy_registry: ConnectionError.
- malformed_output, step 2: soft_failure: malformed source response: documents must be a list.
- malformed_output, step 4: soft_failure: clarification requires a concise question.
- malformed_output, step 5: soft_failure: malformed source response: documents must be a list.
- malformed_output, step 7: soft_failure: malformed source response: documents must be a list.
- tool_timeout, step 2: hard_failure: search_policy_registry: TimeoutError.
- tool_timeout, step 4: hard_failure: search_policy_registry: TimeoutError.
- tool_timeout, step 5: soft_failure: clarification requires a concise question.
- tool_timeout, step 7: hard_failure: search_policy_registry: TimeoutError.
- tool_timeout, step 8: soft_failure: two distinct source publishers must be consulted.

The step-budget case intentionally has one iteration: stopping safely does not count as answering its policy question. The harness records a soft task failure. Unit tests force repeated invalid actions and outages to check all taxonomy branches.
