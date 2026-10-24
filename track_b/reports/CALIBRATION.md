# Development observations

The final comparison uses one fixed Qwen 2.5 3B judge and is exported in `comparison.csv`. Earlier exploratory runs used the 1.5B model as both actor and judge. They exposed substantive problems before the final comparison:

- A two-step agent budget allowed retrieval but no completion step.
- Even with six steps, the small model nested clarification under `arguments`, added `.id` to source IDs, mixed carryover sources with leave entitlement, and retried outages. Those raw decisions motivated v3's concrete terminal schemas and the larger actor.
- The 1.5B judge rejected every one of 36 checks in one three-version run. It hallucinated that a correctly stated 2-day remote-work answer asserted 22 days, and treated a safe clarification as a claim about contractor entitlements. Such evaluator failures are not evidence that all candidate responses are unsafe.
- Candidate text was separated from background evidence in the Evidently template; generic grading instructions replaced task-specific numeric hints, and repeated category wording was removed. All final runs use the same resulting judge implementation. Model/revision selection used these six cases, so this is a development comparison, not a held-out benchmark.

No earlier pilot metric is silently mixed into the final three-version table. SQLite may retain exploratory runs locally; the committed summary run IDs identify the actual compared runs. The six current traces per version, raw judge prompts/verdicts and token counts are stored beside the summaries. No tool trace or judge response is invented.
