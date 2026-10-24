# Pre-implementation decisions

A fixed pipeline is insufficient because verification must decide whether to corroborate agreement, resolve contradictory evidence using an authoritative source, search for missing information, or ask for clarification based on each tool result.

Skill versus agent/tool: a Skill can describe verification policy, but cannot itself observe source failures or acquire evidence; one bounded agent chooses actions, while two bounded source tools expose distinct document collections without creating additional agents.

Chosen domain: a small, versioned fictional employee-policy corpus, used to make cross-source conflicts and missing evidence reproducible; this is not real organizational policy.
