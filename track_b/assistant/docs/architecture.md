# W16 architecture

```mermaid
flowchart TD
    U[User] --> UI[W15 Streamlit UI]
    UI --> CHAT[W15 chat API]
    CHAT --> BASE[RAG + tools + provider fallback + cache]
    BASE --> Q[Qdrant / memory, Redis / memory]
    U --> CLI[Verification CLI]
    U --> API[POST /api/v1/verify]
    API --> AUTH[Authentication + rate limit + concurrency bound]
    CLI --> LOOP
    AUTH --> LOOP[Bounded VerificationAgent loop]
    LOOP --> CTX[Structured notes + compact observations]
    CTX --> MODEL[OpenAI-compatible planner]
    CTX -. explicit regression mode .-> SIM[Handwritten OfflinePolicy]
    MODEL --> ACTION{Next model-selected action}
    SIM --> ACTION
    ACTION -->|search| TOOL[Validated bounded tool call]
    TOOL --> HB[Handbook collection]
    TOOL --> REG[Policy registry collection]
    HB --> OBS[Result / error observation]
    REG --> OBS
    OBS --> NOTES[Deduplicate + cap + excerpt]
    NOTES --> LOOP
    ACTION -->|finish| GUARD[Citation + provenance + authority checks]
    GUARD -->|rejected with feedback| LOOP
    GUARD -->|verified or abstained| OUT[Structured result]
    ACTION -->|clarify| OUT
    LOOP -->|step/token budget or model error| OUT
    LOOP --> TRACE[External full trace + usage + failures]
    EVAL[Custom evaluation harness] --> LOOP
    OUT --> EVAL
    TRACE --> EVAL
    EVAL --> REPORT[JSON / JSONL / Markdown results]
```

The model owns action choice and stopping proposals. Guards own evidence validation, source access, and budgets. Default bounds: eight model decisions, two documents per search, eight notes, 600 characters per excerpt, 30 seconds per model call, and five seconds per tool call. Model calls have no hidden retries. The 12,000-token threshold is checked before and after calls; the final response may cross it and is then stopped.

Each request has a fresh evidence ledger, notes, history, and trace. HTTP requests share only the stateless planner client/read-only corpus and a bounded concurrency semaphore. No raw chain-of-thought is requested: the trace records action JSON and a concise operational reason when the model supplies one. Abstention and clarification return fixed controller text, preserving unverified model prose only in the trace.
