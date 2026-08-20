# Multi-Agent Research System Design

## Problem

Given a research question, the system must retrieve relevant evidence, assess
the evidence, and write a grounded answer with traceable citations. It supports
both an online Tavily search path and an offline search path over
`ai_agent_offline_research_corpus_v2`.

## Why multi-agent?

A single agent can answer short questions efficiently, but long research tasks
mix retrieval, evidence analysis, and technical writing. Separating those
responsibilities makes handoffs explicit, preserves intermediate artifacts, and
allows each stage to be inspected or retried. The trade-off is higher latency,
token usage, and orchestration complexity, so the baseline remains available.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Select the next missing artifact and stop safely | Shared state | Route history | Max-iteration stop |
| Researcher | Retrieve, deduplicate, and summarize evidence | Query | Sources, research notes | Search fallback to offline corpus |
| Analyst | Extract claims, limitations, and evidence gaps | Research notes, sources | Analysis notes | Explicit uncertainty and source IDs |
| Writer | Produce the final grounded answer | Research and analysis notes | Final answer with citations | Deterministic evidence-based fallback |
| Critic | Audit citation IDs and coverage | Final answer, sources | Audit result and trace event | Records unknown citations |

## Shared state

`ResearchState` contains the request, iteration counter, route history, source
documents, research notes, analysis notes, final answer, agent results, trace
events, errors, and the configured iteration limit. This is enough to debug a
handoff without relying on an unstructured conversation transcript.

## Routing policy

```text
Supervisor
  ├─ no sources or research notes → Researcher → Supervisor
  ├─ no analysis notes             → Analyst    → Supervisor
  ├─ no final answer               → Writer     → End
  ├─ final answer exists           → End
  └─ iteration limit reached      → End with recorded error
```

## Guardrails

- Max iterations: configured by `MAX_ITERATIONS`, default 6.
- Timeout: configured by `TIMEOUT_SECONDS`, default 60 seconds.
- Retry: OpenAI client uses SDK retries; provider errors fall back to a
  deterministic local completion.
- Search fallback: Tavily errors or missing search support fall back to the
  local offline corpus.
- Validation: Pydantic schemas validate queries, state, sources, and metrics.
- Observability: every route and agent span records duration, status, and
  relevant metadata.

## Benchmark plan

The configured benchmark queries run through both the baseline and multi-agent
runner. The report records latency, estimated cost, heuristic quality,
citation coverage, and failure rate. Human peer review remains the authority
for final quality scoring; the automated score is intended for regression and
smoke testing.
