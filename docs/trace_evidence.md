# Trace evidence

The workflow records a provider-neutral trace for every run. The trace includes
the route history, each agent span, durations, source count, citation count,
errors, and normalized agent results.

Generate the submission artefact with the configured OpenAI key:

```powershell
.venv\Scripts\python.exe -m multi_agent_research_lab.cli multi-agent `
  --query "Research GraphRAG state-of-the-art" `
  --trace-output reports/trace_evidence.json
```

The generated `reports/trace_evidence.json` is safe to share: it contains run
metadata and model outputs, but never API keys. LangSmith export is opt-in so a
slow tracing endpoint cannot block the research workflow. Enable it explicitly
with `LANGSMITH_ENABLED=true` when a dashboard trace is required.

Current local evidence: [`reports/trace_evidence.json`](../reports/trace_evidence.json).
