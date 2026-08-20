"""Single-agent baseline used by the CLI and benchmark runner."""

from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


def run_baseline(
    query: str | ResearchQuery,
    search_client: SearchClient | None = None,
    llm_client: LLMClient | None = None,
) -> ResearchState:
    """Run one model call after one retrieval step, mirroring a simple assistant."""

    request = query if isinstance(query, ResearchQuery) else ResearchQuery(query=query)
    search = search_client or SearchClient()
    llm = llm_client or LLMClient()
    state = ResearchState(request=request)
    state.add_trace_event("baseline.start", {"query": request.query})
    state.sources = search.search(request.query, request.max_sources)
    evidence = "\n".join(
        f"[{source.metadata.get('source_id', source.title)}] {source.title}: {source.snippet}"
        for source in state.sources
    )
    response = llm.complete(
        "You are a single-agent research assistant. Answer only from supplied evidence.",
        f"Question: {request.query}\nEvidence:\n{evidence}\n"
        "Write a concise answer and cite source IDs in square brackets.",
    )
    state.final_answer = (
        _web_search_fallback(request, state) if response.error else response.content or evidence
    )
    state.add_agent_result(
        AgentResult(
            agent=AgentName.WRITER,
            content=state.final_answer,
            metadata={
                "cost_usd": response.cost_usd,
                "source_count": len(state.sources),
                "provider_fallback": bool(response.error),
            },
        )
    )
    state.add_trace_event(
        "baseline.complete",
        {"source_count": len(state.sources), "provider_fallback": bool(response.error)},
    )
    return state


def _web_search_fallback(request: ResearchQuery, state: ResearchState) -> str:
    """Render Tavily evidence when the LLM provider is unavailable or times out."""

    lines = [
        "## Web-search baseline",
        "",
        f"OpenAI did not respond in time, so this answer lists the retrieved evidence for: "
        f"{request.query}",
        "",
    ]
    for source in state.sources[:5]:
        citation = source.metadata.get("source_id", source.url or source.title)
        lines.append(f"- [{citation}] **{source.title}** — {source.snippet[:350]}")
    return "\n".join(lines)
