"""Researcher agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self, search_client: SearchClient | None = None, llm_client: LLMClient | None = None
    ) -> None:
        self.search_client = search_client or SearchClient()
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Retrieve ranked evidence and turn it into citation-ready notes."""

        sources = self.search_client.search(state.request.query, state.request.max_sources)
        state.sources = self._deduplicate(sources)
        evidence = "\n".join(self._format_source(source) for source in state.sources)
        prompt = (
            f"Research question: {state.request.query}\n\n"
            f"Evidence packets:\n{evidence}\n\n"
            "Summarize only supported findings. Preserve source IDs in square brackets. "
            "Separate direct evidence from limitations."
        )
        response = self.llm_client.complete(
            "You are a careful research agent. Do not invent sources or facts.", prompt
        )
        state.research_notes = response.content or evidence
        state.add_agent_result(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=state.research_notes,
                metadata={"source_count": len(state.sources), "cost_usd": response.cost_usd},
            )
        )
        state.add_trace_event(
            "researcher.search",
            {
                "query": state.request.query,
                "source_count": len(state.sources),
                "provider": "configured",
            },
        )
        return state

    @staticmethod
    def _format_source(source: SourceDocument) -> str:
        document = source
        identifier = document.metadata.get("source_id", document.title)
        synthetic = " (synthetic)" if document.metadata.get("is_synthetic") else ""
        return f"[{identifier}]{synthetic} {document.title}: {document.snippet}"

    @staticmethod
    def _deduplicate(sources: list[SourceDocument]) -> list[SourceDocument]:
        seen: set[str] = set()
        result: list[SourceDocument] = []
        for source in sources:
            identifier = str(source.metadata.get("source_id", source.title))
            if identifier not in seen:
                seen.add(identifier)
                result.append(source)
        return result
