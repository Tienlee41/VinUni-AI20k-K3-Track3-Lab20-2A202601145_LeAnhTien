"""Analyst agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Extract claims, evidence strength, tensions, and evidence gaps."""

        notes = state.research_notes or "No research notes are available."
        source_ids = [
            str(source.metadata.get("source_id", source.title)) for source in state.sources
        ]
        prompt = (
            f"Question: {state.request.query}\nResearch notes:\n{notes}\n\n"
            f"Available source IDs: {', '.join(source_ids)}\n"
            "Produce a compact claim ledger: supported claims, tensions/limitations, and "
            "recommendations. Cite every claim with an available ID."
        )
        response = self.llm_client.complete(
            "You are an evidence analyst. Distinguish evidence from inference and uncertainty.",
            prompt,
        )
        fallback = self._fallback_analysis(state)
        state.analysis_notes = response.content or fallback
        state.add_agent_result(
            AgentResult(
                agent=AgentName.ANALYST,
                content=state.analysis_notes,
                metadata={"source_ids": source_ids, "cost_usd": response.cost_usd},
            )
        )
        state.add_trace_event("analyst.synthesis", {"source_count": len(state.sources)})
        return state

    @staticmethod
    def _fallback_analysis(state: ResearchState) -> str:
        ids = ", ".join(
            f"[{source.metadata.get('source_id', source.title)}]" for source in state.sources
        )
        return (
            f"Supported evidence is available from {ids or 'no identified sources'}. "
            "The main limitation is that evidence quality and applicability must be checked "
            "before generalizing. Treat synthetic documents as benchmark evidence, "
            "not real publications."
        )
