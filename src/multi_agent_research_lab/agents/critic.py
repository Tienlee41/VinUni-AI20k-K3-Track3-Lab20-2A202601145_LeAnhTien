"""Optional citation and consistency critic."""

import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        """Check that cited identifiers exist and append a review to the trace."""

        answer = state.final_answer or ""
        known = {str(source.metadata.get("source_id", source.title)) for source in state.sources}
        cited = set(re.findall(r"\[([^\]]+)\]", answer))
        unknown = sorted(cited - known)
        coverage = len(cited & known) / max(1, len(cited))
        finding = (
            "Citation audit passed."
            if not unknown
            else f"Unknown citation IDs: {', '.join(unknown)}"
        )
        state.add_agent_result(
            AgentResult(
                agent=AgentName.CRITIC,
                content=finding,
                metadata={"citation_coverage": coverage, "unknown_citations": unknown},
            )
        )
        state.add_trace_event("critic.audit", {"citation_coverage": coverage, "unknown": unknown})
        if unknown:
            state.errors.append(finding)
        return state
