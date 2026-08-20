"""Writer agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Synthesize a concise answer from the evidence ledger with citations."""

        sources = "\n".join(
            f"[{source.metadata.get('source_id', source.title)}] {source.title}: {source.snippet}"
            for source in state.sources
        )
        prompt = (
            f"Question: {state.request.query}\n\n"
            f"Research notes:\n{state.research_notes or 'None'}\n\n"
            f"Analysis:\n{state.analysis_notes or 'None'}\n\nSources:\n{sources}\n\n"
            "Write a clear answer for technical learners. Use headings, state uncertainty, "
            "and include source IDs in square brackets next to factual claims."
        )
        response = self.llm_client.complete(
            "You are a technical writer. Every factual claim must be grounded "
            "in the supplied evidence.",
            prompt,
        )
        state.final_answer = response.content or self._fallback_answer(state)
        state.add_agent_result(
            AgentResult(
                agent=AgentName.WRITER,
                content=state.final_answer,
                metadata={
                    "citation_count": self._citation_count(state.final_answer),
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "writer.complete",
            {
                "answer_length": len(state.final_answer),
                "citation_count": self._citation_count(state.final_answer),
            },
        )
        return state

    @staticmethod
    def _fallback_answer(state: ResearchState) -> str:
        citations = " ".join(
            f"[{source.metadata.get('source_id', source.title)}]" for source in state.sources
        )
        return (
            "## Answer\n\n"
            f"{state.analysis_notes or state.research_notes or 'No answer could be grounded.'}\n\n"
            f"## Evidence\n\nThe answer is based on the retrieved evidence {citations}."
        )

    @staticmethod
    def _citation_count(answer: str) -> int:
        import re

        return len(re.findall(r"\[[^\]]+\]", answer))
