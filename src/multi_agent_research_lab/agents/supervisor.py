"""Supervisor / router."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, max_iterations: int | None = None) -> None:
        self.max_iterations = max_iterations or get_settings().max_iterations

    def run(self, state: ResearchState) -> ResearchState:
        """Choose the next missing artifact, with a hard iteration limit."""

        limit = state.max_iterations or self.max_iterations
        if state.final_answer:
            route = "done"
        elif state.iteration >= limit:
            route = "done"
            state.errors.append(f"Maximum supervisor iterations ({limit}) reached")
        elif not state.sources or not state.research_notes:
            route = AgentName.RESEARCHER.value
        elif not state.analysis_notes:
            route = AgentName.ANALYST.value
        else:
            route = AgentName.WRITER.value
        state.record_route(route)
        state.add_agent_result(
            AgentResult(
                agent=AgentName.SUPERVISOR,
                content=f"Next route: {route}",
                metadata={"iteration_limit": limit},
            )
        )
        state.add_trace_event("supervisor.route", {"next": route, "iteration": state.iteration})
        return state
