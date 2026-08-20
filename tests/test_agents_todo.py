from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_routes_missing_artifacts() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    SupervisorAgent(max_iterations=4).run(state)
    assert state.route_history == ["researcher"]


def test_supervisor_stops_after_final_answer() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"), final_answer="done"
    )
    SupervisorAgent(max_iterations=4).run(state)
    assert state.route_history == ["done"]
