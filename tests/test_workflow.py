from pathlib import Path

from multi_agent_research_lab.agents import AnalystAgent, ResearcherAgent, WriterAgent
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


def _offline_settings() -> Settings:
    return Settings(_env_file=None, openai_api_key=None, tavily_api_key=None, timeout_seconds=10)


def test_offline_corpus_search_returns_citation_ready_sources() -> None:
    settings = _offline_settings()
    client = SearchClient(
        settings=settings,
        corpus_root=Path("ai_agent_offline_research_corpus_v2"),
    )
    sources = client.search("multi-agent role specialization", max_results=3)
    assert sources
    assert all(source.metadata.get("source_id") for source in sources)
    assert all(source.snippet for source in sources)


def test_workflow_runs_end_to_end_with_offline_corpus() -> None:
    settings = _offline_settings()
    llm = LLMClient(settings=settings)
    search = SearchClient(
        settings=settings, corpus_root=Path("ai_agent_offline_research_corpus_v2")
    )
    workflow = MultiAgentWorkflow(
        settings=settings,
        researcher=ResearcherAgent(search, llm),
        analyst=AnalystAgent(llm),
        writer=WriterAgent(llm),
    )

    result = workflow.run(
        ResearchState(request=ResearchQuery(query="Research multi-agent role specialization"))
    )

    assert result.route_history == ["researcher", "analyst", "writer"]
    assert result.final_answer
    assert result.trace
    assert not result.errors
