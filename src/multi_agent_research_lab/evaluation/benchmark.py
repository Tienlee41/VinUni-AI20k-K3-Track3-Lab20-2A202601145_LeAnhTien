"""Benchmark skeleton for single-agent vs multi-agent."""

import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency and derive reproducible heuristic process metrics."""

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started
    answer = state.final_answer or ""
    source_ids = {str(source.metadata.get("source_id", source.title)) for source in state.sources}
    citations = set(re.findall(r"\[([^\]]+)\]", answer))
    valid_citations = citations & source_ids
    costs = [
        float(result.metadata["cost_usd"])
        for result in state.agent_results
        if result.metadata.get("cost_usd") is not None
    ]
    quality = _quality_score(state, valid_citations)
    failure_rate = 1.0 if state.errors or not answer else 0.0
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=sum(costs) if costs else None,
        quality_score=quality,
        citation_coverage=len(valid_citations) / max(1, len(citations)),
        failure_rate=failure_rate,
        notes="; ".join(state.errors)
        if state.errors
        else f"{len(state.route_history)} supervisor routes",
    )
    return state, metrics


def _quality_score(state: ResearchState, citations: set[str]) -> float:
    """A transparent smoke-test score; human review remains the authoritative quality metric."""

    score = 0.0
    score += 2.0 if state.sources else 0.0
    score += 2.0 if state.research_notes else 0.0
    score += 2.0 if state.analysis_notes else 0.0
    score += 2.0 if state.final_answer else 0.0
    score += min(2.0, float(len(citations)) / 2)
    return round(max(0.0, min(10.0, score - min(2.0, len(state.errors) * 0.5))), 2)
