"""Command-line entrypoint for the research lab."""

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.baseline import run_baseline
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the single-agent baseline."""

    _init()
    request = _parse_query(query)
    state, metrics = run_benchmark("baseline", request.query, run_baseline)
    console.print(Panel.fit(state.final_answer or "No answer", title="Single-Agent Baseline"))
    console.print(f"Latency: {metrics.latency_seconds:.2f}s | Quality: {metrics.quality_score}/10")


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    trace_output: Annotated[
        Path | None,
        typer.Option("--trace-output", help="Write local trace evidence to JSON"),
    ] = None,
) -> None:
    """Run the Supervisor -> Researcher -> Analyst -> Writer workflow."""

    _init()
    state = ResearchState(request=_parse_query(query))
    workflow = MultiAgentWorkflow()
    result = workflow.run(state)
    if trace_output is not None:
        payload = {
            "query": result.request.query,
            "route_history": result.route_history,
            "trace": result.trace,
            "agent_results": [item.model_dump() for item in result.agent_results],
            "errors": result.errors,
        }
        path = LocalArtifactStore(trace_output.parent).write_text(
            trace_output.name,
            json.dumps(payload, indent=2, ensure_ascii=False),
        )
        console.print(f"Trace evidence written to {path}")
    console.print(result.model_dump_json(indent=2))


@app.command()
def benchmark(
    output: Annotated[Path, typer.Option("--output", help="Report output path")] = Path(
        "reports/benchmark_report.md"
    ),
) -> None:
    """Benchmark baseline and multi-agent runs on configured queries."""

    _init()
    config_path = Path("configs/lab_default.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    queries = config.get("benchmark", {}).get("queries", ["Explain multi-agent systems"])
    metrics = []
    for query in queries:
        _, baseline_metrics = run_benchmark("baseline", query, run_baseline)
        _, multi_metrics = run_benchmark(
            "multi-agent",
            query,
            lambda item: MultiAgentWorkflow().run(ResearchState(request=ResearchQuery(query=item))),
        )
        metrics.extend([baseline_metrics, multi_metrics])
    report = render_markdown_report(metrics)
    path = LocalArtifactStore(output.parent).write_text(output.name, report)
    console.print(f"Benchmark report written to {path}")


if __name__ == "__main__":
    app()
