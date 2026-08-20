"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics and a short interpretation to markdown."""

    lines = [
        "# Benchmark Report",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} "
            f"| {citation} | {failure} | {item.notes.replace('|', '/')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Quality is a transparent automated smoke-test score, not a substitute "
            "for peer review. "
            "Citation coverage counts cited identifiers that match retrieved source IDs.",
            "",
            "## Failure-mode notes",
            "",
            "The main operational risks are retrieval failure, unsupported claims, "
            "provider timeouts, and citation mismatch. The workflow records route history, "
            "trace events, errors, and token "
            "usage so a reviewer can identify which stage needs correction.",
        ]
    )
    return "\n".join(lines) + "\n"
