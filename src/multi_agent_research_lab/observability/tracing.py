"""Tracing hooks.

This file intentionally avoids binding to one provider. The emitted span shape can be
forwarded to LangSmith, Langfuse, OpenTelemetry, or a JSON exporter by the application.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Create a provider-neutral span with timing and completion status."""

    started = perf_counter()
    span: dict[str, Any] = {
        "name": name,
        "attributes": attributes or {},
        "duration_seconds": None,
        "status": "ok",
    }
    try:
        yield span
    except Exception as exc:
        span["status"] = "error"
        span["error"] = str(exc)
        raise
    finally:
        span["duration_seconds"] = perf_counter() - started
