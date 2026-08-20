"""Optional LangSmith exporter for workflow and agent spans."""

from importlib import import_module
from typing import Any

from multi_agent_research_lab.core.config import Settings


class OptionalLangSmithTracer:
    """Send spans to LangSmith when configured without making it mandatory."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client: Any | None = None
        self.root: Any | None = None
        self.url: str | None = None
        self.error: str | None = None
        if not settings.langsmith_api_key or not settings.langsmith_enabled:
            return
        try:
            client_type = import_module("langsmith").Client
            self.client = client_type(
                api_key=settings.langsmith_api_key,
                timeout_ms=max(1000, settings.timeout_seconds * 1000),
            )
        except Exception as exc:
            self.error = str(exc)

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def start(self, query: str) -> None:
        if not self.enabled:
            return
        try:
            run_type = import_module("langsmith.run_trees").RunTree
            self.root = run_type(
                name="multi-agent-research-workflow",
                run_type="chain",
                project_name=self.settings.langsmith_project,
                inputs={"query": query},
                client=self.client,
            )
            self.root.post()
        except Exception as exc:
            self.error = str(exc)
            self.root = None

    def start_child(self, name: str, inputs: dict[str, Any]) -> Any | None:
        if self.root is None:
            return None
        try:
            child = self.root.create_child(name=name, run_type="chain", inputs=inputs)
            child.post()
            return child
        except Exception as exc:
            self.error = str(exc)
            return None

    def finish_child(
        self, child: Any | None, outputs: dict[str, Any] | None = None, error: str | None = None
    ) -> None:
        if child is None:
            return
        try:
            child.end(outputs=outputs, error=error)
            child.patch()
        except Exception as exc:
            self.error = str(exc)

    def finish(self, outputs: dict[str, Any], error: str | None = None) -> None:
        if self.root is None:
            return
        try:
            self.root.end(outputs=outputs, error=error)
            self.root.patch()
            self.url = self.root.get_url()
        except Exception as exc:
            self.error = str(exc)
