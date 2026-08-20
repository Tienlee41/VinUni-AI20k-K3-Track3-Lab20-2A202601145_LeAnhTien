"""LangGraph orchestration for the research agents."""

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, TypedDict, cast

from multi_agent_research_lab.agents import (
    AnalystAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.langsmith import OptionalLangSmithTracer
from multi_agent_research_lab.observability.tracing import trace_span


class WorkflowValues(TypedDict, total=False):
    """Serializable values passed between LangGraph nodes."""

    request: dict[str, Any]
    iteration: int
    max_iterations: int | None
    route_history: list[str]
    sources: list[dict[str, Any]]
    research_notes: str | None
    analysis_notes: str | None
    final_answer: str | None
    agent_results: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    errors: list[str]


class MultiAgentWorkflow:
    """Build and run Supervisor → Researcher → Analyst → Writer."""

    def __init__(
        self,
        settings: Settings | None = None,
        supervisor: SupervisorAgent | None = None,
        researcher: ResearcherAgent | None = None,
        analyst: AnalystAgent | None = None,
        writer: WriterAgent | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.supervisor = supervisor or SupervisorAgent(self.settings.max_iterations)
        self.researcher = researcher or ResearcherAgent()
        self.analyst = analyst or AnalystAgent()
        self.writer = writer or WriterAgent()
        self._compiled: Any | None = None
        self._active_tracer: OptionalLangSmithTracer | None = None

    def build(self) -> Any:
        """Create a compiled LangGraph graph with conditional routing."""

        if self._compiled is not None:
            return self._compiled
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError:
            self._compiled = _FallbackGraph(self)
            return self._compiled

        graph = StateGraph(WorkflowValues)
        graph.add_node("supervisor", self._node(self.supervisor))
        graph.add_node("researcher", self._node(self.researcher))
        graph.add_node("analyst", self._node(self.analyst))
        graph.add_node("writer", self._node(self.writer))
        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            lambda values: values.get("route_history", ["done"])[-1],
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "done": END,
            },
        )
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", END)
        self._compiled = graph.compile()
        return self._compiled

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph with timeout and iteration guards."""

        state.max_iterations = state.max_iterations or self.settings.max_iterations
        state.add_trace_event("workflow.start", {"query": state.request.query})
        self._active_tracer = OptionalLangSmithTracer(self.settings)
        self._active_tracer.start(state.request.query)
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="research-workflow") as executor:
            future = executor.submit(self.build().invoke, state.model_dump())
            try:
                values = future.result(timeout=self.settings.timeout_seconds)
            except FutureTimeoutError:
                state.errors.append(
                    f"Workflow timed out after {self.settings.timeout_seconds} seconds"
                )
                state.add_trace_event(
                    "workflow.timeout", {"timeout_seconds": self.settings.timeout_seconds}
                )
                self._active_tracer.finish({}, error=state.errors[-1])
                return state
            except Exception as exc:
                state.errors.append(f"Workflow failed: {exc}")
                state.add_trace_event("workflow.error", {"error": str(exc)})
                self._active_tracer.finish({}, error=str(exc))
                return state
        result = ResearchState.model_validate(values)
        result.add_trace_event("workflow.complete", {"iterations": result.iteration})
        self._active_tracer.finish({"route_history": result.route_history})
        if self._active_tracer.url:
            result.add_trace_event("langsmith.run", {"url": self._active_tracer.url})
        if self._active_tracer.error:
            result.add_trace_event("langsmith.error", {"error": self._active_tracer.error})
        return result

    def _node(self, agent: Any) -> Any:
        def run(values: WorkflowValues) -> dict[str, Any]:
            current = ResearchState.model_validate(values)
            child = (
                self._active_tracer.start_child(agent.name, {"iteration": current.iteration})
                if self._active_tracer
                else None
            )
            try:
                with trace_span(f"agent.{agent.name}", {"agent": agent.name}) as span:
                    result = agent.run(current)
            except Exception as exc:
                if self._active_tracer:
                    self._active_tracer.finish_child(child, error=str(exc))
                raise
            if self._active_tracer:
                self._active_tracer.finish_child(child, {"iteration": result.iteration})
            result.add_trace_event("agent.span", span)
            return cast(dict[str, Any], result.model_dump())

        return run


class _FallbackGraph:
    """Compatibility graph used when the optional LangGraph package is absent."""

    def __init__(self, workflow: MultiAgentWorkflow) -> None:
        self.workflow = workflow

    def invoke(self, values: dict[str, Any]) -> dict[str, Any]:
        state = ResearchState.model_validate(values)
        while state.iteration <= (state.max_iterations or self.workflow.settings.max_iterations):
            state = self.workflow.supervisor.run(state)
            route = state.route_history[-1]
            if route == "researcher":
                state = self.workflow.researcher.run(state)
            elif route == "analyst":
                state = self.workflow.analyst.run(state)
            elif route == "writer":
                state = self.workflow.writer.run(state)
                break
            else:
                break
        return state.model_dump()
