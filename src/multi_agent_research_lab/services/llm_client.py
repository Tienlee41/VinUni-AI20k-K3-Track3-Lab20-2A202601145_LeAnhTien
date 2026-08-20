"""Small provider-agnostic LLM client with an offline fallback.

The SDK is imported lazily so the repository remains usable for local tests without
installing the optional ``llm`` extra.  The fallback is deliberately deterministic;
it makes the workflow debuggable when a key is absent or a provider is unavailable.
"""

from dataclasses import dataclass
from typing import Any

from multi_agent_research_lab.core.config import Settings, get_settings


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    error: str | None = None


class LLMClient:
    """Provider-agnostic client for OpenAI-compatible chat completions."""

    def __init__(
        self,
        client: Any | None = None,
        settings: Settings | None = None,
        model: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.model = model or self.settings.openai_model
        self._client = client

    @property
    def client(self) -> Any | None:
        if self._client is None and self.settings.openai_api_key:
            try:
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=self.settings.openai_api_key,
                    timeout=self.settings.timeout_seconds,
                    max_retries=2,
                )
            except ImportError:
                self._client = None
        return self._client

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a completion and normalize provider usage metadata."""

        if self.client is None:
            return self._offline_completion(system_prompt, user_prompt)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:
            # Preserve end-to-end availability while exposing the provider failure to tracing.
            fallback = self._offline_completion(system_prompt, user_prompt)
            return LLMResponse(
                content=fallback.content,
                input_tokens=fallback.input_tokens,
                output_tokens=fallback.output_tokens,
                error=str(exc),
            )
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
        return LLMResponse(
            content=str(choice.message.content or "").strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=self._estimate_cost(input_tokens, output_tokens),
        )

    def _offline_completion(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Provide a useful deterministic result for tests and offline corpus runs."""

        del system_prompt
        excerpt = " ".join(user_prompt.split())
        content = excerpt[:1200] if excerpt else "No evidence was provided."
        return LLMResponse(
            content=content,
            input_tokens=len(user_prompt.split()),
            output_tokens=len(content.split()),
            cost_usd=0.0,
        )

    def _estimate_cost(self, input_tokens: int | None, output_tokens: int | None) -> float | None:
        if input_tokens is None or output_tokens is None:
            return None
        # Approximate public gpt-4o-mini rates; callers should treat this as an estimate.
        return (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000
