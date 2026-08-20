"""Provider-agnostic LLM client with a process-level OpenAI timeout."""

import json
import subprocess
import sys
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
    """OpenAI-compatible completion client with deterministic fallback."""

    def __init__(
        self,
        client: Any | None = None,
        settings: Settings | None = None,
        model: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.model = model or self.settings.openai_model
        self._client = client

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a completion without allowing a stuck provider to block the workflow."""

        if not self.settings.openai_api_key and self._client is None:
            return self._offline_completion(system_prompt, user_prompt)
        if self._client is not None:
            return self._complete_with_injected_client(system_prompt, user_prompt)
        return self._complete_with_openai_worker(system_prompt, user_prompt)

    def _complete_with_injected_client(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        client = self._client
        if client is None:
            return self._offline_completion(system_prompt, user_prompt)
        try:
            response = client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:
            return self._provider_fallback(system_prompt, user_prompt, str(exc))
        return self._normalize_response(response)

    def _complete_with_openai_worker(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        payload = json.dumps(
            {
                "model": self.model,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "timeout_seconds": self.settings.llm_timeout_seconds,
                "max_retries": self.settings.openai_max_retries,
            }
        )
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "multi_agent_research_lab.services.openai_worker"],
                input=payload,
                capture_output=True,
                text=True,
                timeout=self.settings.llm_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return self._provider_fallback(
                system_prompt,
                user_prompt,
                f"OpenAI timed out after {self.settings.llm_timeout_seconds} seconds",
            )
        if completed.returncode != 0:
            error = completed.stderr.strip() or "OpenAI worker failed"
            return self._provider_fallback(system_prompt, user_prompt, error)
        try:
            response = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return self._provider_fallback(
                system_prompt, user_prompt, "Invalid OpenAI worker response"
            )
        return LLMResponse(
            content=str(response.get("content", "")).strip(),
            input_tokens=response.get("input_tokens"),
            output_tokens=response.get("output_tokens"),
            cost_usd=self._estimate_cost(
                response.get("input_tokens"), response.get("output_tokens")
            ),
        )

    def _normalize_response(self, response: Any) -> LLMResponse:
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

    def _provider_fallback(self, system_prompt: str, user_prompt: str, error: str) -> LLMResponse:
        fallback = self._offline_completion(system_prompt, user_prompt)
        return LLMResponse(
            content=fallback.content,
            input_tokens=fallback.input_tokens,
            output_tokens=fallback.output_tokens,
            cost_usd=fallback.cost_usd,
            error=error,
        )

    def _offline_completion(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Provide a deterministic result for tests and offline corpus runs."""

        del system_prompt
        excerpt = " ".join(user_prompt.split())
        content = excerpt[:1200] if excerpt else "No evidence was provided."
        return LLMResponse(
            content=content,
            input_tokens=len(user_prompt.split()),
            output_tokens=len(content.split()),
            cost_usd=0.0,
        )

    @staticmethod
    def _estimate_cost(input_tokens: int | None, output_tokens: int | None) -> float | None:
        if input_tokens is None or output_tokens is None:
            return None
        return (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000
