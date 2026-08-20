"""Isolated OpenAI request worker used to enforce a hard parent-process timeout."""

import json
import sys
from typing import Any

from multi_agent_research_lab.core.config import get_settings


def main() -> int:
    payload: dict[str, Any] = json.loads(sys.stdin.read())
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    from openai import OpenAI

    client = OpenAI(
        api_key=settings.openai_api_key,
        timeout=payload["timeout_seconds"],
        max_retries=payload["max_retries"],
    )
    response = client.chat.completions.create(
        model=payload["model"],
        temperature=0.2,
        messages=[
            {"role": "system", "content": payload["system_prompt"]},
            {"role": "user", "content": payload["user_prompt"]},
        ],
    )
    usage = getattr(response, "usage", None)
    print(
        json.dumps(
            {
                "content": response.choices[0].message.content or "",
                "input_tokens": getattr(usage, "prompt_tokens", None),
                "output_tokens": getattr(usage, "completion_tokens", None),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
