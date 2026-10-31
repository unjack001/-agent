"""OpenAI-compatible chat client bound to PLATFORM_LLM_URL."""

from __future__ import annotations

import os
from typing import Any

from src.config import RuntimeConfig
from src.llm.parsing import LLMError, LLMOutputError, LLMTruncatedError, parse_json_object


class GatewayClient:
    """Lazy client: AgentRuntime installs its interceptor before the first call."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.config.llm_url or not self.config.llm_api_key or not self.config.llm_model:
            raise LLMError(
                "gateway URL, API key and model must be set via PLATFORM_LLM_URL/"
                "PLATFORM_API_KEY/AGENT_MODEL_NAME or CHALLENGE_LLM_BASE_URL/"
                "CHALLENGE_LLM_API_KEY/CHALLENGE_LLM_MODEL"
            )
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            base_url=self.config.llm_url,
            api_key=self.config.llm_api_key,
            timeout=self.config.llm_timeout_s,
            max_retries=0,
        )
        return self._client

    async def complete_with_body(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.2,
    ) -> tuple[str, dict[str, Any]]:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.config.llm_model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        # The challenge gateway requires this header in addition to the SDK's
        # telemetry correlation headers. Read it per request, including repairs.
        session_id = os.getenv("CHALLENGE_SESSION_ID")
        if session_id:
            kwargs["extra_headers"] = {"X-Session-ID": session_id}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        try:
            response = await client.chat.completions.create(**kwargs)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError("gateway call failed") from exc

        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMError("gateway returned no choices")
        choice = choices[0]
        try:
            body = response.model_dump()
        except Exception as exc:
            raise LLMError("gateway response cannot be serialized") from exc
        if not isinstance(body, dict):
            raise LLMError("gateway response body is not a JSON object")
        text = getattr(getattr(choice, "message", None), "content", None) or ""
        if getattr(choice, "finish_reason", None) == "length":
            raise LLMTruncatedError("gateway response was truncated by the token limit", response_body=body)
        if getattr(choice, "finish_reason", None) == "content_filter":
            raise LLMError("gateway response was filtered")
        if getattr(getattr(choice, "message", None), "tool_calls", None):
            raise LLMError("unexpected tool calls in a fixed-workflow response")
        if not isinstance(text, str) or not text.strip():
            raise LLMError("gateway returned empty content")
        return text.strip(), body

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.2,
    ) -> str:
        text, _ = await self.complete_with_body(
            messages, max_tokens=max_tokens, temperature=temperature
        )
        return text

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        attempts: int = 1,
    ) -> dict[str, Any]:
        conversation = list(messages)
        last_text = ""
        for _ in range(max(1, attempts)):
            last_text = await self.complete(
                conversation, max_tokens=max_tokens, temperature=temperature
            )
            try:
                return parse_json_object(last_text)
            except LLMOutputError as exc:
                conversation = conversation + [
                    {"role": "assistant", "content": last_text},
                    {
                        "role": "user",
                        "content": (
                            f"上一条回复不是合法 JSON（{exc}）。"
                            "只返回一个 JSON 对象，不要代码围栏或额外解释。"
                        ),
                    },
                ]
        raise LLMOutputError("gateway did not return a valid JSON object")
