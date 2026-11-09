"""Chat (A): answer a research question and ask one scoping question."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from src.dispatch.models import CaseQuestion
from src.handlers.base import CaseHandler
from src.llm.parsing import LLMError, LLMOutputError, LLMTruncatedError
from src.llm.structured import call_json, retry_pause
from src.prompts.templates import chat_messages
from src.time_limits import deadline_end, case_seconds


class ChatHandler(CaseHandler):
    function = "Chat"

    async def handle(self, question: CaseQuestion) -> dict[str, Any]:
        messages = chat_messages(question)
        deadline = deadline_end(case_seconds(question.function))
        last_error = ""
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1
            try:
                data = await call_json(
                    self.context.llm, messages,
                    timeout=max(0.0, deadline - time.monotonic()),
                    max_tokens=3000 if attempt == 1 else 4000, temperature=0.0,
                )
            except (LLMOutputError, LLMTruncatedError) as exc:
                reason = "输出达到Token上限" if isinstance(exc, LLMTruncatedError) else "输出不是合法JSON对象"
                last_error = reason + "；请压缩回答，返回一个完整JSON对象，answer和clarifying_question均为非空字符串"
                messages = messages + [{"role": "user", "content": last_error}]
                continue
            except (asyncio.TimeoutError, LLMError) as exc:
                if isinstance(exc, LLMError) and not exc.retryable:
                    raise
                left = deadline - time.monotonic()
                if left <= 0:
                    raise
                await retry_pause(attempt, left)
                continue
            answer = data.get("answer")
            clarifying = data.get("clarifying_question")
            if (
                isinstance(answer, str)
                and answer.strip()
                and isinstance(clarifying, str)
                and clarifying.strip()
            ):
                return {
                    "case_id": question.case_id,
                    "answer": answer.strip(),
                    "clarifying_question": clarifying.strip(),
                }
            else:
                last_error = "answer 与 clarifying_question 必须都是非空字符串"
            messages = messages + [
                {"role": "assistant", "content": json.dumps(data, ensure_ascii=False)},
                {
                    "role": "user",
                    "content": f"上一条回复不合格：{last_error}。请只返回规定的 JSON 对象。",
                },
            ]
        raise asyncio.TimeoutError(f"{question.case_id}: case budget exhausted; {last_error}")
