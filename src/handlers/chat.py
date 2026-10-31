"""Chat (A): answer a research question and ask one scoping question."""

from __future__ import annotations

import json
from typing import Any

from src.dispatch.models import CaseQuestion
from src.handlers.base import CaseHandler, HandlerError
from src.llm.parsing import LLMOutputError, LLMTruncatedError
from src.prompts.templates import chat_messages


class ChatHandler(CaseHandler):
    function = "Chat"

    async def handle(self, question: CaseQuestion) -> dict[str, Any]:
        messages = chat_messages(question)
        last_error = ""
        for attempt in range(2):
            try:
                data = await self.context.llm.complete_json(
                    messages, max_tokens=3000 if attempt == 0 else 4000,
                    temperature=0.2, attempts=1
                )
            except (LLMOutputError, LLMTruncatedError) as exc:
                reason = "输出达到Token上限" if isinstance(exc, LLMTruncatedError) else "输出不是合法JSON对象"
                last_error = reason + "；请压缩回答，返回一个完整JSON对象，answer和clarifying_question均为非空字符串"
                messages = messages + [{"role": "user", "content": last_error}]
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
            last_error = "answer 与 clarifying_question 必须都是非空字符串"
            messages = messages + [
                {"role": "assistant", "content": json.dumps(data, ensure_ascii=False)},
                {
                    "role": "user",
                    "content": f"上一条回复不合格：{last_error}。请只返回规定的 JSON 对象。",
                },
            ]
        raise HandlerError(f"{question.case_id}: {last_error}")
