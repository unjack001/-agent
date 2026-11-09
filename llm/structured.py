"""A/B JSON calls: one owner for retries, with time left to recover."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
import time
from typing import Any

from src.llm.parsing import LLMOutputError
from src.time_limits import remaining_seconds, local_limits_disabled

_request_end: ContextVar[float | None] = ContextVar('structured_request_end', default=None)
_request_unbounded: ContextVar[bool] = ContextVar('structured_request_unbounded', default=False)
JSON_RULES = (
    '输出协议：只输出一个完整JSON对象，严格使用题目指定的字段和类型。'
    '使用双引号；字符串中的双引号、反斜杠和换行必须按JSON规则转义；'
    '不得使用尾逗号、注释、NaN、重复字段、代码围栏或无效Unicode。'
    '内部完成核对，仅输出最终结果，不输出思考过程。'
    '如需纠正，重写完整对象，不续写上一次残缺片段，不添加新的顶层字段。'
)


def request_seconds(cap: float) -> float | None:
    if _request_unbounded.get() or local_limits_disabled():
        return None
    end = _request_end.get()
    return min(cap, max(0.0, end - time.monotonic())) if end is not None else cap


def response_error(message: str, *, response_body=None):
    # Callers opting into a retry window treat empty replies as correctable output.
    from src.llm.parsing import LLMError
    error = (LLMOutputError if _request_end.get() is not None or _request_unbounded.get() else LLMError)(message)
    error.response_body = response_body
    return error


@contextmanager
def request_window(timeout: float | None):
    if timeout is None:
        mode = _request_unbounded.set(True)
        token = _request_end.set(None)
        try:
            yield
        finally:
            _request_end.reset(token)
            _request_unbounded.reset(mode)
        return
    end = time.monotonic() + timeout
    parent = _request_end.get()
    token = _request_end.set(min(end, parent) if parent is not None else end)
    try:
        yield
    finally:
        _request_end.reset(token)


def validate_json_text(value: Any) -> None:
    """Reject strings that would fail when the final result is saved as UTF-8."""
    try:
        if isinstance(value, str):
            value.encode('utf-8')
        elif isinstance(value, dict):
            for key, item in value.items():
                validate_json_text(key)
                validate_json_text(item)
        elif isinstance(value, list):
            for item in value:
                validate_json_text(item)
    except (UnicodeEncodeError, RecursionError):
        raise LLMOutputError('JSON contains invalid UTF-8 Unicode or excessive nesting') from None


async def call_json(llm, messages, *, timeout: float | None, max_tokens: int,
                    temperature: float = 0.0) -> dict[str, Any]:
    """Make one request; handlers retry only within the original case deadline."""
    if timeout is not None:
        timeout = min(timeout, remaining_seconds(timeout))
        if timeout <= 0:
            raise asyncio.TimeoutError('case budget exhausted before model request')
    conversation = [dict(message) for message in messages]
    if conversation and conversation[0]['role'] == 'system':
        conversation[0]['content'] += '\n' + JSON_RULES
    else:
        conversation.insert(0, {'role': 'system', 'content': JSON_RULES})
    token = _request_end.set(time.monotonic() + timeout if timeout is not None else None)
    mode = _request_unbounded.set(timeout is None)
    try:
        operation = llm.complete_json(conversation, max_tokens=max_tokens, temperature=temperature, attempts=1)
        data = await operation if timeout is None else await asyncio.wait_for(operation, timeout)
        if not isinstance(data, dict):
            raise LLMOutputError('model reply must be a JSON object')
        validate_json_text(data)
        return data
    finally:
        _request_end.reset(token)
        _request_unbounded.reset(mode)


async def retry_pause(attempt: int, left: float) -> None:
    # Only transient transport failures use backoff; output repairs can run now.
    await asyncio.sleep(min(0.5 * (attempt + 1), max(0.0, left)))
