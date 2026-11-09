"""Per-case wall time; nested stages share one monotonic deadline."""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
import time
from typing import Awaitable, TypeVar

CASE_SECONDS = {'A': 540.0, 'B': None, 'C': 900.0, 'D': None, 'E': 900.0}
FUNCTION_PREFIX = {'Chat': 'A', 'Literature_Retrieval': 'B', 'Survey': 'C',
                   'Reproduction': 'D', 'Idea': 'E'}


def case_seconds(category: str | None) -> float | None:
    key = FUNCTION_PREFIX.get(category, category or '')
    return CASE_SECONDS.get(key.split('-', 1)[0], 900.0)


@dataclass(frozen=True)
class CaseDeadline:
    started: float
    end: float


_active: ContextVar[CaseDeadline | None] = ContextVar('research_case_deadline', default=None)
_platform_managed: ContextVar[bool] = ContextVar('platform_managed_timing', default=False)
T = TypeVar('T')


def local_limits_disabled():
    return _platform_managed.get()


def deadline_end(cap: float) -> float:
    now = time.monotonic()
    parent = _active.get()
    return min(now + cap, parent.end) if parent else now + cap


def remaining_seconds(cap: float | None = None, *, reserve: float = 0.0) -> float:
    parent = _active.get()
    left = (parent.end - time.monotonic() - reserve if parent else
            float('inf') if local_limits_disabled() else 1800.0)
    return max(0.0, min(cap, left) if cap is not None else left)


async def within_case(awaitable: Awaitable[T], category: str | None,
                      *, timeout_s: float | None = None) -> T:
    parent = _active.get()
    if case_seconds(category) is None or local_limits_disabled():
        # B/D have no locally invented budget. Only an explicitly supplied
        # platform timeout may bound the HTTP request; nested calls inherit it.
        mode = _platform_managed.set(True)
        token = None
        try:
            if timeout_s is not None:
                now = time.monotonic()
                end = min(now + timeout_s, parent.end if parent else float('inf'))
                token = _active.set(CaseDeadline(now, end))
                return await asyncio.wait_for(awaitable, max(0.0, end - now))
            return await awaitable
        finally:
            if token is not None:
                _active.reset(token)
            _platform_managed.reset(mode)
    started = parent.started if parent else time.monotonic()
    end = min(started + case_seconds(category), parent.end if parent else float('inf'))
    if timeout_s is not None:
        end = min(end, time.monotonic() + timeout_s)
    token = _active.set(CaseDeadline(started, end))
    try:
        # Covers dispatch, PDF/tools, model requests, experiment waiting and saving.
        return await asyncio.wait_for(awaitable, max(0.0, end - time.monotonic()))
    finally:
        _active.reset(token)
