"""Per-case wall time; nested stages share one monotonic deadline."""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
import time
from typing import Awaitable, TypeVar

CASE_SECONDS = {'A': 540.0, 'B': 540.0, 'C': 900.0, 'D': 1800.0, 'E': 900.0}
FUNCTION_PREFIX = {'Chat': 'A', 'Literature_Retrieval': 'B', 'Survey': 'C',
                   'Reproduction': 'D', 'Idea': 'E'}
MODEL_REQUEST_MAX_S = 900.0
D_EXECUTION_MAX_S = 840.0  # Platform executor, not a separate additional allowance.
D_RPC_MAX_S = 870.0        # Controlled by the currently mounted platform client.
D_CLIENT_MAX_S = 880.0
SAVE_RESERVE_S = 5.0


def case_seconds(category: str | None) -> float:
    key = FUNCTION_PREFIX.get(category, category or '')
    return CASE_SECONDS.get(key.split('-', 1)[0], 900.0)


@dataclass(frozen=True)
class CaseDeadline:
    started: float
    end: float


_active: ContextVar[CaseDeadline | None] = ContextVar('research_case_deadline', default=None)
T = TypeVar('T')


def deadline_end(cap: float) -> float:
    now = time.monotonic()
    parent = _active.get()
    return min(now + cap, parent.end) if parent else now + cap


def remaining_seconds(cap: float = MODEL_REQUEST_MAX_S, *, reserve: float = 0.0) -> float:
    parent = _active.get()
    return max(0.0, min(cap, parent.end - time.monotonic() - reserve)) if parent else cap


async def within_case(awaitable: Awaitable[T], category: str | None,
                      *, timeout_s: float | None = None) -> T:
    parent = _active.get()
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
