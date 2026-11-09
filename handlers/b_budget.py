"""B-only 15-minute budget with response time and immutable result checkpoints."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
import time

from src.time_limits import CaseDeadline, _active, _platform_managed, remaining_seconds

TOTAL_SECONDS = 900.0
DELIVERY_RESERVE = 20.0
CLEANUP_SECONDS = 10.0
_current = ContextVar('b_work_budget', default=None)


class BDeadlineReached(TimeoutError):
    pass


class BWorkBudget:
    def __init__(self):
        self.started = time.monotonic()
        available = min(TOTAL_SECONDS, remaining_seconds())
        self.end = self.started + available
        self.work_end = self.end - min(DELIVERY_RESERVE, available / 4)
        self.papers = []
        self.stage = 'dispatch'
        self.expired = False
        self.workers = set()
        self.events = []
        self.checkpoint = None

    def remaining(self):
        return max(0.0, self.work_end - time.monotonic())

    def update(self, papers, stage):
        self.papers = [dict(p) for p in papers]
        self.stage = stage
        if self.checkpoint is not None:
            self.checkpoint(stage)

    @contextmanager
    def scope(self):
        current = _current.set(self)
        active = _active.set(CaseDeadline(self.started, self.work_end))
        # Enable the actual B deadline for SDK/PDF helpers, without changing D.
        mode = _platform_managed.set(False)
        try:
            yield self
        finally:
            _platform_managed.reset(mode)
            _active.reset(active)
            _current.reset(current)

    async def run(self, operation, stage):
        self.stage = stage
        left = self.remaining()
        if left <= 0:
            if hasattr(operation, 'close'):
                operation.close()
            self.expired = True
            raise BDeadlineReached(stage)
        try:
            return await asyncio.wait_for(operation, left)
        except asyncio.TimeoutError:
            if self.remaining() <= .05:
                self.expired = True
                raise BDeadlineReached(stage) from None
            raise

    async def local(self, function, *args, **kwargs):
        # Shield a worker so cancellation does not lose its cleanup/trace end.
        task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        self.workers.add(task)
        def finished(done):
            self.workers.discard(done)
            if not done.cancelled():
                done.exception()  # consume a late, already-diagnosed worker error
        task.add_done_callback(finished)
        return await self.run(asyncio.shield(task), self.stage)

    async def drain(self):
        if not self.workers:
            return
        remaining = max(0.0, self.end - time.monotonic())
        done, pending = await asyncio.wait(tuple(self.workers),
            timeout=min(CLEANUP_SECONDS, remaining / 2))
        self.events.append({'stage': 'file_worker_cleanup', 'completed': len(done),
                            'pending': len(pending)})


def current_budget():
    return _current.get()


@contextmanager
def begin_budget():
    budget = current_budget() or BWorkBudget()
    token = _current.set(budget)
    try:
        yield budget
    finally:
        _current.reset(token)
