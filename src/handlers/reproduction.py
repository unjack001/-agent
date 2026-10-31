"""Reproduction (D): platform-client execution, metrics and real figures."""

from __future__ import annotations

from typing import Any

from src.dispatch.models import CaseQuestion
from src.handlers.base import CaseHandler
from src.reproduction.runner import run_reproduction


class ReproductionHandler(CaseHandler):
    function = "Reproduction"

    async def handle(self, question: CaseQuestion) -> dict[str, Any]:
        return await run_reproduction(self.context, question)
