"""Idea (E): verifiable research plan plus data/expert copy."""

from __future__ import annotations

from typing import Any

from src.dispatch.models import CaseQuestion
from src.handlers.base import CaseHandler
from src.handlers.idea_reporting import run_idea_report
from src.handlers.idea_prompt import idea_messages
from src.submission.paths import idea_markdown


class IdeaHandler(CaseHandler):
    function = "Idea"

    async def handle(self, question: CaseQuestion) -> dict[str, Any]:
        return await run_idea_report(
            self.context, question, idea_messages, idea_markdown
        )
