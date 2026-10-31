"""Survey (C): cited Markdown review plus data/expert copy."""

from __future__ import annotations

from typing import Any

from src.dispatch.models import CaseQuestion
from src.handlers.base import CaseHandler
from src.handlers.reporting import run_report_handler
from src.prompts.templates import survey_messages
from src.submission.paths import survey_markdown


class SurveyHandler(CaseHandler):
    function = "Survey"

    async def handle(self, question: CaseQuestion) -> dict[str, Any]:
        return await run_report_handler(
            self.context, question, survey_messages, survey_markdown
        )
