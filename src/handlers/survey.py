"""Survey (C): cited Markdown review plus data/expert copy."""

from __future__ import annotations

from typing import Any

from src.dispatch.models import CaseQuestion
from src.handlers.base import CaseHandler
from src.handlers.survey_reporting import run_survey_report
from src.handlers.survey_prompt import survey_messages
from src.submission.paths import survey_markdown


class SurveyHandler(CaseHandler):
    function = "Survey"

    async def handle(self, question: CaseQuestion) -> dict[str, Any]:
        return await run_survey_report(
            self.context, question, survey_messages, survey_markdown
        )
