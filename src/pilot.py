"""Top-level AgentHack agent: one /solve call per dispatched case."""

from __future__ import annotations

import logging
from typing import Any

from agenthack import BaseAgent

from src.config import RuntimeConfig
from src.dispatch.loader import FUNCTIONS, DispatchLoader
from src.dispatch.models import CaseQuestion, DispatchRecord
from src.handlers.base import CaseHandler, HandlerContext, HandlerError
from src.handlers.registry import build_handlers
from src.llm.client import GatewayClient
from src.submission.paths import idea_markdown, result_json, survey_markdown
from src.submission.schemas import OUTPUT_FIELDS
from src.submission.writer import ResultWriter, SubmissionError

logger = logging.getLogger(__name__)


class ResearchPilotAgent(BaseAgent):
    """Coordinates dispatch, per-category handling and result persistence."""

    def __init__(
        self,
        config: RuntimeConfig | None = None,
        *,
        llm: Any | None = None,
        library: Any | None = None,
        handlers: dict[str, CaseHandler] | None = None,
    ) -> None:
        self.config = config or RuntimeConfig.from_env()
        self.dispatch = DispatchLoader(self.config)
        self.writer = ResultWriter(self.config)
        self.llm = llm or GatewayClient(self.config)
        if library is None:
            from src.materials.paper_library import PaperLibrary

            library = PaperLibrary(self.config)
        self.library = library
        self.handlers = handlers or build_handlers(
            HandlerContext(config=self.config, llm=self.llm, library=self.library)
        )

    async def solve(self, task_input: dict[str, Any]) -> dict[str, Any]:
        case_id: str | None = None
        try:
            if self._is_direct_question(task_input):
                question = self.dispatch.parse_question(task_input)
                case_id = question.case_id
                self.dispatch.verify_direct_question(question)
                self.library.set_workspace_root(self.config.result_root)
            else:
                record = self.dispatch.load()
                root = self.dispatch.workspace_root()
                self.library.set_workspace_root(root)
                case_id = self._resolve_case_id(task_input, record)
                question = self.dispatch.load_question_by_id(record, case_id)
                self._check_inline(task_input, question)

            handler = self.handlers.get(question.function)
            if handler is None:
                raise HandlerError(f"no handler for {question.function}")
            payload = await handler.handle(question)
            if payload.get("case_id") != case_id:
                raise HandlerError(
                    f"handler returned case_id {payload.get('case_id')!r}, "
                    f"expected {case_id!r}"
                )

            if question.function == "Reproduction":
                self._verify_reproduction(question, payload)
            else:
                if question.function in ("Survey", "Idea"):
                    self._write_report(question, payload)
                self.writer.write_json(
                    result_json(self.config.team, case_id), payload
                )
            return payload
        except Exception:
            logger.exception(
                "case %s failed",
                case_id or task_input.get("case_id") or task_input.get("task_id"),
            )
            raise

    def _write_report(self, question: CaseQuestion, payload: dict[str, Any]) -> None:
        path = payload.get("path")
        markdown = payload.get("markdown")
        if not isinstance(path, str) or not isinstance(markdown, str):
            raise HandlerError(f"{question.case_id}: path/markdown must be strings")
        if question.function == "Survey":
            expected = survey_markdown(self.config.team, question.case_id)
        else:
            expected = idea_markdown(self.config.team, question.case_id)
        if path != expected:
            raise HandlerError(
                f"{question.case_id}: report path {path!r} must be {expected!r}"
            )
        self.writer.write_markdown(path, markdown)

    def _verify_reproduction(
        self,
        question: CaseQuestion,
        payload: dict[str, Any],
    ) -> None:
        relative = result_json(self.config.team, question.case_id)
        try:
            actual = self.writer.read_json(relative)
        except SubmissionError as exc:
            raise HandlerError(
                f"{question.case_id}: platform result not found or unreadable: {exc}"
            ) from exc
        if actual != payload:
            raise HandlerError(
                f"{question.case_id}: platform result file differs from handler payload"
            )
        figures = actual.get("figures")
        if not isinstance(figures, list):
            raise HandlerError(f"{question.case_id}: figures must be a list")
        for figure in figures:
            path = figure.get("path") if isinstance(figure, dict) else None
            if not isinstance(path, str):
                raise HandlerError(f"{question.case_id}: figure path missing")
            try:
                self.writer.verify_exists(path)
            except SubmissionError as exc:
                raise HandlerError(
                    f"{question.case_id}: figure is missing: {path}"
                ) from exc

    @staticmethod
    def _is_direct_question(value: dict[str, Any]) -> bool:
        return (
            isinstance(value, dict)
            and isinstance(value.get("case_id"), str)
            and isinstance(value.get("function"), str)
            and value.get("function") in FUNCTIONS
            and isinstance(value.get("output_schema_ref"), str)
            and value.get("output_schema_ref") == value.get("function")
            and isinstance(value.get("input"), str)
        )

    @staticmethod
    def _resolve_case_id(task_input: dict[str, Any], record: DispatchRecord) -> str:
        sources: list[dict[str, Any]] = [task_input]
        for key in ("question", "input"):
            nested = task_input.get(key)
            if isinstance(nested, dict):
                sources.append(nested)
        for source in sources:
            for key in ("case_id", "task_id"):
                value = source.get(key)
                if isinstance(value, str) and value in record.case_ids:
                    return value
        if len(record.case_ids) == 1:
            return record.case_ids[0]
        raise HandlerError(
            "task_input does not identify a dispatched case_id: "
            + repr(task_input)[:300]
        )

    @staticmethod
    def _check_inline(task_input: dict[str, Any], question: CaseQuestion) -> None:
        known = {
            "function": FUNCTIONS,
            "output_schema_ref": tuple(OUTPUT_FIELDS),
        }
        for key, allowed in known.items():
            value = task_input.get(key)
            if isinstance(value, str) and value in allowed and value != getattr(question, key):
                raise HandlerError(
                    f"inline {key} {value!r} does not match assigned "
                    f"{getattr(question, key)!r}"
                )
        inline_case_id = task_input.get("case_id")
        if inline_case_id is not None and inline_case_id != question.case_id:
            raise HandlerError(
                f"inline case_id {inline_case_id!r} does not match assigned "
                f"{question.case_id!r}"
            )

