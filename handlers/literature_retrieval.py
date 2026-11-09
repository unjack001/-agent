"""Literature_Retrieval (B): rank local PDFs with filename-faithful reasons."""

from __future__ import annotations

import logging
import time
from typing import Any
from uuid import uuid4

from src.dispatch.models import CaseQuestion
from src.handlers.base import CaseHandler
from src.materials.access import CasePolicy
from src.submission.writer import ResultWriter
from src.handlers.b_budget import begin_budget, BDeadlineReached

logger = logging.getLogger(__name__)


class LiteratureRetrievalHandler(CaseHandler):
    function = "Literature_Retrieval"

    async def handle(self, question: CaseQuestion) -> dict[str, Any]:
        audit_prefix = (f'{CasePolicy(self.context.config, question.case_id).work_prefix}'
                        f'/retrieval-validation/{uuid4().hex}')
        with begin_budget() as budget, budget.scope():
            def checkpoint(stage):
                self._record(question, audit_prefix, 'checkpoint', {
                    'last_stage': stage, 'papers': budget.papers,
                    'remaining_work_s': budget.remaining()})
            budget.checkpoint = checkpoint
            try:
                result = await budget.run(self._handle(question, audit_prefix, budget), 'B_processing')
            except BDeadlineReached:
                result = {'case_id': question.case_id, 'papers': budget.papers}
            finally:
                await budget.drain()
            self._record(question, audit_prefix, 'completion', {
                'deadline_reached': budget.expired, 'last_stage': budget.stage,
                'papers': result['papers'], 'cleanup': budget.events,
                'elapsed_s': time.monotonic() - budget.started})
            return result

    async def _handle(self, question, audit_prefix, budget):
        from src.handlers.b_svg_pipeline import Pipeline
        pipeline = Pipeline(self.context, question, budget,
            lambda stage, payload: self._record(question, audit_prefix, stage, payload))
        try:
            return await pipeline.run()
        finally:
            pipeline.record('pipeline', pipeline.diagnostics())

    def _record(
        self, question: CaseQuestion, prefix: str, stage: str, payload: dict[str, Any],
    ) -> None:
        """Save local validation evidence without prompts, headers or settings."""
        try:
            relative = f"{prefix}/{stage}.json"
            CasePolicy(self.context.config, question.case_id).work_path(relative, for_write=True)
            ResultWriter(self.context.config).write_json(relative, {
                "case_id": question.case_id, "stage": stage, **payload,
            })
            logger.info("B %s %s diagnostic: %s", question.case_id, stage, relative)
        except Exception as exc:
            # Optional diagnostics must never replace the actual task outcome,
            # including when /output itself is not writable.
            logger.warning("B %s diagnostics unavailable (%s)",
                           question.case_id, type(exc).__name__)

    @staticmethod
    def _normalise(
        entries: list[Any], valid: dict[str, str], *, issues: list[str] | None = None,
    ) -> list[dict[str, str]]:
        results: list[dict[str, str]] = []
        if issues is None:
            issues = []
        seen: set[str] = set()
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                issues.append(f"papers[{index}] 必须是含filename和reason的对象")
                continue
            filename = entry.get("filename")
            reason = entry.get("reason")
            if not isinstance(filename, str):
                issues.append(f"papers[{index}].filename 必须是字符串")
                continue
            key = filename.strip().strip("\"'")
            basename = key.replace("\\", "/").rsplit("/", 1)[-1]
            actual = valid.get(key.casefold()) or valid.get(basename.casefold())
            if actual is None:
                issues.append(f"papers[{index}].filename 不在候选目录：{filename[:160]!r}")
                continue
            if actual.casefold() in seen:
                issues.append(f"papers[{index}] 文件名重复")
                continue
            if not isinstance(reason, str) or not reason.strip():
                issues.append(f"papers[{index}].reason 必须是非空字符串")
                continue
            try:
                reason.encode("utf-8")
            except UnicodeEncodeError:
                issues.append(f"papers[{index}].reason 含无效Unicode，必须可编码为UTF-8")
                continue
            seen.add(actual.casefold())
            results.append({"filename": actual, "reason": reason.strip()})
        return results
