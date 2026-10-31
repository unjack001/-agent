"""Literature_Retrieval (B): rank local PDFs with filename-faithful reasons."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from uuid import uuid4

from src.dispatch.models import CaseQuestion
from src.handlers.base import CaseHandler, HandlerError
from src.llm.parsing import LLMError, LLMOutputError, LLMTruncatedError
from src.materials.access import CasePolicy
from src.materials.case_materials import CaseMaterials
from src.prompts.templates import literature_messages
from src.submission.writer import ResultWriter

logger = logging.getLogger(__name__)


class LiteratureRetrievalHandler(CaseHandler):
    function = "Literature_Retrieval"

    async def handle(self, question: CaseQuestion) -> dict[str, Any]:
        audit_prefix = (
            f"{CasePolicy(self.context.config, question.case_id).work_prefix}"
            f"/retrieval-validation/{uuid4().hex}"
        )
        try:
            papers = await asyncio.to_thread(CaseMaterials(self.context, question).resolve)
        except Exception as exc:
            self._record(question, audit_prefix, "materials", {
                "materials": question.materials, "exception_type": type(exc).__name__,
            })
            logger.error("B %s material resolution failed (%s); diagnostic_work=%s",
                         question.case_id, type(exc).__name__, audit_prefix)
            raise
        if not papers:
            self._record(question, audit_prefix, "materials", {
                "materials": question.materials, "candidate_filenames": [],
            })
            return {"case_id": question.case_id, "papers": []}

        try:
            catalog = await asyncio.to_thread(
                self.context.library.catalog,
                question.input,
                papers=papers,
                max_papers=150,
            )
        except Exception as exc:
            self._record(question, audit_prefix, "catalog", {
                "candidate_filenames": [paper.filename for paper in papers],
                "exception_type": type(exc).__name__,
            })
            raise
        self._record(question, audit_prefix, "catalog", {
            "candidate_filenames": [entry.filename for entry in catalog],
        })
        if not catalog:
            return {"case_id": question.case_id, "papers": []}

        valid = {entry.filename.casefold(): entry.filename for entry in catalog}
        messages = literature_messages(question, catalog)
        last_issue = ""
        for attempt in range(2):
            try:
                data = await self.context.llm.complete_json(
                    messages, max_tokens=3000, temperature=0.0, attempts=1
                )
            except (LLMOutputError, LLMTruncatedError) as exc:
                self._record(question, audit_prefix, f"attempt-{attempt + 1}", {
                    "validation_accepted": False, "exception_type": type(exc).__name__,
                    "issues": ["模型回复不是完整合法 JSON"],
                })
                if attempt:
                    raise
                last_issue = "上一回复未能生成完整合法JSON，请压缩理由并重写完整对象。"
                messages = messages + [{"role": "user", "content": last_issue}]
                continue
            except LLMError as exc:
                self._record(question, audit_prefix, f"attempt-{attempt + 1}", {
                    "validation_accepted": False, "exception_type": type(exc).__name__,
                })
                raise
            entries = data.get("papers") if isinstance(data, dict) else None
            if not isinstance(entries, list):
                last_issue = "papers 必须是数组。请只返回规定的 JSON 对象。"
                self._record(question, audit_prefix, f"attempt-{attempt + 1}", {
                    "validation_accepted": False, "issues": [last_issue],
                    "papers": entries,
                })
                messages = messages + [
                    {"role": "assistant", "content": json.dumps(data, ensure_ascii=True)},
                    {
                        "role": "user",
                        "content": last_issue,
                    },
                ]
                continue

            issues: list[str] = []
            results = self._normalise(entries, valid, issues=issues)
            self._record(question, audit_prefix, f"attempt-{attempt + 1}", {
                "validation_accepted": bool(results) or not entries,
                "papers": entries, "accepted_papers": results, "issues": issues,
            })
            if results or not entries:
                return {"case_id": question.case_id, "papers": results}

            last_issue = (
                "；".join(issues[:12])
                + "。请从初始候选目录逐字复制完整filename，不要用标题、编号或省略号代替；"
                "每篇提供非空的reason。只返回规定的papers数组，不要编造匹配理由。"
            )
            messages = messages + [
                {
                    "role": "assistant",
                    "content": json.dumps({"papers": entries}, ensure_ascii=True),
                },
                {
                    "role": "user",
                    "content": last_issue,
                },
            ]
        raise HandlerError(
            f"{question.case_id}: retrieval result failed validation: {last_issue}"
            f"; diagnostic_work={audit_prefix}"
        )

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
