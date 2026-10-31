"""Validation and assembly for cited Markdown reports."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Callable
from uuid import uuid4

from src.dispatch.models import CaseQuestion
from src.handlers.base import HandlerContext, HandlerError
from src.handlers.idea_validation import idea_content_checklist, validate_idea_body
from src.handlers.report_citations import cited_numbers as cited_numbers
from src.handlers.report_citations import validate_citations
from src.llm.parsing import LLMTruncatedError
from src.materials.access import CasePolicy
from src.materials.case_materials import CaseMaterials
from src.materials.models import PaperDigest, PaperRef, has_evidence_text
from src.submission.writer import ResultWriter, SubmissionError

logger = logging.getLogger(__name__)

REFERENCE_HEADING_RE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:参考文献|references|bibliography)\s*[:：]?\s*$"
)


def clean_markdown(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except ValueError:
            data = None
        if isinstance(data, dict) and isinstance(data.get("markdown"), str):
            text = data["markdown"].strip()
    return text


def strip_references(body: str) -> str:
    # Keep original offsets: code samples may contain a reference heading but
    # must not truncate the actual report that follows the fence.
    offset = 0
    fence = ""
    fence_length = 0
    for line in body.splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line.rstrip("\r\n"))
        if marker:
            if not fence:
                fence, fence_length = marker[1][0], len(marker[1])
            elif marker[1][0] == fence and len(marker[1]) >= fence_length and not marker[2].strip():
                fence = ""
        elif not fence and REFERENCE_HEADING_RE.fullmatch(line.strip()):
            return body[:offset].rstrip()
        offset += len(line)
    return body.rstrip()


def assemble_report(body: str, digests: list[PaperDigest], cited: list[int]) -> str:
    entries = [
        f"[{index}] {digests[index - 1].title}，{digests[index - 1].filename}"
        for index in cited
    ]
    return body.rstrip() + "\n\n## 参考文献\n\n" + "\n".join(entries) + "\n"


def validate_report_body(
    body: str,
    digests: list[PaperDigest],
    *,
    min_chars: int = 800,
    min_citations: int = 5,
) -> tuple[list[int], list[str]]:
    issues: list[str] = []
    if len(body.strip()) < min_chars:
        issues.append(f"正文至少需要 {min_chars} 个字符")
    cited, citation_issues = validate_citations(body, digests)
    issues.extend(citation_issues)
    required = min(min_citations, len(digests))
    if len(cited) < required:
        issues.append(f"正文只引用了 {len(cited)} 篇，至少需要引用 {required} 篇指定论文")

    return cited, issues


def validate_paper_evidence(papers: list[PaperRef], digests: list[PaperDigest]) -> None:
    """Require real extracted text for every specified paper, without renumbering it."""
    expected = [paper.filename for paper in papers]
    actual = [digest.filename for digest in digests]
    if len(set(actual)) != len(actual) or set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        raise HandlerError(
            f"paper digest/material mismatch: missing={missing}, unexpected={unexpected}"
        )
    unreadable = [
        digest.filename
        for digest in digests
        if not any(
            excerpt.page > 0
            and has_evidence_text(
                excerpt.text, filename=digest.filename, title=digest.title
            )
            for excerpt in digest.excerpts
        )
    ]
    if unreadable:
        raise HandlerError(
            "no paper text available for specified PDF(s): "
            + ", ".join(unreadable)
            + "; provide readable text or OCR before generating a report"
        )


def _record_attempt(
    context: HandlerContext, question: CaseQuestion, prefix: str | None,
    attempt: int, raw: str | None, body: str | None, issues: list[str],
    digests: list[PaperDigest],
) -> None:
    """Keep rejected drafts in this case's work area, never as submissions.

    Optional diagnostics must not turn a valid answer into a failure. The same
    path and symlink checks still apply; no prompts, environment or keys are saved.
    """
    if prefix is None:
        return
    relative = f"{prefix}/attempt-{attempt}.json"
    try:
        CasePolicy(context.config, question.case_id).work_path(relative, for_write=True)
        ResultWriter(context.config).write_json(relative, {
            "case_id": question.case_id, "function": question.function,
            "attempt": attempt, "validation_accepted": not issues,
            "issues": issues, "raw_markdown": raw, "body": body,
            "sources": [{"filename": d.filename,
                         "excerpt_pages": [e.page for e in d.excerpts]} for d in digests],
        })
    except (OSError, ValueError, SubmissionError) as exc:
        logger.warning("report diagnostics unavailable for %s (%s)",
                       question.case_id, type(exc).__name__)


async def run_report_handler(
    context: HandlerContext,
    question: CaseQuestion,
    messages_builder: Callable[[CaseQuestion, list[PaperDigest]], list[dict[str, str]]],
    path_builder: Callable[[str, str], str],
    *,
    max_tokens: int = 6000,
    attempts: int = 2,
) -> dict[str, Any]:
    # Resolve once through the same per-case capability boundary used by the
    # other fixed handlers, before the broad PDF adapter sees any path.
    papers = await asyncio.to_thread(CaseMaterials(context, question).resolve)
    if not papers:
        raise HandlerError(f"{question.case_id}: no materials resolved")

    query = f"{question.title}\n{question.input}"
    digests = await asyncio.to_thread(
        context.library.digests,
        query,
        papers,
        max_pages_per_paper=4,
    )
    validate_paper_evidence(papers, digests)

    initial_messages = messages_builder(question, digests)
    messages = initial_messages
    last_issue = ""
    audit_prefix = (
        f"{CasePolicy(context.config, question.case_id).work_prefix}/report-validation/{uuid4().hex}"
        if question.function == "Idea" else None
    )
    # One normal generation, at most one targeted repair. No model tool loop
    # or nested JSON retry is needed: the application owns final JSON assembly.
    for attempt in range(min(2, max(1, attempts))):
        try:
            text = await context.llm.complete(
                messages, max_tokens=max_tokens, temperature=0.2
            )
        except LLMTruncatedError:
            last_issue = "报告达到输出Token上限，残缺内容未被采用"
            _record_attempt(context, question, audit_prefix, attempt + 1,
                            None, None, [last_issue], digests)
            if attempt == 0:
                messages = initial_messages + [{
                    "role": "user",
                    "content": (
                        f"上一版报告不合格：{last_issue}。"
                        "请重新生成完整Markdown正文，保留全部必要章节、至少800字符和真实引用，"
                        "目标控制在1200至2500字符，减少重复论述；不要续写残缺JSON或输出参考文献。"
                    ),
                }]
            continue
        body = strip_references(clean_markdown(text))
        cited, issues = validate_report_body(body, digests)
        if question.function == "Idea":
            issues.extend(validate_idea_body(body, question, digests))
        _record_attempt(context, question, audit_prefix, attempt + 1, text, body, issues, digests)
        if not issues:
            markdown = assemble_report(body, digests, cited)
            return {
                "case_id": question.case_id,
                "path": path_builder(context.config.team, question.case_id),
                "markdown": markdown,
            }

        last_issue = "；".join(issues)
        messages = initial_messages + [
            {"role": "assistant", "content": text},
            {
                "role": "user",
                "content": (
                    f"上一版报告不合格：{last_issue}。"
                    "请针对具体缺少的正文要素补充分析，不要只改标题或追加空标签；"
                    "请重写完整报告，使用 [编号] 或 [编号, p.页码] 引用给定论文，"
                    "不要输出参考文献部分。"
                    + ("\n逐节检查表：\n" + idea_content_checklist()
                       if question.function == "Idea" else "")
                ),
            },
        ]
    diagnostic = f"; diagnostic_work={audit_prefix}" if audit_prefix else ""
    raise HandlerError(f"{question.case_id}: report validation failed: {last_issue}{diagnostic}")
