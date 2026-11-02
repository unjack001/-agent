"""Validation and assembly for cited Markdown reports."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from src.time_limits import deadline_end
from typing import Any, Callable
from uuid import uuid4

from src.dispatch.models import CaseQuestion
from src.handlers.base import HandlerContext, HandlerError
from src.handlers.idea_validation import idea_content_checklist, idea_min_citations, review_idea_body
from src.handlers.report_citations import cited_numbers as cited_numbers
from src.handlers.report_citations import validate_citations
from src.handlers.report_limits import validate_report_limits
from src.llm.parsing import LLMError, LLMTruncatedError
from src.materials.access import CasePolicy
from src.materials.case_materials import CaseMaterials
from src.materials.models import PaperDigest, PaperRef, has_evidence_text
from src.submission.writer import ResultWriter, SubmissionError

logger = logging.getLogger(__name__)

REFERENCE_HEADING_RE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:\*\*)?"
    r"(?:(?:[0-9]+|[一二三四五六七八九十]+)[.．、)）]\s*)?"
    r"(?:参考文献|references|bibliography)(?:\*\*)?\s*[:：]?\s*$"
)
REPORT_BUDGET_S = 900.0
REPORT_MODEL_TIMEOUT_S = 900.0
REPORT_SAVE_RESERVE_S = 5.0


class ReportFormatError(ValueError):
    """A model wrapper does not contain a complete Markdown report."""


def _unwrap_fence(text: str) -> str | None:
    lines = text.splitlines()
    if not lines:
        return None
    opening = re.fullmatch(r" {0,3}(`{3,}|~{3,})(?:[A-Za-z0-9_+-]*)[ \t]*", lines[0])
    if opening is None:
        return None
    marker = opening[1]
    closing = re.compile(r" {0,3}" + re.escape(marker[0]) + "{" + str(len(marker)) + r",}[ \t]*")
    end = next((index for index, line in enumerate(lines[1:], 1) if closing.fullmatch(line)), None)
    if end is None:
        raise ReportFormatError("报告包装的 Markdown 代码围栏未完整闭合，请重新生成完整正文")
    if end != len(lines) - 1:
        # This is a leading code sample with following prose, not an outer
        # wrapper. Preserve report code examples and their boundaries.
        return None
    return "\n".join(lines[1:end]).strip()


def clean_markdown(text: str) -> str:
    if not isinstance(text, str):
        raise ReportFormatError("模型回复必须包含字符串 Markdown 正文")
    text = text.strip()
    # A model may wrap Markdown in JSON, a code fence, or both. Unwrap only
    # complete containers; never synthesize a report from old structured fields.
    for _ in range(6):
        unwrapped = _unwrap_fence(text)
        if unwrapped is not None:
            text = unwrapped
            continue
        if text.startswith("{"):
            try:
                data = json.loads(text)
            except ValueError as exc:
                raise ReportFormatError("模型返回的 JSON 包装不完整或无效，请重新生成完整 Markdown 正文") from exc
            if not isinstance(data, dict) or "markdown" not in data:
                raise ReportFormatError("模型返回了旧版结构化字段或缺少 markdown；请直接输出完整 Markdown 正文")
            if not isinstance(data["markdown"], str) or not data["markdown"].strip():
                raise ReportFormatError("JSON 包装的 markdown 必须是非空字符串，不能是对象、数组或 null")
            text = data["markdown"].strip()
            continue
        return text
    raise ReportFormatError("报告包装嵌套过深，请直接输出完整 Markdown 正文")


def _repair_text(text: str) -> str:
    """Escape rejected code points before including them in another API call."""
    return text.encode("utf-8", "backslashreplace").decode("utf-8")


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
    try:
        body.encode("utf-8")
    except UnicodeEncodeError:
        issues.append("报告包含无法保存为 UTF-8 的非法 Unicode 字符，请重新生成完整有效正文")
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
    digests: list[PaperDigest], *, warnings: list[str] | None = None,
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
            "attempt": attempt, "validation_accepted": not issues and not warnings,
            "hard_validation_accepted": not issues, "quality_warnings": warnings or [],
            "issues": issues, "raw_markdown": raw, "body": body,
            "sources": [{"filename": d.filename,
                         "excerpt_pages": [e.page for e in d.excerpts]} for d in digests],
        }, ensure_ascii=True)
        if issues:
            logger.warning("report %s attempt=%s rejected: %s; diagnostic_work=%s",
                           question.case_id, attempt, "; ".join(issues)[:3000], prefix)
        elif warnings:
            logger.warning("report %s attempt=%s content review: %s; diagnostic_work=%s",
                           question.case_id, attempt, "; ".join(warnings)[:3000], prefix)
    except (OSError, ValueError, SubmissionError) as exc:
        logger.warning("report diagnostics unavailable for %s (%s)",
                       question.case_id, type(exc).__name__)


def _record_selection(
    context: HandlerContext, question: CaseQuestion, prefix: str,
    attempt: int, warnings: list[str],
) -> None:
    try:
        relative = f"{prefix}/selection.json"
        CasePolicy(context.config, question.case_id).work_path(relative, for_write=True)
        ResultWriter(context.config).write_json(relative, {
            "case_id": question.case_id, "selected_attempt": attempt,
            "hard_validation_accepted": True, "quality_warnings": warnings,
            "content_quality_verified": False,
        }, ensure_ascii=True)
    except (OSError, ValueError, SubmissionError) as exc:
        logger.warning("report selection diagnostic unavailable for %s (%s)",
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
    deadline = deadline_end(REPORT_BUDGET_S)
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
    # Only an E draft passing every hard check may survive a failed optional
    # quality repair. Never use a truncated/oversized/ungrounded draft as fallback.
    best: tuple[dict[str, str], list[str], int] | None = None
    audit_prefix = (
        f"{CasePolicy(context.config, question.case_id).work_prefix}/report-validation/{uuid4().hex}"
    )
    # One normal generation, at most one targeted repair. No model tool loop
    # or nested JSON retry is needed: the application owns final JSON assembly.
    for attempt in range(min(2, max(1, attempts))):
        remaining = deadline - time.monotonic() - REPORT_SAVE_RESERVE_S
        if remaining <= 0:
            last_issue = "报告处理累计时间预算已耗尽，未生成可提交的完整报告"
            _record_attempt(context, question, audit_prefix, attempt + 1,
                            None, None, [last_issue], digests)
            break
        try:
            text = await asyncio.wait_for(
                context.llm.complete(messages, max_tokens=max_tokens, temperature=0.2),
                timeout=min(REPORT_MODEL_TIMEOUT_S, remaining),
            )
        except asyncio.TimeoutError:
            last_issue = "模型请求超时；不自动重试未取得的输出"
            _record_attempt(context, question, audit_prefix, attempt + 1,
                            None, None, [last_issue], digests)
            if best is None:
                raise
            break
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
                        + ("正文最多4500字符。" if question.function == "Idea" else
                           "正文最多5000字符，引用标记最多40次，同段重复引用合并。")
                    ),
                }]
            continue
        except LLMError as exc:
            _record_attempt(context, question, audit_prefix, attempt + 1, None, None,
                            [f"网关请求失败（HTTP {exc.status_code or 'unknown'}，不重试）"], digests)
            if best is None:
                raise
            logger.warning("report %s optional repair gateway failed; retaining valid attempt=%s",
                           question.case_id, best[2])
            break
        warnings: list[str] = []
        try:
            body = strip_references(clean_markdown(text))
        except ReportFormatError as exc:
            body = None
            cited, issues = [], [str(exc)]
        else:
            # E requires grounded citations and cross-paper complementarity,
            # not Survey's five-paper minimum.
            cited, issues = validate_report_body(
                body, digests,
                min_chars=0 if question.function == "Idea" else 800,
                min_citations=(idea_min_citations(question, len(digests))
                               if question.function == "Idea" else 5),
            )
            if question.function == "Idea":
                required = idea_min_citations(question, len(digests))
                if required > len(digests):
                    issues.append(f"题目要求引用至少 {required} 篇，但指定材料只有 {len(digests)} 篇，不能降低题目要求或编造论文")
                review = review_idea_body(body, question, digests)
                issues.extend(review.errors)
                warnings.extend(review.warnings)
                # E's length requirement applies to the complete Markdown
                # field, including the real references appended by us. C
                # separately requires an 800-character report body.
                if len(assemble_report(body, digests, cited).strip()) < 800:
                    issues.append("完整 Markdown 报告至少需要 800 个字符（含参考文献）")
            issues.extend(validate_report_limits(
                assemble_report(body, digests, cited), question.function,
            ))
        # Validation messages may quote a malformed heading or citation too.
        issues = [_repair_text(issue) for issue in issues]
        warnings = [_repair_text(issue) for issue in warnings]
        _record_attempt(context, question, audit_prefix, attempt + 1, text, body, issues, digests,
                        warnings=warnings)
        if not issues:
            assert body is not None
            markdown = assemble_report(body, digests, cited)
            payload = {
                "case_id": question.case_id,
                "path": path_builder(context.config.team, question.case_id),
                "markdown": markdown,
            }
            if not warnings:
                _record_selection(context, question, audit_prefix, attempt + 1, [])
                return payload
            if best is None or len(warnings) < len(best[1]):
                best = (payload, warnings, attempt + 1)

        last_issue = "；".join(issues + warnings)
        messages = initial_messages + [
            # Keep a rejected surrogate out of the next HTTP request while
            # retaining its escaped form for a targeted full regeneration.
            {"role": "assistant", "content": _repair_text(text)},
            {
                "role": "user",
                "content": (
                    (f"上一版报告不合格：{last_issue}。" if issues else
                     f"上一版报告通过硬性检查，但存在内容自检提示：{last_issue}。"
                     "这些提示来自关键词启发式；先对照题目判断是否适用，不为匹配关键词修改事实。")
                    +
                    "请针对具体缺少的正文要素补充分析，不要只改标题或追加空标签；"
                    "请重写完整报告，使用 [编号] 或 [编号, p.页码] 引用给定论文，"
                    "不要输出参考文献部分。"
                    + ("完整正文不得超过4500字符，目标1800至3200字符。"
                       if question.function == "Idea" else
                       "完整正文不得超过5000字符，引用标记最多40次，目标约3500字符；"
                       "同一自然段重复引用请合并。")
                    + ("\n逐节检查表：\n" + idea_content_checklist()
                       if question.function == "Idea" else "")
                ),
            },
        ]
    if best is not None:
        _record_selection(context, question, audit_prefix, best[2], best[1])
        logger.warning("report %s selected hard-valid attempt=%s with content warnings; "
                       "diagnostic_work=%s", question.case_id, best[2], audit_prefix)
        return best[0]
    diagnostic = f"; diagnostic_work={audit_prefix}" if audit_prefix else ""
    raise HandlerError(f"{question.case_id}: report validation failed: {last_issue}{diagnostic}")
