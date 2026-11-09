"""Validation and assembly for cited Markdown reports."""
from __future__ import annotations
import asyncio
import json
import logging
import re
import time
from src.time_limits import deadline_end, case_seconds
from typing import Any, Callable
from uuid import uuid4
from src.dispatch.models import CaseQuestion
from src.handlers.base import HandlerContext, HandlerError
from src.handlers.idea_review import idea_content_checklist, idea_min_citations, review_idea_body
from src.handlers.idea_citations import cited_numbers as cited_numbers
from src.handlers.idea_citations import validate_citations
from src.handlers.idea_limits import validate_report_limits
from src.llm.parsing import LLMError, LLMOutputError, LLMTruncatedError
from src.llm.structured import request_window, retry_pause
from src.materials.access import CasePolicy
from src.materials.case_materials import CaseMaterials
from src.materials.models import PaperDigest, PaperRef, has_evidence_text
from src.submission.writer import ResultWriter, SubmissionError
from src.handlers.report_review import numeric_fact_issues
logger = logging.getLogger(__name__)
REFERENCE_HEADING_RE = re.compile('(?im)^\\s*(?:#{1,6}\\s*)?(?:\\*\\*)?(?:(?:[0-9]+|[一二三四五六七八九十]+)[.．、)）]\\s*)?(?:参考文献|references|bibliography)(?:\\*\\*)?\\s*[:：]?\\s*$')

class _DiagnosticWriter(ResultWriter):
    """E-only work JSON writer compatible with the unchanged base writer."""

    def write_json(self, relative_path, payload, *, ensure_ascii=True):
        parts = self._parts(relative_path)
        if not self._is_work_file(parts):
            raise SubmissionError("Idea diagnostics must remain in own-team work area")
        target = self._target(parts)
        text = json.dumps(payload, ensure_ascii=True, indent=2, allow_nan=False) + "\n"
        self._atomic_write(target, self._encode_utf8(text, relative_path))
        self.verify_json(relative_path, payload)
        return target


class ReportFormatError(ValueError):
    """A model wrapper does not contain a complete Markdown report."""

def _unwrap_fence(text: str) -> str | None:
    lines = text.splitlines()
    if not lines:
        return None
    opening = re.fullmatch(' {0,3}(`{3,}|~{3,})(?:[A-Za-z0-9_+-]*)[ \\t]*', lines[0])
    if opening is None:
        return None
    marker = opening[1]
    closing = re.compile(' {0,3}' + re.escape(marker[0]) + '{' + str(len(marker)) + ',}[ \\t]*')
    end = next((index for index, line in enumerate(lines[1:], 1) if closing.fullmatch(line)), None)
    if end is None:
        raise ReportFormatError('报告包装的 Markdown 代码围栏未完整闭合，请重新生成完整正文')
    if end != len(lines) - 1:
        return None
    return '\n'.join(lines[1:end]).strip()

def clean_markdown(text: str) -> str:
    if not isinstance(text, str):
        raise ReportFormatError('模型回复必须包含字符串 Markdown 正文')
    text = text.strip()
    for _ in range(6):
        unwrapped = _unwrap_fence(text)
        if unwrapped is not None:
            text = unwrapped
            continue
        if text.startswith('{'):
            try:
                data = json.loads(text)
            except ValueError as exc:
                raise ReportFormatError('模型返回的 JSON 包装不完整或无效，请重新生成完整 Markdown 正文') from exc
            if not isinstance(data, dict) or 'markdown' not in data:
                raise ReportFormatError('模型返回了旧版结构化字段或缺少 markdown；请直接输出完整 Markdown 正文')
            if not isinstance(data['markdown'], str) or not data['markdown'].strip():
                raise ReportFormatError('JSON 包装的 markdown 必须是非空字符串，不能是对象、数组或 null')
            text = data['markdown'].strip()
            continue
        return text
    raise ReportFormatError('报告包装嵌套过深，请直接输出完整 Markdown 正文')

def _repair_text(text: str) -> str:
    """Escape rejected code points before including them in another API call."""
    return text.encode('utf-8', 'backslashreplace').decode('utf-8')

def strip_references(body: str) -> str:
    offset = 0
    fence = ''
    fence_length = 0
    for line in body.splitlines(keepends=True):
        marker = re.match('^ {0,3}(`{3,}|~{3,})(.*)$', line.rstrip('\r\n'))
        if marker:
            if not fence:
                fence, fence_length = (marker[1][0], len(marker[1]))
            elif marker[1][0] == fence and len(marker[1]) >= fence_length and (not marker[2].strip()):
                fence = ''
        elif not fence and REFERENCE_HEADING_RE.fullmatch(line.strip()):
            return body[:offset].rstrip()
        offset += len(line)
    return body.rstrip()

def assemble_report(body: str, digests: list[PaperDigest], cited: list[int]) -> str:
    entries = [f'[{index}] {digests[index - 1].title}，{digests[index - 1].filename}' for index in cited]
    return body.rstrip() + '\n\n## 参考文献\n\n' + '\n'.join(entries) + '\n'

def validate_report_body(body: str, digests: list[PaperDigest], *, min_chars: int=800, min_citations: int=1, derived_values=()) -> tuple[list[int], list[str]]:
    issues: list[str] = []
    try:
        body.encode('utf-8')
    except UnicodeEncodeError:
        issues.append('报告包含无法保存为 UTF-8 的非法 Unicode 字符，请重新生成完整有效正文')
    if len(body.strip()) < min_chars:
        issues.append(f'正文至少需要 {min_chars} 个字符')
    cited, citation_issues = validate_citations(body, digests)
    issues.extend(citation_issues)
    required = min(min_citations, len(digests))
    if len(cited) < required:
        issues.append(f'正文只引用了 {len(cited)} 篇，至少需要引用 {required} 篇指定论文')
    issues.extend(numeric_fact_issues(body, digests, derived_values))
    return (cited, issues)

def validate_paper_evidence(papers: list[PaperRef], digests: list[PaperDigest]) -> None:
    """Require real extracted text for every specified paper, without renumbering it."""
    expected = [paper.filename for paper in papers]
    actual = [digest.filename for digest in digests]
    if len(set(actual)) != len(actual) or set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        raise HandlerError(f'paper digest/material mismatch: missing={missing}, unexpected={unexpected}')
    unreadable = [digest.filename for digest in digests if not any((excerpt.page > 0 and has_evidence_text(excerpt.text, filename=digest.filename, title=digest.title) for excerpt in digest.excerpts))]
    if unreadable:
        raise HandlerError('no paper text available for specified PDF(s): ' + ', '.join(unreadable) + '; provide readable text or OCR before generating a report')

def _record_attempt(context: HandlerContext, question: CaseQuestion, prefix: str | None, attempt: int, raw: str | None, body: str | None, issues: list[str], digests: list[PaperDigest], *, warnings: list[str] | None=None) -> None:
    """Keep rejected drafts in this case's work area, never as submissions.

    Optional diagnostics must not turn a valid answer into a failure. The same
    path and symlink checks still apply; no prompts, environment or keys are saved.
    """
    if prefix is None:
        return
    relative = f'{prefix}/attempt-{attempt}.json'
    try:
        CasePolicy(context.config, question.case_id).work_path(relative, for_write=True)
        _DiagnosticWriter(context.config).write_json(relative, {'case_id': question.case_id, 'function': question.function, 'attempt': attempt, 'validation_accepted': not issues and (not warnings), 'hard_validation_accepted': not issues, 'quality_warnings': warnings or [], 'issues': issues, 'raw_markdown': raw, 'body': body, 'sources': [{'filename': d.filename, 'excerpt_pages': [e.page for e in d.excerpts]} for d in digests]}, ensure_ascii=True)
        if issues:
            logger.warning('report %s attempt=%s rejected: %s; diagnostic_work=%s', question.case_id, attempt, '; '.join(issues)[:3000], prefix)
        elif warnings:
            logger.warning('report %s attempt=%s content review: %s; diagnostic_work=%s', question.case_id, attempt, '; '.join(warnings)[:3000], prefix)
    except (OSError, ValueError, SubmissionError) as exc:
        logger.warning('report diagnostics unavailable for %s (%s)', question.case_id, type(exc).__name__)

def _record_selection(context: HandlerContext, question: CaseQuestion, prefix: str, attempt: int, warnings: list[str]) -> None:
    try:
        relative = f'{prefix}/selection.json'
        CasePolicy(context.config, question.case_id).work_path(relative, for_write=True)
        _DiagnosticWriter(context.config).write_json(relative, {'case_id': question.case_id, 'selected_attempt': attempt, 'hard_validation_accepted': True, 'quality_warnings': warnings, 'content_quality_verified': False}, ensure_ascii=True)
    except (OSError, ValueError, SubmissionError) as exc:
        logger.warning('report selection diagnostic unavailable for %s (%s)', question.case_id, type(exc).__name__)

async def run_idea_report(context, question, messages_builder, path_builder, *, max_tokens=6000, attempts=None):
    if question.function != 'Idea':
        raise HandlerError('Idea-only handler cannot process another category')
    from src.handlers.verified_reporting import run_verified_report
    return await run_verified_report(context, question, messages_builder, path_builder,
                                     max_tokens=max_tokens, attempts=attempts)
