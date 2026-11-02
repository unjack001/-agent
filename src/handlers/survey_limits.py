"""The c1-output-limits-v1 body and citation-marker limits."""
from __future__ import annotations

import re

from src.handlers.survey_citations import citation_marker_count

BODY_LIMITS = {"Survey": 5000}
_REFERENCE = re.compile(r"^##[ \t]+参考文献[ \t]*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def report_body(markdown: str) -> str:
    """Return the exact prefix before the reference heading, without stripping.

    Spaces, punctuation, Markdown symbols and the separating newlines count.
    A heading shown inside a code example does not start a reference section.
    """
    offset = 0
    fence = ""
    length = 0
    for line in markdown.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        marker = _FENCE.match(content)
        if marker:
            if not fence:
                fence, length = marker[1][0], len(marker[1])
            elif marker[1][0] == fence and len(marker[1]) >= length and not marker[2].strip():
                fence = ""
        elif not fence and _REFERENCE.fullmatch(content):
            return markdown[:offset]
        offset += len(line)
    return markdown


def validate_report_limits(markdown: str, function: str) -> list[str]:
    body = report_body(markdown)
    issues: list[str] = []
    maximum = BODY_LIMITS.get(function)
    if maximum is not None and len(body) > maximum:
        issues.append(f"{function} 正文为 {len(body)} 字符，最多 {maximum} 字符（参考文献不计入）")
    if function == "Survey":
        count = citation_marker_count(body)
        if count > 40:
            issues.append(f"Survey 正文引用标记为 {count} 次，最多 40 次；复合标注只计一组")
    return issues
