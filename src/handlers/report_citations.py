"""Parse report citations and check them against the excerpts actually supplied.

Numeric square brackets in prose are citations. Markdown links, fenced code and
inline code are excluded, so examples in code cannot satisfy citation coverage.
Page numbers refer to the one-based PDF pages in ``PaperExcerpt``; a page range
is valid only when every page was supplied, not merely its two endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from src.materials.models import PaperDigest, has_evidence_text

_BRACKETS_RE = re.compile(r"(?<!\\)\[([^\[\]\n]*)\](?!\s*\()")
_CANDIDATE_RE = re.compile(r"^\s*[+-]?[0-9]")
_CITATION_RE = re.compile(
    r"^\s*(?P<numbers>[0-9]+(?:\s*[,，;；]\s*[0-9]+)*)\s*"
    r"(?:[,，]\s*pp?\.?\s*(?P<start>[0-9]+)\s*"
    r"(?:[-–—−]\s*(?P<end>[0-9]+))?)?\s*$",
    re.IGNORECASE,
)
_FENCE_RE = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})(?P<tail>.*)$")
_PAGE_LIST_RE = re.compile(
    r"^\s*(?P<number>[0-9]+)\s*[,，]\s*"
    r"(?P<pages>pp?\.?\s*[0-9]+(?:\s*[,，]\s*pp?\.?\s*[0-9]+)+)\s*$",
    re.IGNORECASE,
)
_INLINE_CODE_RE = re.compile(r"(?<!`)(`+)(?!`)(.*?)\1(?!`)", re.DOTALL)
_NUMBER_RE = re.compile(r"[0-9]+")


@dataclass(frozen=True)
class Citation:
    """One valid citation token; range validity is checked separately."""

    numbers: tuple[int, ...]
    page_start: int | None = None
    page_end: int | None = None
    source: str = ""


def prose_only(body: str) -> str:
    """Remove code spans without treating their sample citations as evidence."""
    lines: list[str] = []
    fence_character = ""
    fence_length = 0
    for line in body.splitlines(keepends=True):
        match = _FENCE_RE.match(line.rstrip("\r\n"))
        if fence_character:
            if (
                match
                and match["fence"][0] == fence_character
                and len(match["fence"]) >= fence_length
                and not match["tail"].strip()
            ):
                fence_character = ""
            lines.append("\n")
        elif match:
            fence_character = match["fence"][0]
            fence_length = len(match["fence"])
            lines.append("\n")
        else:
            lines.append(line)
    return _INLINE_CODE_RE.sub(" ", "".join(lines))


def _small_integer(value: str) -> int | None:
    # Neither a citation nor a PDF page can reasonably require billions of
    # entries. Bound conversion as well as range expansion for hostile output.
    significant = value.lstrip("0") or "0"
    return int(significant) if len(significant) <= 9 else None


def parse_citations(body: str) -> tuple[list[Citation], list[str]]:
    """Return citation tokens and errors, including malformed page annotations."""
    citations: list[Citation] = []
    issues: list[str] = []
    for bracket in _BRACKETS_RE.finditer(prose_only(body)):
        content = bracket[1]
        if not _CANDIDATE_RE.match(content):
            continue
        source = bracket[0]
        display = source if len(source) <= 100 else source[:97] + "..."
        match = _CITATION_RE.fullmatch(content)
        if not match:
            # A repeated p. identifies a list of pages from one paper, not
            # additional paper numbers. Validate each page independently.
            page_list = _PAGE_LIST_RE.fullmatch(content)
            if page_list:
                number = _small_integer(page_list["number"])
                pages = [_small_integer(value) for value in _NUMBER_RE.findall(page_list["pages"])]
                if number is None or any(page is None for page in pages):
                    issues.append(f"引用编号或页码过大：{display}")
                else:
                    citations.extend(
                        Citation((number,), page, page, source) for page in pages
                    )
                continue
            issues.append(f"引用格式无效：{display}")
            continue
        values = [_small_integer(value) for value in _NUMBER_RE.findall(match["numbers"])]
        start = _small_integer(match["start"]) if match["start"] is not None else None
        end = _small_integer(match["end"]) if match["end"] is not None else start
        if (
            any(value is None for value in values)
            or (match["start"] is not None and start is None)
            or (match["end"] is not None and end is None)
        ):
            issues.append(f"引用编号或页码过大：{display}")
            continue
        citations.append(
            Citation(tuple(value for value in values if value is not None), start, end, source)
        )
    return citations, issues


def cited_numbers(body: str, max_index: int) -> list[int]:
    """Return distinct in-range paper numbers, without asserting page validity."""
    citations, _ = parse_citations(body)
    return sorted(
        {
            number
            for citation in citations
            for number in citation.numbers
            if 1 <= number <= max_index
        }
    )


def validate_citations(body: str, digests: list[PaperDigest]) -> tuple[list[int], list[str]]:
    """Check every numeric citation and page against supplied paper evidence."""
    citations, issues = parse_citations(body)
    cited: set[int] = set()
    invalid_numbers: set[int] = set()
    for citation in citations:
        for number in citation.numbers:
            if not 1 <= number <= len(digests):
                invalid_numbers.add(number)
                continue
            cited.add(number)
            if citation.page_start is None:
                continue
            start, end = citation.page_start, citation.page_end
            assert end is not None
            digest = digests[number - 1]
            available = {
                excerpt.page
                for excerpt in digest.excerpts
                if excerpt.page > 0 and has_evidence_text(
                    excerpt.text, filename=digest.filename, title=digest.title
                )
            }
            # Count the available pages in the interval instead of expanding a
            # potentially enormous range from untrusted model output.
            if start < 1 or end < start:
                issues.append(f"引用页码范围无效：{citation.source}")
            elif sum(start <= page <= end for page in available) != end - start + 1:
                issues.append(
                    f"引用页码没有指定论文摘录支持：{citation.source}，"
                    f"论文 {number} 可用 PDF 页码为 {sorted(available)}"
                )
    if invalid_numbers:
        issues.append(f"存在无效引用编号：{sorted(invalid_numbers)}")
    return sorted(cited), list(dict.fromkeys(issues))
