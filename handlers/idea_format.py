"""Build C citations and references from one numbered source mapping."""
from __future__ import annotations

import re
import unicodedata

from src.handlers.report_citations import parse_citations, prose_only
from src.handlers.idea_reporting import strip_references
from src.materials.models import PaperDigest, has_evidence_text


class SurveyCitationFormatError(ValueError):
    """The output cannot be formatted without guessing a source or page."""


_BRACKET = re.compile(r"(?<!\\)\\?\[([^\[\]\n]+)\](?:\((?:[^()\n]|\([^()\n]*\))*\))?|【([^【】\n]+)】|［([^［］\n]+)］")
_RANGE = re.compile(r"([1-9][0-9]{0,2})\s*[-–—]\s*([1-9][0-9]{0,2})")
_STANDARD = re.compile(r"\[([1-9][0-9]*), p\. ([1-9][0-9]*)\]")
_REFERENCE = re.compile(r"^\[([1-9][0-9]*)\] (.+)$")


def _prose_mask(text: str) -> str:
    """Mask non-prose without changing character offsets."""
    def blank(value):
        return re.sub(r'[^\r\n]', ' ', value)
    lines = []
    fence, length = '', 0
    for line in text.splitlines(keepends=True):
        match = re.match(r'^ {0,3}(`{3,}|~{3,})(.*)$', line.rstrip('\r\n'))
        if fence:
            if match and match[1][0] == fence and len(match[1]) >= length and not match[2].strip():
                fence = ''
            lines.append(blank(line))
        elif match:
            fence, length = match[1][0], len(match[1])
            lines.append(blank(line))
        else:
            lines.append(line)
    text = ''.join(lines)
    text = re.sub(r'(?<!`)(`+)(?!`)(.*?)\1(?!`)', lambda m: blank(m[0]), text, flags=re.S)
    return re.sub(r'(?<!\\)\$\$[\s\S]*?(?<!\\)\$\$|'
                  r'(?<![\\$])\$(?![$\s])[^\n$]*?(?<![\\\s])\$(?![$0-9])|'
                  r'\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]', lambda m: blank(m[0]), text)


def _normalise_marker(token: str) -> str:
    value = unicodedata.normalize('NFKC', token).strip()
    value = value.replace('，', ',').replace('；', ';').replace('、', ',')
    match = _RANGE.fullmatch(value)
    if match:
        start, end = map(int, match.groups())
        if start > end or end - start > 149:
            raise SurveyCitationFormatError('引用编号范围无效，请使用逐篇编号')
        value = ','.join(str(n) for n in range(start, end + 1))
    return '[' + value + ']'


def format_idea(markdown: str, digests: list[PaperDigest], *, min_citations: int = 1,
                  require_pages: bool = False) -> str:
    """Canonicalise labels; never manufacture a missing paper or page number.

    Input labels refer to the numbered material list in the prompt. Final labels
    keep the passing image's original material numbers, with precisely the same
    mapping in the generated bibliography. Numeric citation links lose their wrapper;
    ordinary named links, code and math remain text.
    """
    body = strip_references(markdown).strip()
    visible = _prose_mask(body)
    numbered: dict[int, int] = {}
    replacements: list[tuple[int, int, str]] = []
    for match in _BRACKET.finditer(visible):
        token = next(group for group in match.groups() if group is not None)
        normal = unicodedata.normalize('NFKC', token).strip()
        if not re.match(r'\^?[+\-]?[0-9]', normal):
            continue
        if (match.start() and visible[match.start() - 1] == '[') or visible[match.end():match.end()+1] == ']':
            raise SurveyCitationFormatError('正文引用不能使用嵌套方括号')
        marker = _normalise_marker(token)
        citations, issues = parse_citations(marker)
        if issues or not citations:
            raise SurveyCitationFormatError('引用格式无法识别，只能使用 [1, p. 3] 这种带页码格式')
        output = []
        for citation in citations:
            for original in citation.numbers:
                if not 1 <= original <= len(digests):
                    raise SurveyCitationFormatError(f'引用编号 {original} 不在指定论文目录中')
                if citation.page_start is None:
                    if not require_pages:
                        numbered.setdefault(original, original)
                        output.append(f'[{original}]')
                        continue
                    raise SurveyCitationFormatError('正文引用缺少 PDF 页码，只能使用 [1, p. 3] 这种格式')
                start, end = citation.page_start, citation.page_end
                available = {e.page for e in digests[original - 1].excerpts
                             if e.page > 0 and has_evidence_text(e.text,
                                 filename=digests[original - 1].filename,
                                 title=digests[original - 1].title)}
                if end is None or start < 1 or end < start or end - start >= len(available):
                    raise SurveyCitationFormatError('引用页码范围没有指定论文摘录支持')
                pages = list(range(start, end + 1))
                if not set(pages) <= available:
                    raise SurveyCitationFormatError(f'论文 {original} 的引用页码未出现在提供的摘录中')
                number = numbered.setdefault(original, original)
                output.extend(f'[{number}, p. {page}]' for page in pages)
        replacements.append((match.start(), match.end(), ''.join(dict.fromkeys(output))))
    if not numbered:
        raise SurveyCitationFormatError('综述正文缺少标准编号引用，请引用实际提供的论文')
    if len(numbered) < min_citations:
        raise SurveyCitationFormatError(f'题目要求至少引用 {min_citations} 篇实际指定论文，目前只有 {len(numbered)} 篇')
    for start, end, value in reversed(replacements):
        body = body[:start] + value + body[end:]
    entries = []
    for original, number in sorted(numbered.items()):
        digest = digests[original - 1]
        title = ' '.join(digest.title.split())
        if not title or not digest.filename or '\n' in digest.filename or '\r' in digest.filename:
            raise SurveyCitationFormatError('指定论文标题或文件名缺失，无法生成参考文献')
        # Match the passing image's exact list-line serialization.
        entries.append(f'[{number}] {title}，{digest.filename}')
    result = body + '\n\n## 参考文献\n\n' + '\n'.join(entries) + '\n'
    validate_idea_format(result, require_pages=require_pages)
    return result


def validate_idea_format(markdown: str, *, require_pages: bool = False) -> None:
    """Assert standard source numbers and body/list correspondence."""
    body = strip_references(markdown)
    tail = markdown[len(body):].strip()
    lines = tail.splitlines()
    if not lines or lines[0] != '## 参考文献':
        raise SurveyCitationFormatError('参考文献标题必须为 ## 参考文献')
    references = []
    for line in lines[1:]:
        if not line.strip():
            continue
        match = _REFERENCE.fullmatch(line)
        if not match:
            raise SurveyCitationFormatError('参考文献必须逐行使用 [1] 条目格式')
        references.append(int(match[1]))
    if references != sorted(set(references)):
        raise SurveyCitationFormatError('参考文献编号必须按指定论文编号升序且不重复')
    prose = prose_only(body)
    used = []
    for match in _BRACKET.finditer(prose):
        token = next(group for group in match.groups() if group is not None)
        if not re.match(r'\^?[+\-]?[0-9]', unicodedata.normalize('NFKC', token).strip()):
            continue
        if (match.start() and prose[match.start() - 1] == '[') or prose[match.end():match.end()+1] == ']':
            raise SurveyCitationFormatError('正文引用不能使用嵌套方括号')
        standard = _STANDARD.fullmatch(match[0])
        if not require_pages and standard is None:
            standard = re.fullmatch(r'\[([1-9][0-9]*)\]', match[0])
        if not standard:
            raise SurveyCitationFormatError('正文引用只能为 [1, p. 3] 这种带页码格式')
        used.append(int(standard[1]))
    if not used or set(used) != set(references):
        raise SurveyCitationFormatError('正文引用与参考文献必须一一对应')
