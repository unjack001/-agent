"""C/D-only evidence selection across complete allowed PDFs and sections.

E continues to use PaperLibrary.digests unchanged. Excerpts always remain
contiguous text on a real page; coverage diagnostics never claim full reading
when a character budget required selection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
import logging
import re
from uuid import uuid4

from src.materials.models import MaterialError, PaperDigest, PaperExcerpt, has_evidence_text, normalize_evidence_text
from src.materials.paper_library import _tokenize
from src.time_limits import remaining_seconds

logger = logging.getLogger(__name__)
CONTEXT_CHARS = 120_000
WINDOW_CHARS = 2400
_KINDS = (
    ('references', r'references|bibliography|参考文献'),
    ('appendix', r'appendi(?:x|ces)|supplement(?:ary)?(?:\s+materials?)?|附录|补充材料'),
    ('method', r'(?:proposed\s+)?method(?:ology|s)?|(?:our\s+)?approach|(?:proposed\s+)?(?:algorithm|framework)s?|implementation(?:\s+details)?|problem\s+(?:formulation|definition)|方法|研究方法|算法|模型设计|实现细节|问题定义'),
    ('experiment', r'experiments?(?:\s+(?:setup|settings?|evaluation))?|experimental(?:\s+(?:setup|settings?|evaluation))?|evaluation(?:\s+(?:setup|protocol))?|datasets?|benchmarks?|baselines?|metrics|实验(?:设置|设计|方法|配置|评估)?|数据集|评价指标|评估设置|基线'),
    ('results', r'results?(?:\s+and\s+discussion)?|ablation(?:\s+stud(?:y|ies))?|analysis|实验结果|结果(?:与讨论|分析)?|消融(?:实验|分析)?'),
    ('overview', r'abstract|introduction|background|related\s+work|conclusions?|discussion|摘要|引言|背景|相关工作|结论|讨论'),
)


def _check():
    if remaining_seconds() <= 0:
        raise TimeoutError('paper evidence reading exhausted the current case budget')


def _kind(line):
    value = line.strip().strip('#').strip()
    if len(value) > 160:
        return None
    value = re.sub(r'^[一二三四五六七八九十]+[、.．]\s*', '', value)
    value = re.sub(r'^(?:(?:\d+(?:\.\d+)*|[A-Z](?:\.\d+)*|[IVX]+)[.、):：]?\s+|第[一二三四五六七八九十\d]+[章节]\s*)', '', value)
    for kind, pattern in _KINDS:
        if re.match(r'^(?:' + pattern + r')(?:\s*[:：.、]\s*.*|\s+(?:and|of|for|on|with|in|details|setup|settings|evaluation|[A-Z])\b.*)?$', value, re.I):
            return kind
    return None


@dataclass
class Section:
    kind: str
    pieces: list[tuple[int, int, str]] = field(default_factory=list)

    @property
    def chars(self):
        return sum(len(text) for _, _, text in self.pieces)


def sections(pages):
    result = []
    current = Section('overview')
    for number, text in enumerate(pages, 1):
        _check()
        start = offset = 0
        for line in text.splitlines(keepends=True):
            kind = _kind(line)
            if kind is not None and kind != current.kind:
                if offset > start:
                    current.pieces.append((number, start, text[start:offset]))
                if current.pieces:
                    result.append(current)
                current = Section(kind)
                start = offset
            offset += len(line)
        if len(text) > start:
            current.pieces.append((number, start, text[start:]))
    if current.pieces:
        result.append(current)
    return result


def _relevance(text, terms):
    tokens = _tokenize(text)
    if not tokens or not terms:
        return 0.0
    counts = Counter(tokens)
    hits = terms.intersection(counts)
    return len(hits) + sum(min(counts[term], 3) for term in hits) / (len(tokens) ** 0.5 + 1)


def fair_budgets(costs, budget):
    """Water-fill quotas, redistributing space unused by short papers/sections."""
    allocated = [0] * len(costs)
    active = [i for i, cost in enumerate(costs) if cost > 0]
    while active and budget > 0:
        share = max(1, budget // len(active))
        following = []
        for index in active:
            give = min(share, costs[index] - allocated[index], budget)
            allocated[index] += give
            budget -= give
            if allocated[index] < costs[index]:
                following.append(index)
        active = following
    return allocated


def _windows(section):
    """Prefer paragraph boundaries; split only oversized paragraphs."""
    result = []
    for page, base, text in section.pieces:
        start = 0
        for match in re.finditer(r'\n\s*\n', text):
            end = match.end()
            if end - start >= WINDOW_CHARS // 2:
                for offset in range(start, end, WINDOW_CHARS // 2):
                    value = text[offset:min(offset + WINDOW_CHARS, end)]
                    result.append((page, base + offset, value))
                start = end
        for offset in range(start, len(text), WINDOW_CHARS // 2):
            result.append((page, base + offset, text[offset:offset + WINDOW_CHARS]))
    return result


def best_window(text, terms, limit):
    if len(text) <= limit:
        return 0, text
    step = max(1, limit // 2)
    starts = list(range(0, len(text) - limit + 1, step)) + [len(text) - limit]
    offset = max(starts, key=lambda n: _relevance(text[n:n + limit], terms))
    return offset, text[offset:offset + limit]


def _take(section, budget, terms):
    if budget <= 0:
        return []
    if section.chars <= budget:
        return list(section.pieces)
    windows = _windows(section)
    # Relevance precedes position: a late algorithm, table or appendix must be
    # able to displace a prefix. The section tail is a useful tie-breaker.
    ranked = sorted(enumerate(windows), key=lambda item: (
        -_relevance(item[1][2], terms),
        0 if item[0] in (0, len(windows) - 1) else 1, item[0]))
    selected = []
    for _, (page, start, text) in ranked:
        _check()
        if budget <= 0:
            break
        if any(p == page and start < pos + len(value) and start + len(text) > pos
               for p, pos, value in selected):
            continue
        if len(text) > budget:
            # Choose the most relevant contiguous subwindow, not a fixed prefix.
            offset, text = best_window(text, terms, budget)
            start += offset
        selected.append((page, start, text))
        budget -= len(text)
    return sorted(selected, key=lambda item: (item[0], item[1]))


def select_sections(pages, query, budget):
    terms = set(_tokenize(query))
    available = [s for s in sections(pages) if s.kind != 'references']
    main = [s for s in available if s.kind in ('method', 'experiment', 'results')]
    main += [s for s in available if s.kind == 'appendix' and
             any(_relevance(text, terms) > 0 for _, _, text in s.pieces)]
    other = [s for s in available if s not in main]
    other.sort(key=lambda s: -max((_relevance(text, terms) for _, _, text in s.pieces), default=0))
    if not main:
        main, other = other, []
    main_cost = sum(s.chars for s in main)
    main_quotas = fair_budgets([s.chars for s in main], min(budget, main_cost))
    remaining = max(0, budget - sum(main_quotas))
    other_quotas = fair_budgets([s.chars for s in other], remaining)
    selected, audit = [], []
    for section, quota in zip(main + other, main_quotas + other_quotas):
        pieces = _take(section, quota, terms)
        selected.extend(pieces)
        audit.append({'kind': section.kind, 'pages': sorted({p for p, _, _ in section.pieces}),
                      'available_chars': section.chars, 'selected_chars': sum(len(t) for _, _, t in pieces),
                      'complete': section.chars <= quota})
    selected.sort(key=lambda item: (item[0], item[1]))
    return selected, audit


class EvidenceDigests(list):
    def __init__(self, items, audit):
        super().__init__(items)
        self.reading_audit = audit


def deep_digests(library, query, papers, *, role, budget=CONTEXT_CHARS):
    if role not in ('C', 'D'):
        raise ValueError('section evidence is reserved for C/D')
    documents = []
    for ref in papers:
        _check()
        pages = [normalize_evidence_text(text) for text in library.pages(ref)]
        title = library._title(pages, ref.filename)
        pages = [text if has_evidence_text(text, filename=ref.filename, title=title) else '' for text in pages]
        if not any(pages):
            raise MaterialError(f'材料问题：{ref.filename} 没有可用正文')
        documents.append((ref, title, pages))
    costs = [sum(s.chars for s in sections(pages) if s.kind != 'references')
             for _, _, pages in documents]
    quotas = fair_budgets(costs, budget)
    digests, audit = [], []
    for (ref, title, pages), quota in zip(documents, quotas):
        selected, coverage = select_sections(pages, query, quota)
        excerpts = tuple(PaperExcerpt(page, text.strip()) for page, _, text in selected
                         if has_evidence_text(text, filename=ref.filename, title=title))
        digests.append(PaperDigest(ref.filename, title, excerpts))
        audit.append({'filename': ref.filename, 'pdf_pages_read': len(pages),
                      'read_chars': sum(map(len, pages)), 'quota_chars': quota,
                      'selected_chars': sum(len(e.text) for e in excerpts),
                      'selected_pages': sorted({e.page for e in excerpts}), 'sections': coverage,
                      'source_char_limit': 500_000})
    return EvidenceDigests(digests, {'role': role, 'context_char_budget': budget,
                                   'papers': audit, 'selection': 'complete_sections_then_relevant_windows'})


def record_reading(context, question, digests):
    """Own-case diagnostic; failure to save must not override a valid result."""
    from src.materials.access import CasePolicy
    from src.submission.writer import ResultWriter
    try:
        policy = CasePolicy(context.config, question.case_id)
        relative = f'{policy.work_prefix}/paper-reading/{uuid4().hex}.json'
        policy.work_path(relative, for_write=True)
        ResultWriter(context.config).write_json(relative, {'case_id': question.case_id,
            'diagnostic_only': True, **digests.reading_audit})
    except Exception as exc:
        logger.warning('paper reading diagnostic unavailable case=%s type=%s', question.case_id, type(exc).__name__)
