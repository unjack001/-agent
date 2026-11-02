"""B-only full-text ranking and longer, page-labelled evidence excerpts."""
from __future__ import annotations

import re
import time
import math
from collections import Counter, OrderedDict

from src.materials.models import MaterialError, PaperCatalogEntry, normalize_evidence_text
from src.materials.paper_library import _score, _tokenize, _unique
from src.time_limits import remaining_seconds
from src.tools.registry import pdf_extract_pages

MAX_RANK_CHARS = 100_000
MAX_SNIPPET_CHARS = 6000
WINDOW_CHARS = 2000
_ABSTRACT = re.compile(r'\babstract\b|摘\s*要', re.I)


class RetrievalCatalog(list):
    """Keep diagnostics with this call, rather than on a shared library instance."""

    def __init__(self, entries, reading_audit):
        super().__init__(entries)
        self.reading_audit = reading_audit


def _excerpts(pages: list[str], query: list[str]) -> str:
    windows = []
    for page, raw in enumerate(pages, 1):
        text = normalize_evidence_text(raw)
        if not text:
            continue
        starts = set(range(0, len(text), WINDOW_CHARS))
        abstract = _ABSTRACT.search(text)
        if abstract:
            starts.add(abstract.start())
        for start in sorted(starts):
            snippet = text[start:start + WINDOW_CHARS]
            tokens = _tokenize(snippet)
            # Keep abstract coverage even for Chinese queries and English PDFs.
            priority = bool(abstract and start <= abstract.start() < start + WINDOW_CHARS)
            windows.append((priority, _score(tokens, query, {}), page, start, snippet))
    windows.sort(key=lambda x: (-x[0], -x[1], x[2], x[3]))
    selected, used, spans = [], 0, []
    for _, _, page, start, text in windows:
        if any(p == page and start < b and start + len(text) > a for p, a, b in spans):
            continue
        header = f'[PDF p.{page}, chars {start + 1}-{start + len(text)}]\n'
        available = MAX_SNIPPET_CHARS - used - len(header) - 2
        if available <= 0:
            break
        text = text[:available]
        selected.append(header + text)
        used += len(header) + len(text) + 2
        spans.append((page, start, start + len(text)))
    return '\n\n'.join(selected)


def _read_index(library, ref):
    """Read two public PDF pages, with a bounded per-library LRU cache."""
    key = None
    try:
        path = ref.path.resolve()
        info = path.stat()
        key = (str(path), info.st_mtime_ns, info.st_size)
    except OSError:
        pass
    with library._lock:
        cache = getattr(library, '_retrieval_index_cache', None)
        if cache is None:
            cache = library._retrieval_index_cache = OrderedDict()
            library._retrieval_index_chars = 0
        if key is not None and key in cache:
            pages = cache.pop(key)
            cache[key] = pages
            return list(pages), True
    pages = pdf_extract_pages(ref.path, max_pages=2)
    result, used = [], 0
    for page in pages:
        text = normalize_evidence_text(page)[:MAX_RANK_CHARS - used]
        result.append(text)
        used += len(text)
        if used >= MAX_RANK_CHARS:
            break
    with library._lock:
        if key is not None:
            existing = cache.pop(key, None)
            if existing is not None:
                library._retrieval_index_chars -= sum(map(len, existing))
            cache[key] = tuple(result)
            library._retrieval_index_chars += used
            while len(cache) > 256 or library._retrieval_index_chars > 4_000_000:
                _, removed = cache.popitem(last=False)
                library._retrieval_index_chars -= sum(map(len, removed))
    return result, False


def build_catalog(library, query: str, papers=None, max_papers: int = 150):
    refs = list(papers) if papers is not None else library.all_papers()
    query_tokens = _unique(_tokenize(query))
    entries, documents, audit = [], [], []
    started = time.monotonic()
    for ref in refs:
        if remaining_seconds(reserve=10) <= 0:
            raise TimeoutError('paper reading exhausted the current case budget')
        try:
            pages, cache_hit = _read_index(library, ref)
        except MaterialError:
            cache_hit = False
            try:
                pages = library.first_page(ref)
            except MaterialError:
                pages = []
        ranked_pages, used = [], 0
        for raw in pages:
            text = normalize_evidence_text(raw)[:MAX_RANK_CHARS - used]
            ranked_pages.append(text)
            used += len(text)
            if used >= MAX_RANK_CHARS:
                break
        title = library._title(pages, ref.filename)
        entries.append(PaperCatalogEntry(ref.filename, title, _excerpts(ranked_pages, query_tokens)))
        tokens = _tokenize(ref.filename + ' ' + ' '.join(ranked_pages))
        counts = Counter(tokens)
        documents.append(({term: counts[term] for term in query_tokens}, len(tokens)))
        audit.append({'filename': ref.filename, 'indexed_pages': len(pages), 'pdf_pages': len(pages), 'cache_hit': cache_hit,
                      'ranked_chars': used, 'shown_chars': len(entries[-1].snippet),
                      'scan_capped': used >= MAX_RANK_CHARS})
    idf = {term: math.log((len(documents) + 1) /
                         (sum(counts[term] > 0 for counts, _ in documents) + 1)) + 1
           for term in query_tokens}
    scored = sorted(zip(entries, documents),
                    key=lambda item: (-sum(item[1][0][term] * idf[term] for term in query_tokens)
                                      / (item[1][1] ** 0.5 + 1), item[0].filename.casefold()))
    reading = {'papers_read': audit, 'seconds': time.monotonic() - started,
               'index_page_limit': 2, 'per_paper_rank_cap': MAX_RANK_CHARS, 'per_paper_prompt_cap': MAX_SNIPPET_CHARS}
    return RetrievalCatalog([entry for entry, _ in (scored[:max_papers] if max_papers > 0 else scored)], reading)
