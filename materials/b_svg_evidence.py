"""SVG B funnel: quick no-OCR metadata, front matter, bounded body hits."""
from __future__ import annotations

from pathlib import Path

from src.materials.models import normalize_evidence_text
from src.materials.paper_library import PaperLibrary
from src.materials.b_pdf_tools import score, window
from src.materials.section_evidence import sections


def native_pages(path, limit=None):
    """No OCR, native text only; the parent owns the killable process budget."""
    from pypdf import PdfReader
    pdf = PdfReader(str(path))
    if pdf.is_encrypted and pdf.decrypt('') == 0:
        raise ValueError('encrypted PDF')
    count = len(pdf.pages)
    pages = [normalize_evidence_text(pdf.pages[i].extract_text() or '')
             for i in range(min(limit,count) if limit is not None else count)]
    return pages, count, pdf.metadata


def quick_meta(path):
    name = Path(path).name
    try:
        pages, count, metadata = native_pages(path, 1)
        title = getattr(metadata, 'title', '') if metadata else ''
        title = title.strip() if isinstance(title, str) else ''
        if not title:
            title = PaperLibrary._title(None, pages, name)
        return {'filename':name, 'title':title, 'snippet':pages[0][:120] if pages else '',
                'page_count':count, 'ocr_used':False}
    except Exception as exc:
        return {'filename':name, 'title':name, 'snippet':'', 'page_count':0,
                'ocr_used':False, 'reading_error':type(exc).__name__}


def front_matter(path):
    # The SVG only forbids OCR in quick_meta and full-text search. Retain the
    # existing sparse-page/OCR fallback for reading at most two front pages.
    from src.materials.pdf import extract_pages
    pages = extract_pages(path, max_pages=2)
    return {'front_matter':'\n\n'.join(f'[PDF p.{i}] {text}' for i,text in enumerate(pages,1) if text.strip()),
            'pages':[i for i,text in enumerate(pages,1) if text.strip()], 'source':'front_two_pages'}


def body_evidence(path, questions, terms):
    pages, _, _ = native_pages(path)
    return body_evidence_from_pages(pages,questions,terms)


def body_evidence_from_pages(pages, questions, terms):
    query = '\n'.join([*questions, *terms])
    best = {}
    for section in sections(pages):
        if section.kind == 'references':
            continue
        for page, _, text in section.pieces:
            value = score(text, query)
            primary=section.kind in ('method','experiment','results')
            if value > 0 and (primary,value) > best.get(page,(False,0,'',''))[:2]:
                best[page] = primary,value,text,section.kind
    ordered = sorted(best, key=lambda page:(-best[page][0],-best[page][1],page))[:3]
    return {'pages':[{'page':page,'text':window(best[page][2],query,1500),
                     'section':best[page][3]} for page in ordered],
            'ocr_used':False}
