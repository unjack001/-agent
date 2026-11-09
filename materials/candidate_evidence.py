"""B-only targeted rereading; no library scan and no change to C evidence."""
from dataclasses import replace
from src.materials.paper_library import _tokenize
from src.materials.retrieval_catalog import _read_index, _excerpts
from src.materials.section_evidence import sections, _windows, _relevance, best_window
from src.time_limits import remaining_seconds
from src.materials.models import MaterialError

def enrich_candidates(library, refs, catalog, names, query, *, total_chars=80000, per_paper=10000):
    allowed={p.filename:p for p in refs}
    entries=[];audit=[];terms=set(_tokenize(query))
    quota=min(per_paper,total_chars//max(1,len(names)))
    for entry in catalog:
        if entry.filename not in names:continue
        if remaining_seconds()<=0:raise TimeoutError('B候选补读超出整题预算')
        ref=allowed.get(entry.filename)
        if ref is None:raise ValueError('候选论文不在当前允许材料中')
        if not callable(getattr(library,'pages',None)):
            entries.append(entry);audit.append({'filename':entry.filename,'expanded':False});continue
        try:
            if hasattr(library,'_lock'):
                pages,hit=_read_index(library,ref)
            else:
                pages=library.pages(ref);hit=False
        except (MaterialError,OSError,ValueError) as exc:
            entries.append(entry)
            audit.append({'filename':entry.filename,'expanded':False,'reading_error':type(exc).__name__})
            continue
        if not any(isinstance(page,str) and page.strip() for page in pages):
            entries.append(entry)
            audit.append({'filename':entry.filename,'expanded':False,'reading_error':'empty_text'})
            continue
        # Reserve actual abstract/overview and method evidence, then fill with
        # query-focused windows, all copied contiguously from allowed PDF pages.
        overview=[];methods=[];experiments=[]
        for section in sections(pages):
            if section.kind=='references':continue
            target=methods if section.kind=='method' else overview if section.kind=='overview' else experiments if section.kind in ('experiment','results') else None
            if target is not None:
                for page,start,text in _windows(section):target.append((_relevance(text,terms),page,start,text))
        blocks=[]
        for label,windows in [('overview',overview),('method',methods),('experiment',experiments)]:
            if not windows:continue
            _,page,_,text=max(windows,key=lambda x:x[0])
            _,text=best_window(text,terms,max(1,quota//5-60))
            blocks.append(f'[PDF p.{page}, {label}]\n{text}')
        used=sum(len(b)+2 for b in blocks)
        blocks.append(_excerpts(pages,list(terms),max(0,quota-used)))
        snippet='\n\n'.join(b for b in blocks if b)
        entries.append(replace(entry,snippet=snippet))
        audit.append({'filename':entry.filename,'expanded':True,'pages':len(pages),
                      'cache_hit':hit,'original_chars':len(entry.snippet),'evidence_chars':len(snippet)})
    return entries,audit
