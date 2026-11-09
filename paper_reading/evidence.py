"""B/D/E reading evidence selected by the supplied skill's core workflow."""
from src.materials.models import PaperDigest,PaperExcerpt,MaterialError,normalize_evidence_text,has_evidence_text
from src.materials.section_evidence import sections,fair_budgets,_take,EvidenceDigests
from src.materials.paper_library import _tokenize
from src.time_limits import remaining_seconds
from src.paper_reading.runtime import skill_metadata

def reading_digests(library,query,papers,*,role):
    if role not in ('D','E'):raise ValueError('skill digest selector is reserved for D/E')
    budget=16000 if role=='D' else 40000
    docs=[]
    for ref in papers:
        if remaining_seconds()<=0:raise TimeoutError('skill reading exhausted the original case deadline')
        pages=[normalize_evidence_text(text) for text in library.pages(ref)]
        title=library._title(pages,ref.filename)
        if not any(has_evidence_text(t,filename=ref.filename,title=title) for t in pages):
            raise MaterialError(f'论文正文不可读：{ref.filename}')
        core=[s for s in sections(pages) if s.kind in ('overview','method','experiment','results') or
              role=='D' and s.kind=='appendix']
        docs.append((ref,title,pages,core))
    quotas=fair_budgets([sum(s.chars for s in core) for _,_,_,core in docs],budget)
    digests=[];audit=[];terms=set(_tokenize(query))
    for (ref,title,pages,core),quota in zip(docs,quotas):
        # Allocate by scientific role, so long introductions cannot crowd out
        # actual methods, datasets and measured results.
        groups=[[s for s in core if s.kind=='overview'],[s for s in core if s.kind=='method'],
                [s for s in core if s.kind in ('experiment','results')],
                [s for s in core if s.kind=='appendix']]
        weights=[1,3,2,1] if role=='D' else [1,2,2,0]
        present=sum(w for group,w in zip(groups,weights) if group)
        allocations=[min(sum(s.chars for s in group),quota*w//max(1,present)) for group,w in zip(groups,weights)]
        extra=fair_budgets([max(0,sum(s.chars for s in group)-a) for group,a in zip(groups,allocations)],
                           max(0,quota-sum(allocations)))
        selected=[];coverage=[]
        for group,allocation,added in zip(groups,allocations,extra):
            per_section=fair_budgets([s.chars for s in group],allocation+added)
            for section,limit in zip(group,per_section):
                pieces=_take(section,limit,terms);selected.extend(pieces)
                coverage.append({'kind':section.kind,'available_chars':section.chars,
                                 'selected_chars':sum(len(t) for _,_,t in pieces)})
        selected.sort(key=lambda item:(item[0],item[1]))
        excerpts=tuple(PaperExcerpt(page,text.strip()) for page,_,text in selected
                       if has_evidence_text(text,filename=ref.filename,title=title))
        digests.append(PaperDigest(ref.filename,title,excerpts))
        audit.append({'filename':ref.filename,'pdf_pages_read':len(pages),'selected_pages':sorted({e.page for e in excerpts}),
                      'selected_chars':sum(len(e.text) for e in excerpts),'sections':coverage})
    return EvidenceDigests(digests,{'role':role,'skill':skill_metadata(role),'context_char_budget':budget,
                                  'papers':audit,'selection':'problem_method_experiment_evidence'})
