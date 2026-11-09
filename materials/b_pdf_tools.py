"""B-only PDFTools adapter: whole authorized library, bounded shown evidence."""
from dataclasses import replace
from collections import Counter
from pathlib import Path
import re
import threading
import os
from agenthack import tool
from src.materials.case_materials import CaseMaterials
from src.materials.access import CasePolicy,ensure_no_links,_private_name
from src.materials.pdf import extract_pages
from src.materials.paper_library import _tokenize
from src.materials.section_evidence import sections
from src.time_limits import remaining_seconds

STOP={'the','a','an','and','of','for','to','is','in','with','does','this','paper','whether','study','on','or','by'}
def terms(query):return set(_tokenize(query))-STOP

@tool(name='b_pdf_read_pages')
def read_pdf(path, max_pages=None):
    """Read actual allowed PDF pages; B full-text search has no character crop."""
    return extract_pages(path,max_pages=max_pages,prefer_native=True,max_total_chars=None)

def score(text,query):
    counts=Counter(_tokenize(text))
    return sum(min(counts[term],4) for term in terms(query))

def window(text,query,limit):
    if len(text)<=limit:return text
    # Center a long-page excerpt on a real matching token, not its page prefix.
    query_terms=sorted(terms(query),key=len,reverse=True)
    hits=[m.start() for term in query_terms if term for m in re.finditer(re.escape(term),text,re.I)]
    if hits:
        anchor=max(hits,key=lambda n:score(text[max(0,n-200):n+400],query))
        start=max(0,min(len(text)-limit,anchor-limit//2))
        return text[start:start+limit]
    return text[:limit]

class WholeLibraryPolicy(CasePolicy):
    """Same file capabilities as CasePolicy, no B paper-count scan crop."""
    def _scan_shared(self,directory):
        anchor=next((r for r in self.shared_roots() if directory.is_relative_to(r)),None)
        if anchor is None:raise PermissionError('only shared-paper directories may be scanned')
        ensure_no_links(directory,anchor);result=[]
        for current,dirs,files in os.walk(directory,followlinks=False):
            if remaining_seconds()<=0:raise TimeoutError('B library scan exhausted business budget')
            kept=[]
            for name in sorted(dirs):
                if _private_name(name):continue
                try:ensure_no_links(Path(current)/name,anchor)
                except PermissionError:continue
                kept.append(name)
            dirs[:]=kept
            for name in sorted(files):
                if remaining_seconds()<=0:raise TimeoutError('B library scan exhausted business budget')
                if not name.lower().endswith('.pdf') or _private_name(name):continue
                try:result.append(self.authorize_paper(Path(current)/name))
                except (PermissionError,ValueError):continue
        return result

class PDFTools:
    def __init__(self,context,question,materials=None):
        self.context=context;self.question=question;self.materials=materials
        cache=getattr(context.library,'_b_pipeline_page_cache',None)
        if cache is None:cache={};setattr(context.library,'_b_pipeline_page_cache',cache)
        self.cache=cache
        lock=getattr(context.library,'_b_pipeline_page_lock',None)
        if lock is None:lock=threading.RLock();setattr(context.library,'_b_pipeline_page_lock',lock)
        self.lock=lock

    def list_pdfs(self):
        scoped=replace(self.question,materials=[] if self.materials is None else self.materials)
        service=CaseMaterials(self.context,scoped)
        cache=getattr(self.context.library,'_b_full_scan_cache',None)
        if cache is None:cache={};setattr(self.context.library,'_b_full_scan_cache',cache)
        service.policy=WholeLibraryPolicy(self.context.config,self.question.case_id,
                                         scan_cache=cache,scan_lock=self.lock)
        return service.resolve()

    def pages(self,ref,limit=None):
        with self.lock:return self._pages(ref,limit)

    def _pages(self,ref,limit=None):
        stat=ref.path.stat();key=(str(ref.path.resolve()),stat.st_mtime_ns,stat.st_size)
        cached=self.cache.get(key)
        if cached and (cached[0] is None or limit is not None and cached[0]>=limit):
            return list(cached[1] if limit is None else cached[1][:limit])
        lib=self.context.library
        if hasattr(lib,'_lock'):
            result=read_pdf(ref.path,limit)
        else:result=lib.pages(ref)[:limit] if limit is not None else lib.pages(ref)
        self.cache[key]=(limit,tuple(result))
        # Cache eviction changes neither the reading queue nor extracted text.
        while sum(sum(map(len,item[1])) for item in self.cache.values())>16000000:
            self.cache.pop(next(iter(self.cache)))
        return list(result)

    def metadata(self,ref,query):
        pages=self.pages(ref,1)
        title=self.context.library._title(pages,ref.filename)
        first=pages[0] if pages else ''
        return {'filename':ref.filename,'title':title,'lexical_score':score(ref.filename+' '+title+' '+first,query)}

    def abstract(self,ref,title,query):
        pages=self.pages(ref,2);shown=[];found=False;started=False
        for i,text in enumerate(pages,1):
            match=re.search(r'\babstract\b|摘\s*要',text,re.I)
            if match:started=found=True;value=text[match.start():]
            elif started:value=text
            else:value=text
            stop=re.search(r'(?im)^\s*(?:[12][.、]?\s*)?(?:introduction|引言|keywords|关键词|methods?|研究方法)\b',value)
            if started and stop:value=value[:stop.start()]
            if value.strip():shown.append({'page':i,'text':window(value,query,9000)})
            if started and stop:break
        if not shown:shown=[{'page':i,'text':window(t,query,9000)} for i,t in enumerate(pages,1) if t.strip()]
        return {'filename':ref.filename,'title':title,'abstract_found':found,'pages':shown}

    def evidence(self,ref,abstract,questions,query):
        pages=self.pages(ref);search='\n'.join(questions) or query
        abstract_pages={p['page'] for p in abstract['pages']}
        best={}
        for section in sections(pages):
            if section.kind=='references':continue
            for i,_,text in section.pieces:
                value=score(text,search)
                if value>0 and value>best.get(i,(0,''))[0]:best[i]=(value,text)
        ranked=[(i,text,value) for i,(value,text) in best.items()]
        ranked.sort(key=lambda row:(row[0] in abstract_pages,-row[2],row[0]))
        chosen=[{'page':i,'text':window(text,search,17000)} for i,text,_ in ranked[:3]]
        return {'filename':ref.filename,'title':abstract['title'],'pages':chosen or abstract['pages'],
                'source':'body_hits' if chosen else 'actual_abstract_fallback'}
