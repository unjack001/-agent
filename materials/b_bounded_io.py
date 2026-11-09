"""B-only killable scan/PDF workers; no models or generated code run here."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from agenthack import tool

from src.handlers.b_budget import BDeadlineReached
from src.materials.access import CasePolicy
from src.materials.models import MaterialError, PaperRef, PaperCatalogEntry
from src.submission.writer import ResultWriter, SubmissionError


class BoundedBReader:
    def __init__(self, context, question, budget):
        self.context, self.question, self.budget = context, question, budget
        self.payload = None

    def _base(self):
        cfg = self.context.config
        return {'data_root': str(cfg.data_root), 'result_root': str(cfg.result_root),
                'team': cfg.team, 'case_id': self.question.case_id}

    def _command(self, operation):
        return [sys.executable, '-m', 'src.materials.b_bounded_io']

    @tool(name='b_bounded_local_files')
    def _invoke(self, operation):
        left = self.budget.remaining()
        if left <= 0:
            raise BDeadlineReached(operation)
        payload = dict(self._base(), operation=operation, **self.payload)
        child = subprocess.Popen(self._command(operation),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=os.name == 'posix')
        try:
            out, _ = child.communicate(json.dumps(payload, ensure_ascii=False).encode(), timeout=left)
        except BaseException:
            # The group includes any OCR child; never leave it running after
            # its tool span has ended and the current answer has been returned.
            if child.poll() is None:
                if os.name == 'posix':
                    try: os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                else:
                    child.kill()
            child.communicate()
            self.budget.events.append({'stage': 'file_worker_stopped', 'operation': operation})
            if self.budget.remaining() <= 0:
                raise BDeadlineReached(operation) from None
            raise
        if child.returncode:
            raise MaterialError('B local file worker failed')
        result = json.loads(out)
        error = result.get('error')
        if error:
            kind = {'PermissionError': PermissionError, 'ValueError': ValueError}.get(
                result.get('error_type'), MaterialError)
            raise kind(error)
        return result

    def resolve(self):
        cache = self.context.library.material_scan_cache
        with self.context.library._lock:
            self.payload = {'materials': self.question.materials,
                'scan_cache': [[k[0], k[1], list(k[2]), [str(p) for p in paths]]
                               for k, paths in cache.items()]}
        result = self._invoke('materials')
        policy = CasePolicy(self.context.config, self.question.case_id)
        refs = [PaperRef(Path(p).name, policy.authorize_paper(Path(p))) for p in result['paths']]
        with self.context.library._lock:
            for data, output, identity, paths in result['scan_cache']:
                cache[(data, output, tuple(identity))] = tuple(Path(p) for p in paths)
        try:
            ResultWriter(self.context.config).write_json(f'{policy.work_prefix}/material-snapshot.json', {
                'case_id': self.question.case_id,
                'papers': [{'filename':p.filename,'path':str(p.path)} for p in refs]})
        except (OSError, ValueError, SubmissionError):
            pass
        return refs

    def catalog(self, query, papers):
        cached = {}
        with self.context.library._lock:
            for ref in papers:
                item = self.context.library._first_cache.get(ref.path.resolve())
                if item is not None:
                    cached[str(ref.path)] = [item[0], item[1]]
        self.payload = {'paths': [str(p.path) for p in papers], 'query': query, 'cached_first': cached}
        result = self._invoke('catalog')
        with self.context.library._lock:
            for path, item in result['cached_first'].items():
                self.context.library._first_cache[Path(path).resolve()] = (item[0], item[1])
        return [PaperCatalogEntry(**entry) for entry in result['entries']]

    def evidence(self, ref, query):
        self.payload = {'paths': [str(ref.path)], 'query': query}
        return self._invoke('evidence')['evidence']

    def svg_index(self, refs):
        with self.context.library._lock:
            cache=dict(getattr(self.context.library,'_b_svg_index_cache',{}))
        self.payload={'paths':[str(r.path) for r in refs],'index_cache':cache}
        result=self._invoke('quick_index')
        with self.context.library._lock:
            self.context.library._b_svg_index_cache=result['index_cache']
        return result['documents']

    def svg_abstract(self, ref):
        self.payload={'paths':[str(ref.path)]}
        return self._invoke('front_matter')['abstract']

    def svg_body(self, ref, questions, terms):
        self.payload={'paths':[str(ref.path)],'questions':questions,'terms':terms}
        return self._invoke('body_hits')['evidence']


def worker(payload):
    from src.config import RuntimeConfig
    from src.materials.pdf import extract_pages
    from src.materials.paper_library import PaperLibrary
    from src.materials.b_best_catalog import best_catalog
    from src.handlers.b_reading_review import read_evidence
    from dataclasses import asdict
    config = RuntimeConfig(Path(payload['data_root']), Path(payload['result_root']),
                           payload['team'], '', '', '')
    policy = CasePolicy(config, payload['case_id'])
    if payload['operation'] == 'materials':
        cache = {(d, o, tuple(i)): tuple(Path(p) for p in paths)
                 for d, o, i, paths in payload['scan_cache']}
        policy = CasePolicy(config, payload['case_id'], scan_cache=cache)
        paths = policy.material_files(payload['materials'], allow_library_fallback=True)
        return {'paths': [str(p) for p in paths], 'scan_cache': [
            [d, o, list(i), [str(p) for p in paths]] for (d, o, i), paths in cache.items()]}
    refs = [PaperRef(Path(p).name, policy.authorize_paper(Path(p))) for p in payload['paths']]
    if payload['operation']=='quick_index':
        from src.materials.b_svg_evidence import quick_meta
        cache=payload['index_cache'];documents=[]
        for ref in refs:
            stat=ref.path.stat();key=str(ref.path)
            item=cache.get(key)
            if item is None or item['mtime_ns']!=stat.st_mtime_ns or item['size']!=stat.st_size:
                item={'mtime_ns':stat.st_mtime_ns,'size':stat.st_size,'meta':quick_meta(ref.path)}
                cache[key]=item
            documents.append(item['meta'])
        return {'documents':documents,'index_cache':cache}
    if payload['operation']=='front_matter':
        from src.materials.b_svg_evidence import front_matter
        return {'abstract':front_matter(refs[0].path)}
    if payload['operation']=='body_hits':
        from src.materials.b_svg_evidence import body_evidence
        return {'evidence':body_evidence(refs[0].path,payload['questions'],payload['terms'])}
    class Reader:
        def __init__(self): self.first = payload.get('cached_first', {})
        def first_page(self, ref):
            key = str(ref.path)
            mtime = ref.path.stat().st_mtime
            cached = self.first.get(key)
            if cached is not None and cached[0] == mtime:
                return cached[1]
            pages = extract_pages(ref.path, max_pages=1)
            self.first[key] = [mtime, pages]
            return pages
        def pages(self, ref): return extract_pages(ref.path)
        def _title(self, pages, fallback): return PaperLibrary._title(self, pages, fallback)
    library = Reader()
    if payload['operation'] == 'catalog':
        entries = best_catalog(library, payload['query'], refs, max_papers=150)
        return {'entries': [asdict(e) for e in entries], 'cached_first': library.first}
    if payload['operation'] == 'evidence':
        return {'evidence': read_evidence(library, refs[0], payload['query'])}
    raise ValueError('unsupported B file operation')


if __name__ == '__main__':
    try:
        response = worker(json.loads(sys.stdin.buffer.read()))
    except Exception as exc:
        response = {'error_type': type(exc).__name__, 'error': str(exc)[:1000]}
    sys.stdout.write(json.dumps(response, ensure_ascii=False))
