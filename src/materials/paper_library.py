"""Filename-faithful paper library with lexical selection and page digests."""

from __future__ import annotations

import math
import os
import re
import threading
from collections import Counter
from pathlib import Path

from agenthack import tool

from src.config import RuntimeConfig
from src.materials.identity import unique_locations, unique_pdfs
from src.materials.models import (
    MaterialError,
    PaperCatalogEntry,
    PaperDigest,
    PaperExcerpt,
    PaperRef,
    has_evidence_text,
    normalize_evidence_text,
)
from src.tools.registry import pdf_extract_pages

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CJK_RE = re.compile(r"[一-鿿]+")
MAX_SCAN_FILES = 2000
MAX_SCAN_ENTRIES = 20_000
MAX_CONTEXT_CHARS = 120_000
MAX_PAGE_CHARS = 3000


def _tokenize(text: str) -> list[str]:
    lowered = text.lower()
    tokens = _TOKEN_RE.findall(lowered)
    for run in _CJK_RE.findall(lowered):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


def _unique(tokens: list[str]) -> list[str]:
    return list(dict.fromkeys(tokens))


def _build_idf(documents: list[list[str]]) -> dict[str, float]:
    if not documents:
        return {}
    counts: Counter[str] = Counter()
    for tokens in documents:
        counts.update(set(tokens))
    total = len(documents)
    return {token: math.log((total + 1) / (count + 1)) + 1.0 for token, count in counts.items()}


def _score(tokens: list[str], query_tokens: list[str], idf: dict[str, float]) -> float:
    if not tokens or not query_tokens:
        return 0.0
    counts = Counter(tokens)
    raw = sum(counts[token] * idf.get(token, 1.0) for token in query_tokens)
    return raw / (len(tokens) ** 0.5 + 1.0)


class PaperLibrary:
    def __init__(self, config: RuntimeConfig, workspace_root: Path | None = None) -> None:
        self.config = config
        self._workspace_root = workspace_root
        self._lock = threading.RLock()
        self._cache: dict[Path, tuple[float, list[str]]] = {}
        self._first_cache: dict[Path, tuple[float, list[str]]] = {}
        self._filename_index: dict[str, list[Path]] | None = None

    def set_workspace_root(self, root: Path) -> None:
        with self._lock:
            if self._workspace_root == root:
                return
            self._workspace_root = root
            self._filename_index = None

    @tool(name="paper_library_resolve")
    def resolve_materials(
        self,
        materials: list[str],
        *,
        allow_library_fallback: bool,
        strict_names: bool = True,
    ) -> list[PaperRef]:
        resolved: list[Path] = []
        missing: list[str] = []

        for material in materials:
            path = self._resolve_path(material)
            if path is None:
                path = self._resolve_by_name(material, strict_names=strict_names)
            if path is None:
                missing.append(material)
                continue
            if path.is_dir():
                resolved.extend(self._pdfs_in(path))
            else:
                resolved.append(path)

        if not resolved and allow_library_fallback:
            resolved = [ref.path for ref in self.all_papers(strict=strict_names)]
            missing = []

        missing_files = [
            item
            for item in missing
            if item.strip().lower().endswith(".pdf") or "/" in item
        ]
        if missing_files and not allow_library_fallback:
            raise MaterialError("materials not found: " + ", ".join(missing_files))

        if strict_names:
            try:
                resolved = unique_pdfs(resolved, conflict_message="duplicate PDF filename in materials")
            except ValueError as exc:
                raise MaterialError(str(exc)) from exc

        refs: list[PaperRef] = []
        seen_paths: set[Path] = set()
        seen_names: set[str] = set()
        for path in resolved:
            try:
                canonical = path.resolve()
            except OSError:
                continue
            if canonical in seen_paths:
                continue
            seen_paths.add(canonical)
            if canonical.name in seen_names:
                if strict_names:
                    raise MaterialError(
                        f"duplicate PDF filename in materials: {canonical.name}"
                    )
                continue
            seen_names.add(canonical.name)
            refs.append(PaperRef(filename=canonical.name, path=canonical))

        if not refs and not allow_library_fallback:
            raise MaterialError("no allowed PDF materials resolved")
        return refs

    def all_papers(self, *, strict: bool = True) -> list[PaperRef]:
        paths = [path for root in unique_locations(self._library_roots()) for path in self._pdfs_in(root)]
        if strict:
            try:
                paths = unique_pdfs(paths, conflict_message="duplicate PDF filename in library")
            except ValueError as exc:
                raise MaterialError(str(exc)) from exc
        refs: list[PaperRef] = []
        seen_paths: set[Path] = set()
        seen_names: set[str] = set()
        for path in paths:
            if path in seen_paths:
                continue
            seen_paths.add(path)
            if path.name in seen_names:
                if strict:
                    raise MaterialError(f"duplicate PDF filename in library: {path.name}")
                continue
            seen_names.add(path.name)
            refs.append(PaperRef(filename=path.name, path=path))
        return refs

    @tool(name="paper_library_catalog")
    def catalog(
        self,
        query: str,
        papers: list[PaperRef] | None = None,
        max_papers: int = 150,
    ) -> list[PaperCatalogEntry]:
        refs = list(papers) if papers is not None else self.all_papers()
        if not refs:
            return []

        # Extract first pages only for a query-relevant subset of a large library.
        if max_papers > 0 and len(refs) > max(max_papers * 2, 300):
            query_tokens = set(_unique(_tokenize(query)))
            refs.sort(
                key=lambda ref: (
                    -len(query_tokens & set(_tokenize(ref.filename))),
                    ref.filename.lower(),
                )
            )
            refs = refs[: max(max_papers * 2, 300)]

        entries: list[PaperCatalogEntry] = []
        documents: list[list[str]] = []
        for ref in refs:
            try:
                pages = self.first_page(ref)
            except MaterialError:
                pages = []
            first_page = pages[0] if pages else ""
            entries.append(
                PaperCatalogEntry(
                    filename=ref.filename,
                    title=self._title(pages, ref.filename),
                    snippet=first_page[:800].strip(),
                )
            )
            documents.append(_tokenize(ref.filename + " " + first_page))

        query_tokens = _unique(_tokenize(query))
        idf = _build_idf(documents)
        scored = [
            (_score(tokens, query_tokens, idf), entry)
            for tokens, entry in zip(documents, entries)
        ]
        scored.sort(key=lambda item: (-item[0], item[1].filename.lower()))
        if max_papers > 0:
            scored = scored[:max_papers]
        return [entry for _, entry in scored]

    @tool(name="paper_library_digests")
    def digests(
        self,
        query: str,
        papers: list[PaperRef],
        *,
        max_pages_per_paper: int = 4,
    ) -> list[PaperDigest]:
        query_tokens = _unique(_tokenize(query))
        result: list[PaperDigest] = []
        for ref in papers:
            pages = self.pages(ref)
            title = self._title(pages, ref.filename)
            evidence_pages = [
                normalize_evidence_text(page)
                if has_evidence_text(page, filename=ref.filename, title=title)
                else ""
                for page in pages
            ]
            indices = self._select_pages(evidence_pages, query_tokens, max_pages_per_paper)
            excerpts = tuple(
                PaperExcerpt(page=index + 1, text=evidence_pages[index][:MAX_PAGE_CHARS].strip())
                for index in indices
                if has_evidence_text(
                    evidence_pages[index][:MAX_PAGE_CHARS], filename=ref.filename, title=title
                )
            )
            result.append(PaperDigest(ref.filename, title, excerpts))
        return self._fit_budget(result)

    @tool(name="paper_library_catalog_page")
    def catalog_page(
        self, query: str, papers: list[PaperRef], *, offset: int = 0, limit: int = 15,
    ) -> list[PaperCatalogEntry]:
        """Page a stable filename ranking before extracting any PDF text.

        A catalog observation returns only a small page. Parsing hundreds of
        first pages before returning that observation can consume the entire
        task deadline, so extraction is limited to the requested page.
        """
        query_tokens = _unique(_tokenize(query))
        documents = [_tokenize(ref.filename) for ref in papers]
        idf = _build_idf(documents)
        ranked = sorted(
            zip(papers, documents),
            key=lambda item: (-_score(item[1], query_tokens, idf), item[0].filename.casefold()),
        )
        result: list[PaperCatalogEntry] = []
        for ref, _ in ranked[offset:offset + limit]:
            try:
                pages = self.first_page(ref)
            except MaterialError:
                pages = []
            result.append(PaperCatalogEntry(
                filename=ref.filename, title=self._title(pages, ref.filename),
                snippet=pages[0][:800].strip() if pages else "",
            ))
        return result

    def pages(self, paper: PaperRef | Path) -> list[str]:
        path = paper.path if isinstance(paper, PaperRef) else paper
        try:
            canonical = path.resolve()
            mtime = canonical.stat().st_mtime
        except OSError as exc:
            raise MaterialError(f"cannot read paper {path}: {exc}") from exc

        with self._lock:
            cached = self._cache.get(canonical)
            if cached is not None and cached[0] == mtime:
                return list(cached[1])

        pages = pdf_extract_pages(canonical)
        with self._lock:
            self._cache[canonical] = (mtime, pages)
        return list(pages)

    def first_page(self, paper: PaperRef | Path) -> list[str]:
        path = paper.path if isinstance(paper, PaperRef) else paper
        try:
            canonical = path.resolve()
            mtime = canonical.stat().st_mtime
        except OSError as exc:
            raise MaterialError(f"cannot read paper {path}: {exc}") from exc

        with self._lock:
            full = self._cache.get(canonical)
            if full is not None and full[0] == mtime:
                return list(full[1][:1])
            cached = self._first_cache.get(canonical)
            if cached is not None and cached[0] == mtime:
                return list(cached[1])

        pages = pdf_extract_pages(canonical, max_pages=1)
        with self._lock:
            self._first_cache[canonical] = (mtime, pages)
        return list(pages)

    def _select_pages(
        self,
        pages: list[str],
        query_tokens: list[str],
        limit: int,
    ) -> list[int]:
        available = [index for index, page in enumerate(pages) if page.strip()]
        if not available:
            return []
        if limit <= 1:
            return available[:1]
        if len(available) <= limit:
            return available

        page_tokens = [_tokenize(page) for page in pages]
        idf = _build_idf(page_tokens)
        scores = [_score(tokens, query_tokens, idf) for tokens in page_tokens]
        selected = {available[0]}
        order = sorted(available, key=lambda index: (-scores[index], index))
        for index in order:
            if len(selected) >= limit:
                break
            selected.add(index)
        return sorted(selected)

    def select_page_numbers(self, query: str, pages: list[str], *, limit: int = 4) -> list[int]:
        """Return real one-based page numbers for query-relevant extracted text."""
        return [index + 1 for index in self._select_pages(pages, _unique(_tokenize(query)), limit)]

    def _fit_budget(self, digests: list[PaperDigest]) -> list[PaperDigest]:
        total = sum(len(excerpt.text) for digest in digests for excerpt in digest.excerpts)
        if total <= MAX_CONTEXT_CHARS or not digests:
            return digests

        per_paper = max(1500, MAX_CONTEXT_CHARS // len(digests))
        adjusted: list[PaperDigest] = []
        for digest in digests:
            remaining = per_paper
            excerpts: list[PaperExcerpt] = []
            for excerpt in digest.excerpts:
                if remaining <= 0:
                    break
                text = normalize_evidence_text(excerpt.text[:remaining])
                if has_evidence_text(text, filename=digest.filename, title=digest.title):
                    excerpts.append(PaperExcerpt(page=excerpt.page, text=text))
                    remaining -= len(text)
            adjusted.append(
                PaperDigest(filename=digest.filename, title=digest.title, excerpts=tuple(excerpts))
            )
        return adjusted

    def _title(self, pages: list[str], fallback: str) -> str:
        for page in pages:
            for line in normalize_evidence_text(page).splitlines():
                line = line.strip()
                if len(line) >= 3:
                    return line[:300]
        return fallback

    def _material_roots(self) -> list[Path]:
        candidates: list[Path] = []
        if self._workspace_root is not None:
            candidates.extend([self._workspace_root, self._workspace_root / "data"])
        candidates.extend(
            [
                self.config.data_root,
                self.config.result_root,
                self.config.result_root / "data",
            ]
        )
        candidates.extend(self._library_roots())
        return self._existing_unique(candidates)

    def _library_roots(self) -> list[Path]:
        candidates: list[Path] = []
        if self._workspace_root is not None:
            candidates.append(self._workspace_root / "data" / "论文库")
        candidates.extend(
            [
                self.config.data_root / "论文库",
                self.config.data_root / "papers",
                self.config.data_root / "paper_library",
                self.config.result_root / "data" / "论文库",
            ]
        )
        existing = self._existing_unique(candidates)
        if existing:
            return existing
        if self.config.data_root.is_dir():
            return [self.config.data_root.resolve()]
        return []

    def _existing_unique(self, candidates: list[Path]) -> list[Path]:
        roots: list[Path] = []
        for candidate in candidates:
            try:
                resolved = candidate.expanduser().resolve()
            except OSError:
                continue
            if resolved.is_dir() and resolved not in roots:
                roots.append(resolved)
        return roots

    def _allowed_roots(self) -> list[Path]:
        roots: list[Path] = []
        for candidate in (self.config.data_root, self.config.result_root, self._workspace_root):
            if candidate is None:
                continue
            try:
                resolved = candidate.expanduser().resolve()
            except OSError:
                continue
            if resolved == Path("/"):
                continue
            if resolved not in roots:
                roots.append(resolved)
        return roots

    def _is_allowed(self, path: Path) -> bool:
        resolved = path.resolve()
        for root in self._allowed_roots():
            if resolved == root or resolved.is_relative_to(root):
                return True
        return False

    def _resolve_path(self, material: str) -> Path | None:
        raw = material.strip().strip("\"'")
        if not raw:
            return None
        path = Path(raw).expanduser()
        candidates: list[Path] = []
        if path.is_absolute():
            candidates.append(path)
        else:
            for root in self._material_roots():
                candidates.append(root / path)
                if path.parts and path.parts[0] == "data":
                    candidates.append(root.joinpath(*path.parts[1:]))
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if not resolved.exists() or not self._is_allowed(resolved):
                continue
            if resolved.is_dir():
                return resolved
            if resolved.is_file() and resolved.suffix.lower() == ".pdf":
                return resolved
        return None

    def _resolve_by_name(self, material: str, *, strict_names: bool = True) -> Path | None:
        name = Path(material.strip().strip("\"'")).name
        if not name.lower().endswith(".pdf"):
            return None
        index = self._index()
        exact = index.get(name, [])
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            if strict_names:
                raise MaterialError(f"duplicate PDF filename: {name}")
            return sorted(exact)[0]
        matches = [
            path
            for filename, paths in index.items()
            if filename.lower() == name.lower()
            for path in paths
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            if strict_names:
                raise MaterialError(f"ambiguous PDF filename: {name}")
            return sorted(matches)[0]
        return None

    def _index(self) -> dict[str, list[Path]]:
        with self._lock:
            if self._filename_index is not None:
                return self._filename_index
            index: dict[str, list[Path]] = {}
            seen: set[Path] = set()
            for root in self._library_roots():
                for path in self._pdfs_in(root):
                    if path in seen:
                        continue
                    seen.add(path)
                    index.setdefault(path.name, []).append(path)
            self._filename_index = index
            return index

    def _pdfs_in(self, directory: Path) -> list[Path]:
        results: list[Path] = []
        visited = 0
        try:
            for current, dirs, files in os.walk(directory):
                dirs.sort()
                for name in sorted(files):
                    visited += 1
                    if visited > MAX_SCAN_ENTRIES or len(results) >= MAX_SCAN_FILES:
                        return results
                    if not name.lower().endswith(".pdf"):
                        continue
                    try:
                        resolved = (Path(current) / name).resolve()
                        if self._is_allowed(resolved):
                            results.append(resolved)
                    except OSError:
                        continue
        except OSError as exc:
            raise MaterialError(f"cannot scan paper library {directory}: {exc}") from exc
        return results
