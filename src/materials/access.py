"""Filesystem capabilities for one competition case; no arbitrary file access."""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path, PurePosixPath, PureWindowsPath

from src.config import RuntimeConfig
from src.materials.identity import location_key, unique_locations, unique_pdfs

SHARED_DIRECTORIES = ("论文库", "papers", "paper_library")
TEXT_SUFFIXES = {".txt", ".md", ".json", ".py", ".csv", ".tsv", ".log"}
MAX_TEXT_BYTES = 128_000
MAX_PAPERS = 2_000
MAX_SCAN_ENTRIES = 20_000
_PRIVATE = {
    "assigned",
    "dispatch",
    "expert",
    "figure",
    "work",
    "score",
    "scores",
    "scoring",
    "judge",
    "judging",
    "answers",
    "answer",
    "gold",
    "golden",
    "ground_truth",
    "reference_answers",
    "hidden",
    "secrets",
    "credentials",
    "评分",
    "答案",
    "金标",
    "裁判",
    ".git",
    ".aws",
    ".codex",
}
_SECRET_NAME = re.compile(r"secret|credential|password|passwd|api[_-]?key|access[_-]?token", re.I)
_CASE_ID = re.compile(r"^[A-E]-\d{2,}$")


def ensure_no_links(path: Path, anchor: Path) -> Path:
    """Reject symlinks, junctions and reparse points below the trusted anchor.

    A lexical containment check happens before resolution, followed by physical
    containment. Existing path components are checked individually, including
    the anchor; callers may safely use this for paths not created yet.
    """
    anchor = Path(os.path.abspath(anchor))
    target = Path(os.path.abspath(path))
    if not target.is_relative_to(anchor):
        raise PermissionError("path is outside this case's permitted directory")
    relative = target.relative_to(anchor)
    current = anchor
    for component in (None, *relative.parts):
        if component is not None:
            current = current / component
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise PermissionError("cannot verify the permitted path") from exc
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise PermissionError("symlink, junction and reparse-point paths are forbidden")
    resolved = target.resolve()
    if not resolved.is_relative_to(anchor.resolve()):
        raise PermissionError("physical path escapes its permitted directory")
    return target


def _relative_parts(value: str) -> tuple[str, ...]:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise PermissionError("use a nonempty relative path with '/' separators")
    pure = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if pure.is_absolute() or windows.drive or any(part in (".", "..") for part in value.split("/")):
        raise PermissionError("absolute and traversal paths are forbidden")
    if any(":" in part for part in pure.parts):
        raise PermissionError("alternate streams and drive paths are forbidden")
    return pure.parts


def _private_name(part: str) -> bool:
    lowered = part.casefold()
    return (
        lowered.startswith(".")
        or lowered in _PRIVATE
        or Path(lowered).stem in _PRIVATE
        or bool(_SECRET_NAME.search(lowered))
    )


class CasePolicy:
    """Bind shared PDF capabilities and a case-specific scratch directory."""

    def __init__(self, config: RuntimeConfig, case_id: str) -> None:
        if not _CASE_ID.fullmatch(case_id):
            raise ValueError("invalid current case_id")
        self.config = config
        self.case_id = case_id
        self.result_root = Path(os.path.abspath(config.result_root))
        self.data_root = Path(os.path.abspath(config.data_root))
        self.work_prefix = f"data/{config.team}/work/cases/{case_id}"
        self.case_work_dir = self.result_root.joinpath(*self.work_prefix.split("/"))

    def work_path(self, relative_path: str, *, for_write: bool = False) -> Path:
        parts = _relative_parts(relative_path)
        prefix = tuple(self.work_prefix.split("/"))
        if len(parts) <= len(prefix) or parts[: len(prefix)] != prefix:
            raise PermissionError("only this case's work/cases directory is accessible")
        tail = parts[len(prefix) :]
        if any(_private_name(part) for part in tail):
            raise PermissionError("hidden or credential-like work files are forbidden")
        if Path(parts[-1]).suffix.lower() not in TEXT_SUFFIXES:
            raise PermissionError("work files must use an allowed text extension")
        target = ensure_no_links(self.result_root.joinpath(*parts), self.result_root)
        if not for_write and (not target.is_file() or target.stat().st_size > MAX_TEXT_BYTES):
            raise ValueError("work file is missing, not a regular file, or too large")
        return target

    def shared_roots(self) -> list[Path]:
        candidates = [self.data_root / name for name in SHARED_DIRECTORIES]
        candidates += [self.result_root / "data" / name for name in SHARED_DIRECTORIES]
        roots: list[Path] = []
        for candidate in candidates:
            anchor = (
                self.data_root if candidate.is_relative_to(self.data_root) else self.result_root
            )
            if candidate.exists():
                safe = ensure_no_links(candidate, anchor)
                if safe.is_dir() and safe not in roots:
                    roots.append(safe)
        return roots

    def authorize_paper(self, path: Path) -> Path:
        """Authorize an already chosen PDF, not a model-supplied arbitrary path."""
        candidate = Path(os.path.abspath(path))
        roots = self.shared_roots()
        anchors = [root for root in roots if candidate.is_relative_to(root)]
        # Explicit PDF files mounted directly at DATA_PATH are permitted. A
        # directory there is never a whole-library fallback capability.
        if candidate.parent == self.data_root:
            anchors.append(self.data_root)
        if not anchors:
            raise PermissionError("PDF is outside the allowed shared-paper locations")
        anchor = anchors[0]
        safe = ensure_no_links(candidate, anchor)
        if any(_private_name(part) for part in candidate.relative_to(anchor).parts):
            raise PermissionError("private or credential-like material paths are forbidden")
        if safe.suffix.lower() != ".pdf" or not safe.is_file():
            raise ValueError("the specified PDF is missing or not a regular PDF file")
        return safe

    def _scan_shared(self, directory: Path) -> list[Path]:
        roots = self.shared_roots()
        if not any(directory.is_relative_to(root) for root in roots):
            raise PermissionError("only shared-paper directories may be scanned")
        anchor = next(root for root in roots if directory.is_relative_to(root))
        ensure_no_links(directory, anchor)
        result: list[Path] = []
        visited = 0
        for current, directories, files in os.walk(directory, followlinks=False):
            kept = []
            for name in sorted(directories):
                child = Path(current) / name
                if _private_name(name):
                    continue
                try:
                    ensure_no_links(child, anchor)
                except PermissionError:
                    continue
                kept.append(name)
            directories[:] = kept
            for name in sorted(files):
                visited += 1
                if visited > MAX_SCAN_ENTRIES or len(result) >= MAX_PAPERS:
                    return result
                if not name.lower().endswith(".pdf") or _private_name(name):
                    continue
                try:
                    result.append(self.authorize_paper(Path(current) / name))
                except (PermissionError, ValueError):
                    continue
        return result

    def material_files(self, materials: list[str], *, allow_library_fallback: bool) -> list[Path]:
        """Resolve descriptions before the broad legacy library sees any path."""
        result: list[Path] = []
        shared_index: list[Path] | None = None
        scan_cache: dict[tuple, list[Path]] = {}

        def scan(directory: Path) -> list[Path]:
            roots = self.shared_roots()
            anchors = [root for root in roots if directory.is_relative_to(root)]
            if not anchors:
                raise PermissionError("only shared-paper directories may be scanned")
            ensure_no_links(directory, anchors[0])
            key = location_key(directory)
            if key not in scan_cache:
                scan_cache[key] = self._scan_shared(directory)
            return scan_cache[key]

        def scan_libraries() -> list[Path]:
            # Keep all logical roots for authorization, but scan a physical
            # shared directory once when bind mounts expose it more than once.
            return [path for root in unique_locations(self.shared_roots()) for path in scan(root)]

        for material in materials:
            if not isinstance(material, str) or not material.strip():
                raise ValueError("materials must identify permitted PDF files")
            raw = material.strip().strip("\"'")
            if "\\" in raw or "\x00" in raw or ".." in raw.split("/"):
                raise PermissionError("unsafe material path")
            windows = PureWindowsPath(raw)
            absolute = Path(raw).is_absolute()
            if windows.drive and not absolute:
                raise PermissionError("foreign drive paths are forbidden")
            if any(_private_name(part) for part in PurePosixPath(raw).parts if part != "/"):
                raise PermissionError("private material locations are forbidden")
            candidates = (
                [Path(raw)]
                if absolute
                else [
                    self.data_root / raw,
                    self.result_root / raw,
                    self.result_root / "data" / raw,
                ]
            )
            if raw.startswith("data/"):
                candidates.append(self.data_root.joinpath(*PurePosixPath(raw).parts[1:]))
            # A relative material may be rooted at the workspace's data/
            # directory. Do not silently select one of multiple real files.
            existing = list(dict.fromkeys(
                Path(os.path.abspath(candidate)) for candidate in candidates if candidate.exists()
            ))
            if existing and all(path.is_file() for path in existing):
                existing = unique_pdfs(
                    [self.authorize_paper(path) for path in existing],
                    conflict_message="the specified material path is ambiguous",
                )
            elif existing and all(path.is_dir() for path in existing):
                existing = unique_locations(existing)
                if len(existing) > 1 and allow_library_fallback:
                    for directory in existing:
                        result.extend(scan(directory))
                    continue
            if len(existing) > 1:
                raise ValueError("the specified material path is ambiguous")
            chosen = existing[0] if existing else None
            if chosen is not None:
                if chosen.is_dir():
                    result.extend(scan(Path(os.path.abspath(chosen))))
                else:
                    result.append(self.authorize_paper(chosen))
                continue
            # Logical library paths in the task may differ from the platform's
            # mount name. Keep the entire relative suffix, including the exact
            # PDF filename; never substitute by basename or scan another team.
            parts = PurePosixPath(raw).parts
            logical_parts = parts[1:] if parts and parts[0] == "data" else parts
            if (
                not absolute and len(logical_parts) > 1
                and logical_parts[0] in SHARED_DIRECTORIES
                and raw.casefold().endswith(".pdf")
            ):
                matches = []
                for root in self.shared_roots():
                    candidate = root.joinpath(*logical_parts[1:])
                    if candidate.exists():
                        matches.append(self.authorize_paper(candidate))
                matches = unique_pdfs(
                    matches, conflict_message="the specified PDF library path is ambiguous",
                )
                if matches:
                    result.append(matches[0])
                    continue
            # Only B may interpret a whole-library alias as an unrestricted
            # query over the one shared library. C/E still require their PDFs.
            aliases = {(name,) for name in SHARED_DIRECTORIES} | {
                ("data", name) for name in SHARED_DIRECTORIES
            }
            if allow_library_fallback and not absolute and parts in aliases:
                roots = self.shared_roots()
                if roots:
                    result.extend(scan_libraries())
                    continue
            if "/" not in raw and raw.lower().endswith(".pdf"):
                if shared_index is None:
                    shared_index = scan_libraries()
                matches = [path for path in shared_index if path.name.casefold() == raw.casefold()]
                matches = unique_pdfs(
                    matches, conflict_message="the specified PDF filename is ambiguous",
                )
                if matches:
                    result.append(matches[0])
                    continue
            explicit_file = raw.casefold().endswith(".pdf") or "/" in raw
            if explicit_file or not allow_library_fallback:
                raise ValueError("a specified PDF material could not be resolved")
        if not result and allow_library_fallback:
            result = scan_libraries()
        return unique_pdfs(result, conflict_message="duplicate PDF filenames in the allowed materials")
