"""Recognize one shared PDF exposed through multiple authorized mounts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


def location_key(path: Path) -> tuple:
    """Bind mounts can have different paths but share device/inode identity."""
    info = path.stat()
    if info.st_ino:
        return (info.st_dev, info.st_ino)
    return (str(path.resolve()),)


def unique_locations(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[tuple] = set()
    for path in paths:
        key = location_key(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def unique_pdfs(paths: Iterable[Path], *, conflict_message: str) -> list[Path]:
    """Collapse equivalent filenames only when their file/content is equal.

    Some volume drivers give aliases different inode numbers. Equal size plus
    a streaming SHA256 comparison handles those copies without trusting names.
    Different filenames retain separate identities even if their bytes match.
    Callers must authorize all paths before this function reads any content.
    """
    result: list[Path] = []
    by_name: dict[str, Path] = {}
    digests: dict[Path, bytes] = {}

    def digest(path: Path) -> bytes:
        if path not in digests:
            with path.open("rb") as stream:
                digests[path] = hashlib.file_digest(stream, "sha256").digest()
        return digests[path]

    for path in paths:
        key = path.name.casefold()
        previous = by_name.get(key)
        if previous is None:
            by_name[key] = path
            result.append(path)
            continue
        if path == previous or path.samefile(previous):
            continue
        if path.stat().st_size == previous.stat().st_size and digest(path) == digest(previous):
            continue
        raise ValueError(f"{conflict_message}: {path.name}")
    return result
