"""Stdlib-only data models shared by materials and handlers."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


class MaterialError(RuntimeError):
    """Raised when allowed materials cannot be resolved or parsed."""


def normalize_evidence_text(text: str) -> str:
    """Discard invisible PDF control/format characters without inventing text."""
    return "".join(
        char
        for char in text
        if char in "\n\r\t" or not unicodedata.category(char).startswith("C")
    ).strip()


def has_evidence_text(text: str, *, filename: str, title: str) -> bool:
    """Reject empty extraction and metadata standing in for paper evidence.

    A short extracted sentence is still usable: a minimum character count would
    incorrectly reject concise abstracts and non-English text. A lone title
    without sentence punctuation, however, cannot support a research claim.
    """
    cleaned = normalize_evidence_text(text)
    if not any(char.isalnum() for char in cleaned):
        return False
    normalized = " ".join(cleaned.split()).casefold()
    metadata = {
        " ".join(normalize_evidence_text(value).split()).casefold()
        for value in (filename, Path(filename).stem)
    }
    if normalized in metadata:
        return False
    normalized_title = " ".join(normalize_evidence_text(title).split()).casefold()
    if normalized == normalized_title and not re.search(r"[.;!?。；！？][\"'”’）)]*$", cleaned):
        return False
    return True


@dataclass(frozen=True)
class PaperRef:
    filename: str
    path: Path


@dataclass(frozen=True)
class PaperCatalogEntry:
    filename: str
    title: str
    snippet: str


@dataclass(frozen=True)
class PaperExcerpt:
    page: int
    text: str


@dataclass(frozen=True)
class PaperDigest:
    filename: str
    title: str
    excerpts: tuple[PaperExcerpt, ...]
