"""PDF text/page extraction for allowed papers."""

from __future__ import annotations

from pathlib import Path

from src.materials.models import MaterialError

MAX_TOTAL_CHARS = 500_000


def extract_pages(pdf_path: Path, max_pages: int | None = None) -> list[str]:
    """Return one stripped text string per page; empty list for unreadable PDFs."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise MaterialError("pypdf is not installed in the agent image") from exc

    try:
        reader = PdfReader(str(pdf_path))
        if reader.is_encrypted:
            try:
                if reader.decrypt("") == 0:
                    return []
            except Exception:
                return []

        pages: list[str] = []
        total = 0
        for index, page in enumerate(reader.pages):
            if max_pages is not None and index >= max_pages:
                break
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            text = text.strip()
            pages.append(text)
            total += len(text)
            if total >= MAX_TOTAL_CHARS:
                break
        return pages
    except Exception as exc:
        raise MaterialError(f"cannot read PDF {pdf_path}: {exc}") from exc
