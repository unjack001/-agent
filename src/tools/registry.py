"""Custom tools exposed to the agent loop; every executed tool uses @tool."""

from __future__ import annotations

from pathlib import Path

from agenthack import tool

from src.materials.pdf import extract_pages


@tool(name="pdf_extract_pages")
def pdf_extract_pages(pdf_path: Path, max_pages: int | None = None) -> list[str]:
    return extract_pages(pdf_path, max_pages=max_pages)
