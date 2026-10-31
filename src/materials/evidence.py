"""Evidence ledger: paper -> fact -> page -> report section."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceItem:
    paper_filename: str
    fact: str
    section: str
    page: int | None = None
