"""Common contract and shared dependencies for the five category handlers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src.config import RuntimeConfig
from src.dispatch.models import CaseQuestion
from src.materials.models import PaperCatalogEntry, PaperDigest, PaperRef


class HandlerError(RuntimeError):
    """Raised when a case cannot produce a contract-valid answer."""


class LLMClient(Protocol):
    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.2,
    ) -> str: ...

    async def complete_with_body(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.2,
    ) -> tuple[str, dict[str, Any]]: ...

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        attempts: int = 1,
    ) -> dict[str, Any]: ...


class PaperSource(Protocol):
    def set_workspace_root(self, root: Path) -> None: ...

    def resolve_materials(
        self,
        materials: list[str],
        *,
        allow_library_fallback: bool,
        strict_names: bool = True,
    ) -> list[PaperRef]: ...

    def catalog(
        self,
        query: str,
        papers: list[PaperRef] | None = None,
        max_papers: int = 150,
    ) -> list[PaperCatalogEntry]: ...

    def digests(
        self,
        query: str,
        papers: list[PaperRef],
        *,
        max_pages_per_paper: int = 4,
    ) -> list[PaperDigest]: ...


@dataclass
class HandlerContext:
    config: RuntimeConfig
    llm: LLMClient
    library: PaperSource


class CaseHandler(ABC):
    function: str

    def __init__(self, context: HandlerContext) -> None:
        self.context = context

    @abstractmethod
    async def handle(self, question: CaseQuestion) -> dict[str, Any]:
        """Return an answer matching question.output_schema_ref."""
