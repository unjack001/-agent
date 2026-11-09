"""Typed views of dispatch.json and assigned question files."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DispatchCase:
    case_id: str
    function: str
    title: str
    question_file: str
    question_sha256: str
    difficulty: str | None = None


@dataclass(frozen=True)
class DispatchRecord:
    dispatch_id: str
    team: str
    schema_version: int
    case_ids: list[str]
    cases: list[DispatchCase]
    cases_per_category: dict[str, int]
    raw: dict[str, Any]


@dataclass(frozen=True)
class CaseQuestion:
    case_id: str
    function: str
    title: str
    input: str
    materials: list[str]
    constraints: list[str]
    expected_output: str
    output_schema_ref: str
    raw: dict[str, Any]
