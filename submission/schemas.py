"""Required output fields and report field-type checks."""

from pathlib import PurePosixPath
from typing import Any

FUNCTION_BY_CASE_PREFIX: dict[str, str] = {
    "A": "Chat",
    "B": "Literature_Retrieval",
    "C": "Survey",
    "D": "Reproduction",
    "E": "Idea",
}

OUTPUT_FIELDS: dict[str, tuple[str, ...]] = {
    "Chat": ("case_id", "answer", "clarifying_question"),
    "Literature_Retrieval": ("case_id", "papers"),
    "Survey": ("case_id", "path", "markdown"),
    "Reproduction": ("case_id", "status", "metrics", "figures"),
    "Idea": ("case_id", "path", "markdown"),
}

REPRODUCTION_STATUSES = ("completed", "failed", "materials_missing")


class OutputSchemaError(ValueError):
    """Safe field diagnostics; never includes a report or request body."""

    def __init__(self, issues: list[str]) -> None:
        self.issues = [item.encode("utf-8", "backslashreplace").decode("utf-8") for item in issues]
        super().__init__("; ".join(self.issues))


def validate_report_payload(
    payload: Any, case_id: str, *, expected_path: str | None = None,
) -> None:
    """Check only the fields, non-empty text, UTF-8 and assigned report path."""
    if not isinstance(payload, dict):
        raise OutputSchemaError(["output must be a JSON object"])
    issues: list[str] = []
    fields = set(OUTPUT_FIELDS["Idea"])
    missing = fields - set(payload)
    extra = set(payload) - fields
    if missing:
        issues.append("missing required fields: " + ", ".join(sorted(missing)))
    if extra:
        issues.append("unexpected fields: " + ", ".join(sorted(map(str, extra))))
    for field in OUTPUT_FIELDS["Idea"]:
        if field not in payload:
            continue
        value = payload[field]
        if not isinstance(value, str):
            issues.append(f"{field} must be a string, got {type(value).__name__}")
        elif not value.strip():
            issues.append(f"{field} must be non-empty")
        else:
            try:
                value.encode("utf-8")
            except UnicodeEncodeError:
                issues.append(f"{field} must contain valid UTF-8 text")
    if isinstance(payload.get("case_id"), str) and payload["case_id"] != case_id:
        issues.append("case_id must equal the assigned case_id")
    path = payload.get("path")
    if isinstance(path, str):
        pure = PurePosixPath(path)
        if (
            pure.is_absolute() or ".." in pure.parts or "\\" in path
            or len(pure.parts) != 3 or pure.parts[:2] != ("data", "expert")
            or not pure.name.startswith(case_id + "-") or not pure.name.endswith(".md")
        ):
            issues.append("path must be the current case's relative data/expert Markdown path")
        if expected_path is not None and path != expected_path:
            issues.append("path must equal the assigned team's report path")
    if issues:
        raise OutputSchemaError(issues)
