"""Pure parsing helpers for LLM responses."""

from __future__ import annotations

import json
import math
from typing import Any


class LLMError(RuntimeError):
    """Raised when a gateway response cannot be used."""


class LLMTruncatedError(LLMError):
    """A token-limited reply was discarded and may be regenerated safely."""

    def __init__(self, message: str, *, response_body: dict[str, Any] | None = None):
        super().__init__(message)
        self.response_body = response_body


class LLMOutputError(LLMError):
    """The model completed a reply that does not satisfy the requested format."""


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate JSON field")
        result[name] = value
    return result


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite JSON number")
    return result


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from a reply that may contain prose or a code fence."""
    if not isinstance(text, str) or not text.strip():
        raise LLMOutputError("empty LLM response")

    raw = text.strip()
    candidates: list[str] = [raw]

    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidates.append("\n".join(lines).strip())

    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        candidates.append(raw[start : end + 1])

    for candidate in candidates:
        try:
            data = json.loads(
                candidate, parse_constant=_reject_constant, parse_float=_finite_float,
                object_pairs_hook=_unique_object,
            )
        except (ValueError, TypeError):
            continue
        if isinstance(data, dict):
            return data
        raise LLMOutputError("model reply must be a JSON object")
    raise LLMOutputError("response does not contain an unambiguous finite JSON object")
