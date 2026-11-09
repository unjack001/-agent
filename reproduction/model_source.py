"""Extract a complete model response without rewriting its Python source.

This is an offline format check, not a platform source credential.
"""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path
from typing import Any

FENCE_RE = re.compile(
    r"^```(?:python|py)?[ \t]*\r?\n(.*?)^```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)


def _normalize(source: str) -> str:
    if not isinstance(source, str) or not 0 < len(source.encode("utf-8")) <= 65536:
        raise ValueError("Source must be 1..65536 UTF-8 bytes")
    source = source.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n") + "\n"
    tree = ast.parse(source, feature_version=(3, 9))
    if not any(
        isinstance(node, ast.FunctionDef) and node.name == "run_experiment"
        for node in tree.body
    ):
        raise ValueError("Missing run_experiment(request)")
    return source


def extract_source(response: dict[str, Any]) -> str:
    if not isinstance(response, dict):
        raise ValueError("Expected a complete chat/completions JSON response")
    found: set[str] = set()
    for choice in response.get("choices", []):
        if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
            continue
        message = choice.get("message", {})
        if message.get("role") != "assistant" or not isinstance(message.get("content"), str):
            continue
        content = message["content"]
        candidates = [content]
        candidates.extend(FENCE_RE.findall(content))
        for candidate in candidates:
            try:
                found.add(_normalize(candidate))
            except (SyntaxError, ValueError, RecursionError):
                pass
    if len(found) != 1:
        raise ValueError(
            "Expected exactly one complete Python implementation with "
            "finish_reason=stop; request a full new response"
        )
    return found.pop()


def save_source(response: dict[str, Any], destination: str | Path) -> str:
    source = extract_source(response)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(source.encode("utf-8"))
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def diagnose(response: dict[str, Any]) -> str:
    """Describe why a response has no unique usable implementation."""
    if not isinstance(response, dict):
        return "response is not a JSON object"
    errors: list[str] = []
    choices = response.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return "response has no choices"
    for choice in choices:
        if not isinstance(choice, dict):
            errors.append("choice is not an object")
            continue
        finish = choice.get("finish_reason")
        if finish != "stop":
            errors.append(f"finish_reason={finish!r}")
        message = choice.get("message", {})
        if not isinstance(message, dict):
            errors.append("message is not an object")
            continue
        if message.get("role") != "assistant":
            errors.append(f"role={message.get('role')!r}")
        content = message.get("content")
        if not isinstance(content, str):
            errors.append("content is not a string")
            continue
        candidates = [content]
        candidates.extend(FENCE_RE.findall(content))
        for candidate in candidates:
            try:
                _normalize(candidate)
            except (SyntaxError, ValueError, RecursionError) as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
    unique = list(dict.fromkeys(errors))
    return "; ".join(unique)[:2000] or "no usable Python implementation"
