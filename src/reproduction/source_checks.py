"""Pure static diagnostics for model-generated experiment implementations.

This module reads and parses source text. It never imports, evaluates or runs
that source. Only source size, UTF-8, Python 3.9 syntax and the callable entrypoint
are blocking checks here. The older capability heuristics are advisory: their
numeric-module list is not a platform requirement. Only the trusted formal
platform may execute a model-generated implementation.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any

_NUMERIC_IMPORTS = {
    "math", "statistics", "random", "collections", "heapq", "bisect", "itertools",
    "functools", "operator", "decimal", "fractions", "array", "re", "json",
    "typing", "dataclasses", "enum", "time", "__future__",
}
_UNSAFE_NAMES = {
    "open", "eval", "exec", "compile", "__import__", "getattr", "setattr", "delattr",
    "globals", "locals", "vars", "dir", "input", "breakpoint", "help", "type",
    "os", "sys", "pathlib", "socket", "subprocess", "importlib", "builtins",
    "attrgetter", "methodcaller", "format", "format_map", "Formatter", "get_field",
}


def _local_source_guard(source: str) -> None:
    """Legacy heuristic retained only for best-effort advisory diagnostics.

    Its allowlist is narrower than the permitted Python standard library and
    must not prevent a compliant source from reaching the isolated executor.
    Passing this heuristic never authorizes execution inside the Agent process.
    """
    tree = ast.parse(source, feature_version=(3, 9))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] not in _NUMERIC_IMPORTS for alias in node.names):
                raise ValueError("static preflight allows only the numeric standard-library allowlist")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in _NUMERIC_IMPORTS:
                raise ValueError("static preflight rejects non-allowlisted imports")
            if any(alias.name == "*" or alias.name.startswith("_") or alias.name in _UNSAFE_NAMES
                   for alias in node.names):
                raise ValueError("static preflight rejects private or reflective imports")
        elif isinstance(node, ast.Name) and node.id in _UNSAFE_NAMES:
            raise ValueError("static preflight rejects file, process and reflection capabilities")
        elif isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in _UNSAFE_NAMES):
            raise ValueError("static preflight rejects private or reflective attributes")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            raise ValueError("static preflight rejects dynamic dunder lookup")


def check_source_file(path: Path) -> dict[str, Any]:
    """Check the transport/entrypoint contract without executing or rewriting."""
    data = Path(path).read_bytes()
    if not 0 < len(data) <= 65_536:
        raise ValueError("static preflight source must contain 1..65536 UTF-8 bytes")
    source = data.decode("utf-8")
    tree = ast.parse(source, feature_version=(3, 9))
    entrypoints = [node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name == "run_experiment"]
    if len(entrypoints) != 1:
        raise ValueError("static preflight requires one run_experiment(request) entrypoint")
    signature = entrypoints[0].args
    positional = signature.posonlyargs + signature.args
    if len(positional) != 1 or any(default is None for default in signature.kw_defaults):
        raise ValueError("static preflight requires a callable single-request entrypoint")
    warnings: list[str] = []
    try:
        _local_source_guard(source)
    except ValueError as exc:
        # This historic heuristic rejects valid constructs such as copy,
        # type(request), and str.format. It is not the experiment protocol.
        warnings.append(str(exc))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
    return {
        "check": "static_only", "executed": False,
        "source_sha256": hashlib.sha256(data).hexdigest(),
        "entrypoint": "run_experiment", "syntax": "python3.9", "imports": sorted(imports),
        "legacy_heuristic_warnings": warnings, "legacy_heuristic_blocking": False,
        "message": "Source was parsed only. Public sample correctness and formal metrics are unverified.",
    }
