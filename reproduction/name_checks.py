"""Static name resolution only: never import or execute generated code."""
from __future__ import annotations

import ast
from pyflakes.checker import Checker
from pyflakes.messages import UndefinedName, UndefinedLocal


def undefined_name_issues(source: str) -> list[str]:
    checker = Checker(ast.parse(source), filename='implementation.py')
    return [str(message) for message in sorted(checker.messages, key=lambda m: m.lineno)
            if isinstance(message, (UndefinedName, UndefinedLocal))]
