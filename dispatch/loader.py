"""Locate dispatch.json and load the 11 assigned question files."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from src.config import RuntimeConfig
from src.dispatch.models import CaseQuestion, DispatchCase, DispatchRecord
from src.submission.schemas import FUNCTION_BY_CASE_PREFIX, OUTPUT_FIELDS

CASE_ID_PATTERN = re.compile(r"^[A-E]-\d{2,}$")
FUNCTIONS = tuple(FUNCTION_BY_CASE_PREFIX.values())
FORMAL_CASES_PER_CATEGORY = {"A": 1, "B": 3, "C": 2, "D": 3, "E": 2}
FORMAL_CASE_COUNT = sum(FORMAL_CASES_PER_CATEGORY.values())
FORMAL_RULES_VERSION = "research-assistant-objective-v3"


class DispatchError(RuntimeError):
    """Raised when the dispatch record or an assigned question is invalid."""


class DispatchNotFound(DispatchError):
    """The external direct-question interface did not provide dispatch files."""


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


class DispatchLoader:
    """Resolves the workspace root and exposes dispatch/question records."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self._workspace_root: Path | None = None
        self._dispatch_path: Path | None = None

    def workspace_root(self) -> Path:
        """Return the directory that contains data/<team>/dispatch.json."""
        return self._locate()[0]

    def dispatch_path(self) -> Path:
        return self._locate()[1]

    def load(self) -> DispatchRecord:
        """Read dispatch.json and validate team, counts and case ids."""
        _, path = self._locate()
        raw = self._read_json_object(path, "dispatch record")

        team = self._require_str(raw, "team", path)
        if team != self.config.team:
            raise DispatchError(
                f"{path}: dispatch team {team!r} does not match "
                f"RESEARCH_WORKSPACE_TEAM {self.config.team!r}"
            )

        dispatch_id = self._require_str(raw, "dispatch_id", path)
        case_ids = self._parse_case_ids(raw, path)
        cases = self._parse_cases(raw, team, path)
        self._check_case_alignment(case_ids, cases, path)

        schema_version = self._parse_schema_version(raw, path)
        if schema_version not in (0, 1, 2, 3):
            raise DispatchError(f"{path}: unsupported schema_version {schema_version}")
        if schema_version == 3:
            self._validate_v3(raw, case_ids, cases, path)

        cases_per_category = self._parse_cases_per_category(raw, path)
        by_id = {case.case_id: case for case in cases}

        return DispatchRecord(
            dispatch_id=dispatch_id,
            team=team,
            schema_version=schema_version,
            case_ids=case_ids,
            cases=[by_id[case_id] for case_id in case_ids],
            cases_per_category=cases_per_category,
            raw=raw,
        )

    def load_question(self, case: DispatchCase) -> CaseQuestion:
        """Read one assigned question file and verify its case_id and hash."""
        expected_file = f"data/{self.config.team}/assigned/{case.case_id}.json"
        if case.question_file != expected_file:
            raise DispatchError(
                f"{case.case_id}: question_file {case.question_file!r} must be "
                f"{expected_file!r}"
            )

        path = self._question_path(self.workspace_root(), case.question_file)
        data = self._read_bytes(path, "question file")

        if case.question_sha256:
            actual = hashlib.sha256(data).hexdigest()
            if actual.lower() != case.question_sha256.lower():
                raise DispatchError(
                    f"{path}: question_sha256 mismatch "
                    f"(expected {case.question_sha256}, got {actual})"
                )

        raw = self._decode_json(data, path)
        if not isinstance(raw, dict):
            raise DispatchError(f"{path}: question file must contain a JSON object")

        case_id = self._require_str(raw, "case_id", path)
        if case_id != case.case_id:
            raise DispatchError(
                f"{path}: case_id {case_id!r} does not match dispatch {case.case_id!r}"
            )

        function = self._require_str(raw, "function", path)
        if function not in FUNCTIONS:
            raise DispatchError(f"{path}: unsupported function {function!r}")
        if function != case.function:
            raise DispatchError(
                f"{path}: function {function!r} does not match dispatch {case.function!r}"
            )

        output_schema_ref = self._require_str(raw, "output_schema_ref", path)
        if output_schema_ref not in OUTPUT_FIELDS:
            raise DispatchError(f"{path}: unsupported output_schema_ref {output_schema_ref!r}")
        if output_schema_ref != function:
            raise DispatchError(
                f"{path}: output_schema_ref {output_schema_ref!r} does not match "
                f"function {function!r}"
            )

        title = raw.get("title")
        if not isinstance(title, str):
            raise DispatchError(f"{path}: field 'title' must be a string")

        task_input = raw.get("input")
        if not isinstance(task_input, str) or not task_input.strip():
            raise DispatchError(f"{path}: field 'input' must be a non-empty string")

        materials = raw.get("materials")
        if not isinstance(materials, list) or any(not isinstance(item, str) for item in materials):
            raise DispatchError(f"{path}: field 'materials' must be a list of strings")

        constraints = raw.get("constraints")
        if not isinstance(constraints, list) or any(
            not isinstance(item, str) for item in constraints
        ):
            raise DispatchError(f"{path}: field 'constraints' must be a list of strings")

        expected_output = raw.get("expected_output")
        if not isinstance(expected_output, str):
            raise DispatchError(f"{path}: field 'expected_output' must be a string")

        return CaseQuestion(
            case_id=case_id,
            function=function,
            title=title,
            input=task_input,
            materials=materials,
            constraints=constraints,
            expected_output=expected_output,
            output_schema_ref=output_schema_ref,
            raw=raw,
        )

    def load_question_by_id(self, record: DispatchRecord, case_id: str) -> CaseQuestion:
        for case in record.cases:
            if case.case_id == case_id:
                return self.load_question(case)
        raise DispatchError(f"case {case_id!r} is not in the dispatch record")

    def verify_direct_question(self, question: CaseQuestion) -> None:
        """When dispatch files are mounted, bind the direct task to that round.

        A platform may send all eight fields without mounting dispatch files.
        Absence is compatible; a present but invalid record must not be ignored.
        """
        try:
            record = self.load()
        except DispatchNotFound:
            return
        assigned = self.load_question_by_id(record, question.case_id)
        for field in (
            "case_id", "function", "title", "input", "materials", "constraints",
            "expected_output", "output_schema_ref",
        ):
            if getattr(question, field) != getattr(assigned, field):
                raise DispatchError(
                    f"{question.case_id}: direct field {field!r} differs from assigned question"
                )

    def parse_question(self, raw: dict[str, Any]) -> CaseQuestion:
        """Validate an inline eight-field question sent directly by the platform."""
        label = Path("<direct-question>")
        if not isinstance(raw, dict):
            raise DispatchError("inline question must be a JSON object")

        case_id = self._require_str(raw, "case_id", label)
        if not CASE_ID_PATTERN.fullmatch(case_id):
            raise DispatchError(f"inline question has invalid case_id {case_id!r}")
        function = self._require_str(raw, "function", label)
        if function not in FUNCTIONS:
            raise DispatchError(f"inline question has unsupported function {function!r}")
        expected_function = FUNCTION_BY_CASE_PREFIX[case_id[0]]
        if function != expected_function:
            raise DispatchError(
                f"inline question {case_id} must use function {expected_function!r}, "
                f"not {function!r}"
            )

        output_schema_ref = self._require_str(raw, "output_schema_ref", label)
        if output_schema_ref not in OUTPUT_FIELDS:
            raise DispatchError(
                f"inline question has unsupported output_schema_ref {output_schema_ref!r}"
            )
        if output_schema_ref != function:
            raise DispatchError(
                f"inline output_schema_ref {output_schema_ref!r} does not match "
                f"function {function!r}"
            )

        title = raw.get("title")
        if not isinstance(title, str):
            raise DispatchError("inline question field 'title' must be a string")
        task_input = raw.get("input")
        if not isinstance(task_input, str) or not task_input.strip():
            raise DispatchError("inline question field 'input' must be a non-empty string")
        materials = raw.get("materials")
        if not isinstance(materials, list) or any(
            not isinstance(item, str) for item in materials
        ):
            raise DispatchError("inline question field 'materials' must be a list of strings")
        constraints = raw.get("constraints")
        if not isinstance(constraints, list) or any(
            not isinstance(item, str) for item in constraints
        ):
            raise DispatchError("inline question field 'constraints' must be a list of strings")
        expected_output = raw.get("expected_output")
        if not isinstance(expected_output, str):
            raise DispatchError("inline question field 'expected_output' must be a string")

        return CaseQuestion(
            case_id=case_id,
            function=function,
            title=title,
            input=task_input,
            materials=materials,
            constraints=constraints,
            expected_output=expected_output,
            output_schema_ref=output_schema_ref,
            raw=raw,
        )

    def _locate(self) -> tuple[Path, Path]:
        if self._workspace_root is not None and self._dispatch_path is not None:
            return self._workspace_root, self._dispatch_path

        team = self.config.team.strip()
        if not team:
            raise DispatchError("RESEARCH_WORKSPACE_TEAM is not set")

        candidates: list[tuple[Path, Path]] = []
        seen: set[tuple[str, str]] = set()

        def add(root: Path, path: Path) -> None:
            key = (str(root), str(path))
            if key not in seen:
                seen.add(key)
                candidates.append((root, path))

        roots = [self.config.result_root, self.config.data_root, Path.cwd(), Path.cwd().parent]
        for root in roots:
            add(root, root / "data" / team / "dispatch.json")

        # Some platforms mount the team directory directly, e.g.
        # /data/<team>/dispatch.json; the workspace root is then its parent.
        add(self.config.data_root.parent, self.config.data_root / team / "dispatch.json")

        cwd = Path.cwd()
        if cwd.name == team:
            root = cwd.parent.parent if cwd.parent.name == "data" else cwd.parent
            add(root, cwd / "dispatch.json")

        for root, path in candidates:
            try:
                if not path.is_file():
                    continue
                resolved_root = root.resolve()
                resolved_path = path.resolve()
            except OSError:
                continue
            if not resolved_path.is_relative_to(resolved_root):
                continue
            self._workspace_root = resolved_root
            self._dispatch_path = resolved_path
            return resolved_root, resolved_path

        searched = ", ".join(str(path) for _, path in candidates)
        raise DispatchNotFound(f"dispatch.json not found for team {team!r}; searched: {searched}")

    def _question_path(self, root: Path, question_file: str) -> Path:
        if "\\" in question_file:
            raise DispatchError(f"question_file {question_file!r} must use '/' separators")
        pure = PurePosixPath(question_file)
        if pure.is_absolute() or ".." in pure.parts:
            raise DispatchError(f"question_file {question_file!r} is not a safe relative path")
        target = root.joinpath(*pure.parts)
        try:
            resolved = target.resolve()
            resolved_root = root.resolve()
        except OSError as exc:
            raise DispatchError(f"cannot resolve question file {question_file}: {exc}") from exc
        if not resolved.is_relative_to(resolved_root):
            raise DispatchError(f"question file {resolved} escapes workspace root {resolved_root}")
        return resolved

    def _parse_case_ids(self, raw: dict[str, Any], path: Path) -> list[str]:
        value = raw.get("case_ids")
        if not isinstance(value, list) or not value:
            raise DispatchError(f"{path}: field 'case_ids' must be a non-empty list")
        case_ids: list[str] = []
        for item in value:
            if not isinstance(item, str) or not CASE_ID_PATTERN.fullmatch(item):
                raise DispatchError(f"{path}: invalid case_id {item!r} in case_ids")
            case_ids.append(item)
        if len(set(case_ids)) != len(case_ids):
            raise DispatchError(f"{path}: case_ids contains duplicates")
        return case_ids

    def _parse_cases(self, raw: dict[str, Any], team: str, path: Path) -> list[DispatchCase]:
        value = raw.get("cases")
        if not isinstance(value, list) or not value:
            raise DispatchError(f"{path}: field 'cases' must be a non-empty list")

        cases: list[DispatchCase] = []
        for item in value:
            if not isinstance(item, dict):
                raise DispatchError(f"{path}: every entry in 'cases' must be an object")
            case_id = self._require_str(item, "case_id", path)
            if not CASE_ID_PATTERN.fullmatch(case_id):
                raise DispatchError(f"{path}: invalid case_id {case_id!r}")
            function = self._require_str(item, "function", path)
            if function not in FUNCTIONS:
                raise DispatchError(f"{path}: {case_id} has unsupported function {function!r}")
            expected_function = FUNCTION_BY_CASE_PREFIX[case_id[0]]
            if function != expected_function:
                raise DispatchError(
                    f"{path}: {case_id} must use function {expected_function!r}, not {function!r}"
                )
            title = self._require_str(item, "title", path)
            question_file = self._require_str(item, "question_file", path)
            self._validate_question_file(question_file, team, case_id, path)

            question_sha256 = item.get("question_sha256")
            if question_sha256 is None:
                question_sha256 = ""
            elif not isinstance(question_sha256, str):
                raise DispatchError(f"{path}: {case_id} question_sha256 must be a string")
            elif question_sha256 and not re.fullmatch(r"[0-9a-fA-F]{64}", question_sha256):
                raise DispatchError(f"{path}: {case_id} has an invalid question_sha256")

            difficulty = item.get("difficulty")
            if difficulty is not None and not isinstance(difficulty, str):
                raise DispatchError(f"{path}: {case_id} difficulty must be a string")

            cases.append(
                DispatchCase(
                    case_id=case_id,
                    function=function,
                    title=title,
                    question_file=question_file,
                    question_sha256=question_sha256.lower(),
                    difficulty=difficulty,
                )
            )
        return cases

    def _validate_question_file(
        self,
        value: str,
        team: str,
        case_id: str,
        path: Path,
    ) -> None:
        if "\\" in value:
            raise DispatchError(f"{path}: {case_id} question_file must use '/' separators")
        pure = PurePosixPath(value)
        if pure.is_absolute() or ".." in pure.parts:
            raise DispatchError(f"{path}: {case_id} question_file is not a safe relative path")
        expected = ("data", team, "assigned", f"{case_id}.json")
        if pure.parts != expected:
            raise DispatchError(
                f"{path}: {case_id} question_file must be data/{team}/assigned/{case_id}.json"
            )

    def _check_case_alignment(
        self,
        case_ids: list[str],
        cases: list[DispatchCase],
        path: Path,
    ) -> None:
        if len(cases) != len(case_ids):
            raise DispatchError(
                f"{path}: case_ids has {len(case_ids)} entries but cases has {len(cases)}"
            )
        if {case.case_id for case in cases} != set(case_ids):
            raise DispatchError(f"{path}: cases do not match case_ids one-to-one")

    def _parse_schema_version(self, raw: dict[str, Any], path: Path) -> int:
        value = raw.get("schema_version")
        if value is None:
            return 0
        if isinstance(value, bool) or not isinstance(value, int):
            raise DispatchError(f"{path}: schema_version must be an integer")
        return value

    def _validate_v3(
        self,
        raw: dict[str, Any],
        case_ids: list[str],
        cases: list[DispatchCase],
        path: Path,
    ) -> None:
        selection_mode = raw.get("selection_mode")
        if not isinstance(selection_mode, str) or not selection_mode.strip():
            raise DispatchError(f"{path}: v3 requires a non-empty selection_mode")
        if raw.get("rules_version") != FORMAL_RULES_VERSION:
            raise DispatchError(
                f"{path}: rules_version must be {FORMAL_RULES_VERSION!r} for schema_version 3"
            )
        if raw.get("cases_per_category") != FORMAL_CASES_PER_CATEGORY:
            raise DispatchError(
                f"{path}: cases_per_category must be {FORMAL_CASES_PER_CATEGORY} for v3"
            )
        if len(case_ids) != FORMAL_CASE_COUNT:
            raise DispatchError(
                f"{path}: v3 requires {FORMAL_CASE_COUNT} cases, got {len(case_ids)}"
            )
        counts = Counter(case_id[0] for case_id in case_ids)
        if dict(counts) != FORMAL_CASES_PER_CATEGORY:
            raise DispatchError(
                f"{path}: v3 categories must be {FORMAL_CASES_PER_CATEGORY}, got {dict(counts)}"
            )
        for case in cases:
            if not case.question_sha256:
                raise DispatchError(f"{path}: v3 requires question_sha256 for {case.case_id}")
            if case.case_id.startswith("D-") and not case.difficulty:
                raise DispatchError(f"{path}: v3 requires difficulty for {case.case_id}")

    def _parse_cases_per_category(self, raw: dict[str, Any], path: Path) -> dict[str, int]:
        value = raw.get("cases_per_category")
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise DispatchError(f"{path}: cases_per_category must be an object")
        parsed: dict[str, int] = {}
        for key, count in value.items():
            if not isinstance(key, str) or isinstance(count, bool) or not isinstance(count, int):
                raise DispatchError(f"{path}: cases_per_category must map strings to integers")
            parsed[key] = count
        return parsed

    def _read_json_object(self, path: Path, label: str) -> dict[str, Any]:
        data = self._read_bytes(path, label)
        raw = self._decode_json(data, path)
        if not isinstance(raw, dict):
            raise DispatchError(f"{path}: {label} must contain a JSON object")
        return raw

    @staticmethod
    def _read_bytes(path: Path, label: str) -> bytes:
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise DispatchError(f"cannot read {label} {path}: {exc}") from exc
        if not data:
            raise DispatchError(f"{label} {path} is empty")
        return data

    @staticmethod
    def _decode_json(data: bytes, path: Path) -> Any:
        try:
            return json.loads(data, parse_constant=_reject_constant)
        except (ValueError, UnicodeDecodeError) as exc:
            raise DispatchError(f"{path}: invalid UTF-8 JSON: {exc}") from exc

    @staticmethod
    def _require_str(raw: dict[str, Any], key: str, path: Path) -> str:
        value = raw.get(key)
        if not isinstance(value, str) or not value.strip():
            raise DispatchError(f"{path}: field {key!r} must be a non-empty string")
        return value
