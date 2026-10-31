"""Write submission files and read them back for verification."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from src.config import RuntimeConfig
from src.materials.access import ensure_no_links
from src.submission.paths import idea_markdown, survey_markdown
from src.submission.schemas import FUNCTION_BY_CASE_PREFIX, OUTPUT_FIELDS

RESULT_FILE_PATTERN = re.compile(r"^[A-E]-\d{2,}\.json$")


class SubmissionError(RuntimeError):
    """Raised when a path or file violates the submission contract."""


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


class ResultWriter:
    """Writes only inside the team's authorized submission directories."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.root = config.result_root
        self.team = config.team

    def write_json(self, relative_path: str, payload: dict[str, Any]) -> Path:
        """Write JSON and verify it by reading it back.

        Survey/Idea answers must be written after their data/expert Markdown
        report; the JSON path and markdown must match that file byte-for-byte.
        """
        if not isinstance(payload, dict):
            raise SubmissionError("JSON payload must be an object")
        parts = self._parts(relative_path)
        if not self._is_result_json(parts) and not self._is_work_file(parts):
            raise SubmissionError(
                f"{relative_path}: JSON results belong in data/{self.team}/<case_id>.json "
                f"or data/{self.team}/work/"
            )
        target = self._target(parts)
        if self._is_result_json(parts):
            expected_case_id = parts[2][:-5]
            category = FUNCTION_BY_CASE_PREFIX[expected_case_id[0]]
            expected_fields = set(OUTPUT_FIELDS[category])
            if set(payload) != expected_fields:
                raise SubmissionError(
                    f"{relative_path}: top-level fields must be "
                    f"{sorted(expected_fields)}, got {sorted(map(str, payload))}"
                )
            if payload.get("case_id") != expected_case_id:
                raise SubmissionError(
                    f"{relative_path}: case_id {payload.get('case_id')!r} does not match "
                    f"file name {expected_case_id!r}"
                )
            expected_report = self._expected_report(category, expected_case_id)
            if expected_report is not None:
                if payload.get("path") != expected_report:
                    raise SubmissionError(
                        f"{relative_path}: path must be {expected_report!r}"
                    )
                markdown = payload.get("markdown")
                if not isinstance(markdown, str):
                    raise SubmissionError(f"{relative_path}: markdown must be a string")
                try:
                    on_disk = self.read_text(expected_report)
                except SubmissionError as exc:
                    raise SubmissionError(
                        f"{relative_path}: write the Markdown report before its JSON: {exc}"
                    ) from exc
                if on_disk != markdown:
                    raise SubmissionError(
                        f"{relative_path}: markdown does not match {expected_report}"
                    )
        try:
            text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        except (TypeError, ValueError) as exc:
            raise SubmissionError(f"{relative_path}: payload is not valid JSON: {exc}") from exc
        self._atomic_write(target, self._encode_utf8(text, relative_path))
        self.verify_json(relative_path, payload)
        return target

    def write_markdown(self, relative_path: str, text: str) -> Path:
        if not isinstance(text, str):
            raise SubmissionError("markdown content must be a string")
        parts = self._parts(relative_path)
        if parts[:2] != ("data", "expert"):
            raise SubmissionError("markdown reports must live under data/expert/")
        target = self._target(parts)
        self._atomic_write(target, self._encode_utf8(text, relative_path))
        self.verify_markdown(relative_path, text)
        return target

    def read_json(self, relative_path: str) -> dict[str, Any]:
        target = self._target(self._parts(relative_path))
        data = self._read_bytes(target, relative_path)
        try:
            payload = json.loads(data, parse_constant=_reject_constant)
        except (ValueError, UnicodeDecodeError) as exc:
            raise SubmissionError(f"{relative_path}: invalid UTF-8 JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise SubmissionError(f"{relative_path}: JSON file must contain an object")
        return payload

    def read_text(self, relative_path: str) -> str:
        target = self._target(self._parts(relative_path))
        data = self._read_bytes(target, relative_path)
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SubmissionError(f"{relative_path}: file is not valid UTF-8") from exc

    def verify_json(self, relative_path: str, expected: dict[str, Any]) -> dict[str, Any]:
        actual = self.read_json(relative_path)
        if actual != expected:
            raise SubmissionError(f"{relative_path}: read-back JSON does not match payload")
        return actual

    def verify_markdown(self, relative_path: str, expected: str) -> str:
        actual = self.read_text(relative_path)
        if actual != expected:
            raise SubmissionError(
                f"{relative_path}: read-back Markdown does not match the JSON markdown field"
            )
        return actual

    def verify_exists(self, relative_path: str) -> Path:
        target = self._target(self._parts(relative_path))
        if not target.is_file() or target.stat().st_size == 0:
            raise SubmissionError(f"{relative_path}: required file is missing or empty")
        return target

    def _parts(self, relative_path: str) -> tuple[str, ...]:
        if not isinstance(relative_path, str) or not relative_path:
            raise SubmissionError("submission path must be a non-empty string")
        if "\\" in relative_path:
            raise SubmissionError(f"{relative_path}: use '/' separators")
        pure = PurePosixPath(relative_path)
        if pure.is_absolute() or ".." in pure.parts:
            raise SubmissionError(f"{relative_path}: not a safe relative path")
        parts = pure.parts
        if not self._allowed(parts):
            raise SubmissionError(f"{relative_path}: outside the team's authorized paths")
        return parts

    def _allowed(self, parts: tuple[str, ...]) -> bool:
        if len(parts) < 3 or parts[0] != "data":
            return False
        if parts[1] == self.team:
            if len(parts) == 3:
                return bool(RESULT_FILE_PATTERN.fullmatch(parts[2]))
            return len(parts) >= 4 and parts[2] == "work"
        if parts[1] == "expert":
            if len(parts) != 3:
                return False
            name = parts[2]
            return name.endswith(f"-{self.team}.md") and (
                "-综述-" in name or "-灵感Idea报告-" in name
            )
        if parts[1] == "figure":
            return len(parts) >= 4 and parts[2] == self.team
        return False

    def _is_result_json(self, parts: tuple[str, ...]) -> bool:
        return (
            len(parts) == 3
            and parts[:2] == ("data", self.team)
            and bool(RESULT_FILE_PATTERN.fullmatch(parts[2]))
        )

    def _is_work_file(self, parts: tuple[str, ...]) -> bool:
        return (
            len(parts) >= 4
            and parts[0] == "data"
            and parts[1] == self.team
            and parts[2] == "work"
        )

    def _expected_report(self, category: str, case_id: str) -> str | None:
        if category == "Survey":
            return survey_markdown(self.team, case_id)
        if category == "Idea":
            return idea_markdown(self.team, case_id)
        return None

    def _target(self, parts: tuple[str, ...]) -> Path:
        target = self.root.joinpath(*parts)
        try:
            ensure_no_links(target, self.root)
            resolved = target.resolve()
            root = self.root.resolve()
        except OSError as exc:
            raise SubmissionError(f"{'/'.join(parts)}: cannot resolve path: {exc}") from exc
        if not resolved.is_relative_to(root):
            raise SubmissionError(f"{'/'.join(parts)}: path escapes result root {root}")
        return resolved

    @staticmethod
    def _encode_utf8(text: str, relative_path: str) -> bytes:
        try:
            return text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise SubmissionError(f"{relative_path}: content is not valid UTF-8") from exc

    @staticmethod
    def _read_bytes(target: Path, relative_path: str) -> bytes:
        if not target.is_file():
            raise SubmissionError(f"{relative_path}: file does not exist")
        try:
            data = target.read_bytes()
        except OSError as exc:
            raise SubmissionError(f"{relative_path}: cannot read file: {exc}") from exc
        if not data:
            raise SubmissionError(f"{relative_path}: file is empty")
        return data

    @staticmethod
    def _atomic_write(target: Path, data: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temp_name, 0o644)
            os.replace(temp_name, target)
        except BaseException:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise
