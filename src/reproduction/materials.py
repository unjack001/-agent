"""Access to /data/experiments/<case_id>/ public materials."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import RuntimeConfig
from src.materials.models import MaterialError


# The formal D question input states the three required metrics as
# "name=Macro-F1，unit=分数；name=Accuracy，unit=%；...".  Keep parsing
# tolerant of ASCII/full-width punctuation so a missing protocol file can
# still produce the required three-metric failure answer.
_METRIC_PAIR_RE = re.compile(
    r"name\s*[=:：]\s*(?P<name>[^,，;；。\n]+?)\s*[,，;；]\s*"
    r"unit\s*[=:：]\s*(?P<unit>[^,，;；。.!！？\n]+?)\s*"
    r"(?=[,，;；。.!！？]|$)",
    re.IGNORECASE,
)


def metric_specs_from_text(text: str) -> list[dict[str, str]]:
    """Return ``name``/``unit`` pairs declared in the task input.

    The caller must require exactly three pairs before treating this as the
    current case's protocol; arbitrary prose may mention other metrics.
    """
    if not isinstance(text, str):
        return []
    specs: list[dict[str, str]] = []
    for match in _METRIC_PAIR_RE.finditer(text):
        name = match.group("name").strip().strip("\"'`")
        unit = match.group("unit").strip().strip("\"'`")
        if name and unit:
            specs.append({"name": name, "unit": unit})
    return specs


@dataclass(frozen=True)
class ExperimentBundle:
    case_id: str
    case_dir: Path
    method: str
    protocol: dict[str, Any]
    sample: Any
    paper_notes: str = ""

    @property
    def request(self) -> Any:
        """Return only an actually supplied public request; do not invent phases."""
        return json.loads(json.dumps(self.sample, ensure_ascii=False))


class ExperimentMaterials:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.root = config.experiments_dir

    def case_dir(self, case_id: str) -> Path:
        if not isinstance(case_id, str) or not re.fullmatch(r"D-[0-9]{2,}", case_id):
            raise MaterialError(f"unsupported experiment case: {case_id}")
        try:
            root = self.root.resolve()
            path = (root / case_id).resolve()
        except OSError as exc:
            raise MaterialError(f"cannot resolve experiment materials: {exc}") from exc
        if not path.is_relative_to(root):
            raise MaterialError(f"experiment path escapes {root}: {path}")
        if not path.is_dir():
            raise MaterialError(f"experiment materials missing: {path}")
        return path

    def bundle(self, case_id: str, question=None) -> ExperimentBundle:
        # Current runtime files are optional. There is no offline-manual fallback.
        if not isinstance(case_id, str) or not re.fullmatch(r"D-[0-9]{2,}", case_id):
            raise MaterialError(f"invalid experiment case: {case_id!r}")
        root = self.root.resolve()
        directory = (root / case_id).resolve()
        if not directory.is_relative_to(root):
            raise MaterialError("experiment path escapes current experiment root")
        if question is None and not directory.is_dir():
            raise MaterialError(f"experiment materials missing: {directory}")
        if (directory / "manifest.json").exists():
            self.verify_manifest(directory)
        method = self.read_method(case_id, directory) if (directory / "method.md").exists() else ""
        protocol = self.read_protocol(case_id, directory) if (directory / "protocol.json").exists() else {}
        sample = self.read_sample(case_id, directory) if (directory / "sample.json").exists() else {}
        if question is not None:
            if not method:
                method = question.input + "\n" + "\n".join(question.constraints)
            if not protocol:
                specs = metric_specs_from_text("\n".join([question.input, *question.constraints, question.expected_output]))
                specs = list({(s['name'], s['unit']): s for s in specs}.values())
                protocol = {"case_id": case_id}
                if len(specs) == 3:
                    protocol['metrics'] = specs
        return ExperimentBundle(case_id, directory, method, protocol, sample)

    def read_method(self, case_id: str, case_dir: Path | None = None) -> str:
        directory = case_dir or self.case_dir(case_id)
        path = directory / "method.md"
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise MaterialError(f"cannot read {path}: {exc}") from exc
        if not text.strip():
            raise MaterialError(f"{path} is empty")
        return text

    def read_protocol(self, case_id: str, case_dir: Path | None = None) -> dict[str, Any]:
        directory = case_dir or self.case_dir(case_id)
        path = directory / "protocol.json"
        raw = self._read_json(path)
        if not isinstance(raw, dict):
            raise MaterialError(f"{path} must contain an object")
        if "case_id" in raw and raw["case_id"] != case_id:
            raise MaterialError(f"{path}: case_id does not match {case_id}")
        metrics = raw.get("metrics")
        if metrics is None:
            return raw
        if not isinstance(metrics, list) or len(metrics) != 3:
            raise MaterialError(f"{path}: expected exactly three metrics")
        for item in metrics:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("name"), str)
                or not item["name"].strip()
                or not isinstance(item.get("unit"), str)
                or not item["unit"].strip()
            ):
                raise MaterialError(
                    f"{path}: each metric needs a non-empty name and unit"
                )
        return raw

    def read_sample(self, case_id: str, case_dir: Path | None = None) -> Any:
        directory = case_dir or self.case_dir(case_id)
        path = directory / "sample.json"
        raw = self._read_json(path)
        if not isinstance(raw, (dict, list)):
            raise MaterialError(f"{path}: public examples must be an object or list")
        if isinstance(raw, dict) and "case_id" in raw and raw['case_id'] != case_id:
            raise MaterialError(f"{path}: case_id does not match {case_id}")
        return raw

    def verify_manifest(self, case_dir: Path) -> None:
        path = case_dir / "manifest.json"
        raw = self._read_json(path)
        if not isinstance(raw, dict) or not isinstance(raw.get("files"), dict):
            raise MaterialError(f"{path}: manifest must contain files")
        root = case_dir.resolve()
        for name, expected in raw["files"].items():
            if not isinstance(name, str) or not isinstance(expected, str):
                raise MaterialError(f"{path}: invalid manifest entry")
            file_path = (case_dir / name).resolve()
            if not file_path.is_relative_to(root):
                raise MaterialError(f"{path}: manifest path escapes case directory: {name}")
            try:
                actual = hashlib.sha256(file_path.read_bytes()).hexdigest()
            except OSError as exc:
                raise MaterialError(f"cannot read manifest file {file_path}: {exc}") from exc
            if actual.lower() != expected.lower():
                raise MaterialError(f"manifest hash mismatch: {name}")

    def client_path(self) -> Path:
        path = self.root / "run_experiment.py"
        if not path.is_file():
            raise MaterialError(f"platform experiment client missing: {path}")
        return path

    @staticmethod
    def _read_json(path: Path) -> Any:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise MaterialError(f"cannot read {path}: {exc}") from exc
        try:
            return json.loads(text)
        except ValueError as exc:
            raise MaterialError(f"{path}: invalid JSON: {exc}") from exc
