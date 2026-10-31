"""Invoke the platform-mounted run_experiment.py and validate its submission."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

from src.config import RuntimeConfig
from src.materials.models import MaterialError
from src.reproduction.errors import (
    PlatformError,
    PlatformExecutionUnknownError,
    PlatformUnavailableError,
)
from src.submission.paths import result_json
from src.submission.writer import ResultWriter, SubmissionError

STATUSES = ("completed", "failed", "materials_missing")
METRIC_KEYS = {"name", "unit", "reference", "observed", "relative_error"}
FIGURE_KEYS = {"path", "caption", "x_label", "y_label", "legend"}


def run_case(
    config: RuntimeConfig,
    team: str,
    case_id: str,
    implementation_path: Path,
    protocol: dict[str, Any],
    work_dir: Path,
    *,
    timeout: float,
) -> dict[str, Any]:
    client = config.experiments_dir / "run_experiment.py"
    if not client.is_file():
        # The frozen client is part of the allowed experiment materials, so a
        # missing client is a material problem, not a failed experiment.
        raise MaterialError(f"platform experiment client missing: {client}")

    active = os.getenv("RESEARCH_EXPERIMENT_CASE")
    if active != case_id:
        raise PlatformError(f"active experiment case is {active!r}, expected {case_id!r}")
    if os.getenv("RESEARCH_LAYERED_D") != "layered-d-v3":
        raise PlatformError("RESEARCH_LAYERED_D must be layered-d-v3")
    if not (os.getenv("RESEARCH_EXPERIMENT_URL") or "").startswith("unix:///"):
        raise PlatformError("RESEARCH_EXPERIMENT_URL is not a platform Unix socket")
    if os.getenv("RESEARCH_D_SOURCE_PROVENANCE") != "d-model-source-v1":
        raise PlatformError("RESEARCH_D_SOURCE_PROVENANCE must be d-model-source-v1")

    work_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(client),
        "--case",
        case_id,
        "--implementation",
        str(implementation_path.resolve()),
    ]
    env = os.environ.copy()
    env["RESEARCH_WORKSPACE_TEAM"] = team
    env["RESEARCH_RESULT_ROOT"] = str(config.result_root)
    before = _result_fingerprint(config, team, case_id)
    try:
        completed = subprocess.run(
            command,
            cwd=str(work_dir),
            env=env,
            capture_output=True,
            timeout=max(1.0, timeout),
        )
    except subprocess.TimeoutExpired as exc:
        _save_logs(work_dir, exc.stdout, exc.stderr)
        submission = _read_current_result(config, team, case_id, before)
        if submission is not None:
            validate_submission(submission, case_id, team, protocol)
            _verify_figures(config, submission)
            return submission
        raise PlatformExecutionUnknownError("platform client timed out before a confirmed current result") from exc
    except OSError as exc:
        raise PlatformError(f"cannot start platform client: {exc}") from exc

    _save_logs(work_dir, completed.stdout, completed.stderr)
    if completed.returncode != 0:
        submission = _read_current_result(config, team, case_id, before)
        if submission is not None:
            validate_submission(submission, case_id, team, protocol)
            _verify_figures(config, submission)
            return submission
        message = _message(completed.stderr, completed.stdout)
        if "D_SOURCE_AUDIT_UNAVAILABLE" in message or re.search(r"\b503\b", message):
            raise PlatformUnavailableError(message)
        raise PlatformError(message)

    file_submission = _read_result(config, team, case_id)
    if file_submission is None:
        raise PlatformError("platform client did not write the result file")
    stdout_submission = _parse_stdout(completed.stdout)
    if before is not None and _result_fingerprint(config, team, case_id) == before and stdout_submission != file_submission:
        raise PlatformExecutionUnknownError("platform client did not confirm or update the existing result")
    if stdout_submission is not None and stdout_submission != file_submission:
        raise PlatformError("platform client stdout and result file disagree")

    validate_submission(file_submission, case_id, team, protocol)
    _verify_figures(config, file_submission)
    return file_submission


def validate_submission(
    submission: dict[str, Any],
    case_id: str,
    team: str,
    protocol: dict[str, Any],
) -> None:
    if not isinstance(submission, dict):
        raise PlatformError("submission is not an object")
    if set(submission) != {"case_id", "status", "metrics", "figures"}:
        raise PlatformError("submission must contain only case_id/status/metrics/figures")
    if submission.get("case_id") != case_id:
        raise PlatformError(f"submission case_id does not match {case_id}")
    status = submission.get("status")
    if status not in STATUSES:
        raise PlatformError(f"invalid submission status: {status!r}")

    metrics = submission.get("metrics")
    expected = protocol.get("metrics")
    if not isinstance(metrics, list) or len(metrics) != 3:
        raise PlatformError("submission must contain exactly three metrics")
    if not isinstance(expected, list) or len(expected) != 3:
        raise PlatformError("protocol must declare exactly three metrics")
    for metric, spec in zip(metrics, expected):
        if (
            not isinstance(spec, dict)
            or not isinstance(spec.get("name"), str)
            or not spec["name"].strip()
            or not isinstance(spec.get("unit"), str)
            or not spec["unit"].strip()
        ):
            raise PlatformError("protocol metric needs a non-empty name and unit")
        if not isinstance(metric, dict) or set(metric) != METRIC_KEYS:
            raise PlatformError("each metric must contain exactly its five fields")
        if metric.get("name") != spec.get("name") or metric.get("unit") != spec.get("unit"):
            raise PlatformError("metric name/unit does not match protocol")
        for key in ("reference", "observed", "relative_error"):
            value = metric.get(key)
            if value is not None and not _finite_number(value):
                raise PlatformError(f"metric {key} must be null or a finite number")
        if (metric["reference"] is None or metric["observed"] is None) and metric["relative_error"] is not None:
            raise PlatformError("relative_error must be null without reference and observed values")
    if status == "completed":
        if not all(_finite_number(metric["observed"]) for metric in metrics):
            raise PlatformError("completed submission needs three finite observed values")
    elif any(metric["observed"] is not None for metric in metrics):
        raise PlatformError("failed/materials_missing submissions must have null observed values")

    figures = submission.get("figures")
    if not isinstance(figures, list):
        raise PlatformError("figures must be a list")
    for figure in figures:
        if not isinstance(figure, dict) or set(figure) != FIGURE_KEYS:
            raise PlatformError("each figure must contain exactly its five fields")
        path = figure.get("path")
        if not isinstance(path, str):
            raise PlatformError("figure path must be a string")
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts or "\\" in path:
            raise PlatformError(f"unsafe figure path: {path}")
        if pure.parts[:3] != ("data", "figure", team):
            raise PlatformError(f"figure path must be under data/figure/{team}/")
        for key in ("caption", "x_label", "y_label", "legend"):
            if not isinstance(figure.get(key), str) or not figure[key].strip():
                raise PlatformError(f"figure {key} must be a non-empty string")


def _verify_figures(config: RuntimeConfig, submission: dict[str, Any]) -> None:
    writer = ResultWriter(config)
    for figure in submission["figures"]:
        try:
            writer.verify_exists(figure["path"])
        except SubmissionError as exc:
            raise PlatformError(f"figure is missing: {figure['path']}") from exc


def _read_result(
    config: RuntimeConfig,
    team: str,
    case_id: str,
) -> dict[str, Any] | None:
    relative = result_json(team, case_id)
    path = config.result_root / relative
    if not path.is_file():
        return None
    try:
        return ResultWriter(config).read_json(relative)
    except SubmissionError as exc:
        raise PlatformError(f"platform result is unreadable or invalid: {exc}") from exc


def _result_fingerprint(config: RuntimeConfig, team: str, case_id: str) -> tuple | None:
    path = config.result_root / result_json(team, case_id)
    if not path.exists():
        return None
    # Apply the same capability checks before reading either metadata or bytes.
    writer = ResultWriter(config)
    target = writer._target(writer._parts(result_json(team, case_id)))
    info = target.stat()
    return (info.st_ino, info.st_size, info.st_mtime_ns, hashlib.sha256(target.read_bytes()).digest())


def _read_current_result(
    config: RuntimeConfig, team: str, case_id: str, before: tuple | None,
) -> dict[str, Any] | None:
    current = _result_fingerprint(config, team, case_id)
    if current is None or current == before:
        return None
    return _read_result(config, team, case_id)


def _parse_stdout(stdout: bytes | None) -> dict[str, Any] | None:
    text = (stdout or b"").decode("utf-8", "replace").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except ValueError:
            return None
    return data if isinstance(data, dict) else None


def _save_logs(work_dir: Path, stdout: Any, stderr: Any) -> None:
    try:
        if stdout:
            (work_dir / "experiment.stdout").write_bytes(
                stdout if isinstance(stdout, bytes) else str(stdout).encode("utf-8")
            )
        if stderr:
            (work_dir / "experiment.stderr").write_bytes(
                stderr if isinstance(stderr, bytes) else str(stderr).encode("utf-8")
            )
    except OSError:
        pass


def _message(stderr: Any, stdout: Any) -> str:
    text = stderr or stdout or b""
    if not isinstance(text, bytes):
        text = str(text).encode("utf-8")
    decoded = text.decode("utf-8", "replace").strip()
    return "platform client failed: " + decoded[-4000:]


def _finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False
