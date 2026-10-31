"""Coordinate the D-class pipeline: materials -> source -> self-check -> platform."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from src.config import RuntimeConfig
from src.dispatch.models import CaseQuestion
from src.handlers.base import HandlerContext
from src.materials.models import MaterialError
from src.reproduction.errors import (
    ReproductionError,
    SelfCheckError,
    SourceGenerationError,
    PlatformExecutionUnknownError,
    PlatformUnavailableError,
)
from src.reproduction.materials import (
    ExperimentMaterials,
    metric_specs_from_text,
)
from src.reproduction.platform_client import run_case
from src.reproduction.source_checks import check_source_file
from src.reproduction.source_generation import SourceAttempt, SourceGenerator
from src.submission.paths import result_json
from src.submission.writer import ResultWriter

logger = logging.getLogger(__name__)


class Deadline:
    def __init__(self, seconds: float) -> None:
        self.end = time.monotonic() + seconds

    def remaining(self) -> float:
        return self.end - time.monotonic()

    def timeout(self, cap: float) -> float:
        return max(1.0, min(cap, self.remaining()))


def _metrics(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    specs = protocol.get("metrics") if isinstance(protocol, dict) else None
    if not isinstance(specs, list) or len(specs) != 3:
        raise ReproductionError("protocol must declare exactly three metrics")
    result: list[dict[str, Any]] = []
    for spec in specs:
        if (
            not isinstance(spec, dict)
            or not isinstance(spec.get("name"), str)
            or not spec["name"].strip()
            or not isinstance(spec.get("unit"), str)
            or not spec["unit"].strip()
        ):
            raise ReproductionError("protocol metric needs a non-empty name and unit")
        result.append(
            {
                "name": spec["name"],
                "unit": spec["unit"],
                "reference": None,
                "observed": None,
                "relative_error": None,
            }
        )
    return result


def _input_metric_specs(question: CaseQuestion) -> list[dict[str, str]]:
    """Use input metric declarations only when exactly three are present."""
    specs = metric_specs_from_text(question.input)
    return specs if len(specs) == 3 else []


def _failure_protocol(
    protocol: dict[str, Any] | None,
    input_specs: list[dict[str, str]],
) -> dict[str, Any] | None:
    """Choose metric metadata for a self-built failed/materials_missing answer.

    ``protocol.json`` remains authoritative while it is readable because the
    formal executor and judge use it. When the protocol file itself is missing,
    the task input's three ``name=.../unit=...`` declarations are the fallback
    that still lets the answer keep the required three metric objects.
    """
    if protocol is not None:
        return protocol
    if len(input_specs) == 3:
        return {"metrics": input_specs}
    return None


def _same_metric_specs(
    input_specs: list[dict[str, str]],
    protocol_specs: Any,
) -> bool:
    if not isinstance(protocol_specs, list) or len(input_specs) != len(protocol_specs):
        return False
    return all(
        isinstance(protocol_spec, dict)
        and input_spec["name"] == protocol_spec.get("name")
        and input_spec["unit"] == protocol_spec.get("unit")
        for input_spec, protocol_spec in zip(input_specs, protocol_specs)
    )


def build_failure_submission(
    case_id: str,
    status: str,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "status": status,
        "metrics": _metrics(protocol),
        "figures": [],
    }


def _write_submission(
    config: RuntimeConfig,
    case_id: str,
    payload: dict[str, Any],
) -> None:
    ResultWriter(config).write_json(result_json(config.team, case_id), payload)


def _validate_generated_source(source_path: Path) -> None:
    # Generated code runs only in the platform's isolated formal executor.
    # Static checks provide useful feedback without granting it this process's
    # filesystem, environment or network permissions.
    try:
        check_source_file(source_path)
    except (ValueError, SyntaxError, OSError) as exc:
        raise SelfCheckError(str(exc)) from exc


async def run_reproduction(
    context: HandlerContext,
    question: CaseQuestion,
) -> dict[str, Any]:
    config = context.config
    case_id = question.case_id
    materials = ExperimentMaterials(config)
    input_specs = _input_metric_specs(question)

    try:
        bundle = materials.bundle(case_id)
    except MaterialError as exc:
        protocol = _read_protocol_quietly(materials, case_id)
        failure_protocol = _failure_protocol(protocol, input_specs)
        if failure_protocol is None:
            raise
        try:
            payload = build_failure_submission(
                case_id, "materials_missing", failure_protocol
            )
        except ReproductionError:
            # Neither protocol.json nor the question input declares the three
            # required metrics; preserve the original material error.
            raise exc
        logger.warning("D %s materials missing: %s", case_id, exc)
        _write_submission(config, case_id, payload)
        return payload

    if input_specs and not _same_metric_specs(
        input_specs, bundle.protocol.get("metrics")
    ):
        logger.warning(
            "D %s input metric declarations differ from protocol.json; "
            "the formal experiment keeps the protocol contract",
            case_id,
        )
    failure_protocol = _failure_protocol(bundle.protocol, input_specs)

    work_dir = config.result_root / "data" / config.team / "work" / "dcase" / case_id
    work_dir.mkdir(parents=True, exist_ok=True)
    # The platform hard-stops /solve at 180s; keep a small return margin.
    deadline = Deadline(170.0)
    generator = SourceGenerator(context.llm, work_dir)

    current: SourceAttempt | None = None
    last_error = ""

    for attempt_index in range(2):
        try:
            attempt = await generator.generate(
                bundle, question, timeout=deadline.timeout(60.0)
            )
        except SourceGenerationError as exc:
            last_error = str(exc)
            if attempt_index == 0 and deadline.remaining() > 90:
                continue
            break

        if attempt.source_path is not None:
            current = attempt
            break

        last_error = attempt.error or "source extraction failed"
        if deadline.remaining() > 70:
            try:
                repaired = await generator.repair(
                    bundle,
                    question,
                    previous=attempt,
                    error=last_error,
                    timeout=deadline.timeout(45.0),
                )
            except SourceGenerationError as exc:
                last_error = str(exc)
                break
            if repaired.source_path is not None:
                current = repaired
        break

    if current is None:
        logger.warning("D %s source generation failed: %s", case_id, last_error)
        payload = build_failure_submission(case_id, "failed", failure_protocol)
        _write_submission(config, case_id, payload)
        return payload

    if deadline.remaining() > 55:
        try:
            await asyncio.to_thread(
                _validate_generated_source,
                current.source_path,
            )
        except SelfCheckError as exc:
            logger.warning("D %s source validation failed: %s", case_id, exc)
            repaired: SourceAttempt | None = None
            if deadline.remaining() > 90:
                try:
                    repaired = await generator.repair(
                        bundle,
                        question,
                        previous=current,
                        error=str(exc),
                        timeout=deadline.timeout(45.0),
                    )
                except SourceGenerationError:
                    repaired = None
            if repaired is None or repaired.source_path is None:
                payload = build_failure_submission(case_id, "failed", failure_protocol)
                _write_submission(config, case_id, payload)
                return payload
            current = repaired
            if deadline.remaining() <= 50:
                payload = build_failure_submission(case_id, "failed", failure_protocol)
                _write_submission(config, case_id, payload)
                return payload
            try:
                await asyncio.to_thread(
                    _validate_generated_source,
                    current.source_path,
                )
            except SelfCheckError as second_exc:
                logger.warning(
                    "D %s repaired source still fails validation: %s",
                    case_id,
                    second_exc,
                )
                payload = build_failure_submission(case_id, "failed", failure_protocol)
                _write_submission(config, case_id, payload)
                return payload

    if deadline.remaining() < 15:
        logger.warning("D %s has no budget left for the formal experiment", case_id)
        payload = build_failure_submission(case_id, "failed", failure_protocol)
        _write_submission(config, case_id, payload)
        return payload

    try:
        return await asyncio.to_thread(
            run_case,
            config,
            config.team,
            case_id,
            current.source_path,
            bundle.protocol,
            work_dir,
            timeout=deadline.timeout(150.0),
        )
    except MaterialError as exc:
        logger.warning("D %s formal experiment material missing: %s", case_id, exc)
        if (config.result_root / result_json(config.team, case_id)).exists():
            raise
        payload = build_failure_submission(case_id, "materials_missing", failure_protocol)
        _write_submission(config, case_id, payload)
        return payload
    except Exception as exc:
        logger.warning("D %s formal experiment failed: %s", case_id, exc)
        if isinstance(exc, (PlatformExecutionUnknownError, PlatformUnavailableError)):
            raise
        if (config.result_root / result_json(config.team, case_id)).exists():
            # The client may already have saved a real or partial submission.
            # Preserve its bytes and figures even if subsequent parsing fails.
            logger.error("D %s keeps existing platform result; refusing to overwrite it", case_id)
            raise
        payload = build_failure_submission(case_id, "failed", failure_protocol)
        _write_submission(config, case_id, payload)
        return payload


def _read_protocol_quietly(
    materials: ExperimentMaterials,
    case_id: str,
) -> dict[str, Any] | None:
    try:
        return materials.read_protocol(case_id)
    except MaterialError:
        return None
