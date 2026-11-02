"""Coordinate D: current model source -> current platform experiment client."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from dataclasses import replace
from typing import Any

from src.time_limits import deadline_end, D_CLIENT_MAX_S, SAVE_RESERVE_S
from src.config import RuntimeConfig
from src.dispatch.models import CaseQuestion
from src.handlers.base import HandlerContext
from src.materials.models import MaterialError
from src.materials.case_materials import CaseMaterials
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
from src.submission.writer import ResultWriter, SubmissionError

logger = logging.getLogger(__name__)

# Stage maxima are not guaranteed allocations. Allow long model generation;
# the formal client receives the remaining case budget, capped at 880 seconds.
FORMAL_RESERVE_S = 60.0 + SAVE_RESERVE_S
GENERATION_CAPS_S = (900.0, 900.0)
MIN_GENERATION_S = 10.0


class Deadline:
    def __init__(self, seconds: float) -> None:
        self.end = deadline_end(seconds)

    def remaining(self) -> float:
        return self.end - time.monotonic()

    def timeout(self, cap: float, *, reserve: float = 0.0) -> float:
        return max(0.0, min(cap, self.remaining() - reserve))


class PipelineDiagnostics:
    """Own-team diagnostics; these records are not platform execution evidence."""

    def __init__(self, config: RuntimeConfig, case_id: str, deadline: Deadline) -> None:
        self.config = config
        self.case_id = case_id
        self.deadline = deadline
        self.events: list[dict[str, Any]] = []

    def record(self, stage: str, **details: Any) -> None:
        event = {"stage": stage, "remaining_s": round(max(0.0, self.deadline.remaining()), 3)}
        event.update(details)
        self.events.append(event)
        logger.info("D %s stage=%s remaining_s=%s", self.case_id, stage, event["remaining_s"])
        relative = f"data/{self.config.team}/work/dcase/{self.case_id}/pipeline-state.json"
        try:
            ResultWriter(self.config).write_json(relative, {
                "case_id": self.case_id,
                "diagnostic_only": True,
                "trusted_execution_evidence": False,
                "events": self.events,
            })
        except (OSError, ValueError, SubmissionError) as exc:
            logger.warning("D %s pipeline diagnostics unavailable (%s)",
                           self.case_id, type(exc).__name__)


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
    # Include material loading in the same whole-question budget. The final
    # client needs time for source admission, formal execution and artifact copy.
    deadline = Deadline(1800.0)
    diagnostics = PipelineDiagnostics(config, case_id, deadline)
    diagnostics.record("started")
    materials = ExperimentMaterials(config)
    input_specs = _input_metric_specs(question)

    try:
        bundle = materials.bundle(case_id, question=question)
        try:
            pdfs = [name for name in question.materials if name.lower().endswith(".pdf")]
            if pdfs:
                scoped = replace(question, materials=pdfs)
                refs = await asyncio.to_thread(CaseMaterials(context, scoped).resolve)
                digests = await asyncio.to_thread(context.library.digests,
                    question.title + "\n" + question.input, refs, max_pages_per_paper=8)
                notes = "\n\n".join(
                    f"{d.filename}\n" + "\n".join(f"[PDF p.{e.page}] {e.text}" for e in d.excerpts)
                    for d in digests)
                bundle = replace(bundle, paper_notes=notes)
        except (ValueError, PermissionError, OSError) as exc:
            raise MaterialError("current question PDF material could not be read") from exc
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
        diagnostics.record("materials_missing", reason=str(exc))
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
    generator = SourceGenerator(context.llm, work_dir)

    current: SourceAttempt | None = None
    previous: SourceAttempt | None = None
    last_error = ""

    # At most two complete model replies. Both initial generation and repair
    # consume only the budget above FORMAL_RESERVE_S, never the formal reserve.
    for attempt_index, cap in enumerate(GENERATION_CAPS_S):
        generation_timeout = deadline.timeout(cap, reserve=FORMAL_RESERVE_S)
        if generation_timeout < MIN_GENERATION_S:
            last_error = "generation budget exhausted; formal experiment time was reserved"
            diagnostics.record("source_generation_skipped", attempt=attempt_index + 1,
                               reason=last_error, available_s=round(generation_timeout, 3))
            break
        diagnostics.record("source_generation_requested", attempt=attempt_index + 1,
                           timeout_s=round(generation_timeout, 3), repair=previous is not None)
        try:
            if previous is None:
                attempt = await generator.generate(bundle, question, timeout=generation_timeout)
            else:
                attempt = await generator.repair(
                    bundle, question, previous=previous, error=last_error,
                    timeout=generation_timeout,
                )
        except SourceGenerationError as exc:
            last_error = str(exc)
            diagnostics.record("source_generation_error", attempt=attempt_index + 1,
                               reason=last_error, upstream_status=exc.upstream_status,
                               retryable=exc.retryable)
            previous = None
            # Only a received, correctable output can enter repair.
            break

        if attempt.source_path is not None:
            current = attempt
            diagnostics.record("source_saved", attempt=attempt.attempt,
                               source_sha256=attempt.source_sha256)
            break

        last_error = attempt.error or "source extraction failed"
        diagnostics.record("source_rejected", attempt=attempt.attempt, reason=last_error)
        previous = attempt

    if current is None:
        logger.warning("D %s source generation failed: %s", case_id, last_error)
        diagnostics.record("source_generation_failed", reason=last_error,
                           formal_client_attempted=False)
        payload = build_failure_submission(case_id, "failed", failure_protocol)
        _write_submission(config, case_id, payload)
        return payload

    # The official extractor has already checked the complete response, size,
    # UTF-8, Python 3.9 syntax and top-level entrypoint. Additional local
    # heuristics must not turn an extractable source into a local-only run.
    if deadline.remaining() > FORMAL_RESERVE_S + 1.0:
        try:
            await asyncio.wait_for(
                asyncio.to_thread(_validate_generated_source, current.source_path),
                timeout=min(1.0, deadline.remaining() - FORMAL_RESERVE_S),
            )
            diagnostics.record("local_static_checked", blocking=False, executed=False)
        except Exception as exc:
            logger.warning("D %s advisory source check: %s", case_id, exc)
            diagnostics.record("local_static_warning", blocking=False, executed=False,
                               reason=str(exc))
    else:
        diagnostics.record("local_static_skipped", reason="formal reserve protected")

    if deadline.remaining() < 1:
        logger.warning("D %s has no budget left for the formal experiment", case_id)
        diagnostics.record("formal_client_not_started", reason="whole-question deadline expired")
        payload = build_failure_submission(case_id, "failed", failure_protocol)
        _write_submission(config, case_id, payload)
        return payload

    try:
        # This is a client invocation, not an assertion that trusted execution
        # started. Only the platform's /source and /run receipts prove that.
        diagnostics.record("formal_client_started", client_invocation_only=True,
                           source_sha256=current.source_sha256)
        payload = await asyncio.to_thread(
            run_case,
            config,
            config.team,
            case_id,
            current.source_path,
            bundle.protocol,
            work_dir,
            timeout=deadline.timeout(D_CLIENT_MAX_S, reserve=SAVE_RESERVE_S),
        )
        diagnostics.record("formal_result_received", submission_status=payload.get("status"))
        return payload
    except MaterialError as exc:
        logger.warning("D %s formal experiment material missing: %s", case_id, exc)
        diagnostics.record("formal_client_material_error", reason=str(exc))
        if (config.result_root / result_json(config.team, case_id)).exists():
            raise
        payload = build_failure_submission(case_id, "materials_missing", failure_protocol)
        _write_submission(config, case_id, payload)
        return payload
    except Exception as exc:
        logger.warning("D %s formal experiment failed: %s", case_id, exc)
        diagnostics.record("formal_client_error", error_type=type(exc).__name__, reason=str(exc))
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
