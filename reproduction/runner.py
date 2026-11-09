"""Coordinate D: current model source -> current platform experiment client."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from dataclasses import replace
from typing import Any

from src.config import RuntimeConfig
from src.dispatch.models import CaseQuestion
from src.handlers.base import HandlerContext
from src.materials.models import MaterialError, has_evidence_text
from src.materials.section_evidence import record_reading
from src.materials.case_materials import CaseMaterials
from src.reproduction.errors import (
    ReproductionError,
    SelfCheckError,
    PlatformExecutionUnknownError,
    PlatformUnavailableError,
)
from src.reproduction.materials import (
    ExperimentMaterials,
    metric_specs_from_text,
)
from src.reproduction.platform_client import run_case
from src.reproduction.metric_contract import input_contract, resolve_contract, public_reference
from src.reproduction.source_checks import check_source_file
from src.reproduction.preparation import prepare_source
from src.submission.paths import result_json
from src.submission.writer import ResultWriter, SubmissionError

logger = logging.getLogger(__name__)


class PipelineDiagnostics:
    """Own-team diagnostics; these records are not platform execution evidence."""

    def __init__(self, config: RuntimeConfig, case_id: str) -> None:
        self.config = config
        self.case_id = case_id
        self.started = time.monotonic()
        self.events: list[dict[str, Any]] = []

    def record(self, stage: str, **details: Any) -> None:
        event = {"stage": stage, "elapsed_s": round(time.monotonic() - self.started, 3)}
        event.update(details)
        self.events.append(event)
        logger.info("D %s stage=%s elapsed_s=%s", self.case_id, stage, event["elapsed_s"])
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
                "reference": public_reference(spec),
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
    """Current input names/units remain authoritative even on material failure."""
    return input_contract(protocol, input_specs)


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
    diagnostics = PipelineDiagnostics(config, case_id)
    diagnostics.record("started")
    from src.paper_reading.runtime import skill_metadata
    diagnostics.record('paper_reading_skill',**skill_metadata('D'))
    materials = ExperimentMaterials(config)
    input_specs = _input_metric_specs(question)
    paper_evidence = []

    try:
        bundle = materials.bundle(case_id, question=question)
        bundle = replace(bundle, protocol=resolve_contract(bundle.protocol, input_specs, question.input))
        try:
            pdfs = [name for name in question.materials if name.lower().endswith(".pdf")]
            if pdfs:
                scoped = replace(question, materials=pdfs)
                refs = await asyncio.to_thread(CaseMaterials(context, scoped).resolve)
                query = '\n'.join([question.title, question.input, *question.constraints, question.expected_output])
                from src.paper_reading.evidence import reading_digests
                digests = await asyncio.to_thread(reading_digests, context.library, query, refs, role='D')
                record_reading(context, question, digests)
                readable = {d.filename for d in digests if any(
                    e.page > 0 and has_evidence_text(e.text, filename=d.filename, title=d.title)
                    for e in d.excerpts)}
                unreadable = [ref.filename for ref in refs if ref.filename not in readable]
                if not refs or unreadable:
                    raise MaterialError("材料问题：D 类指定 PDF 没有可用正文：" + ", ".join(unreadable or pdfs))
                notes = "\n\n".join(
                    f"{d.filename}\n" + "\n".join(f"[PDF p.{e.page}] {e.text}" for e in d.excerpts)
                    for d in digests)
                bundle = replace(bundle, paper_notes=notes)
                paper_evidence = [{'filename':d.filename,'page':e.page,'text':e.text}
                                  for d in digests for e in d.excerpts]
        except (ValueError, PermissionError, OSError, asyncio.TimeoutError) as exc:
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

    failure_protocol = _failure_protocol(bundle.protocol, input_specs)

    work_dir = config.result_root / "data" / config.team / "work" / "dcase" / case_id
    work_dir.mkdir(parents=True, exist_ok=True)
    current, last_error = await prepare_source(context, bundle, question, work_dir,
        diagnostics, paper_evidence)

    if current is None:
        logger.warning("D %s source generation failed: %s", case_id, last_error)
        diagnostics.record("source_generation_failed", reason=last_error,
                           formal_client_attempted=False)
        payload = build_failure_submission(case_id, "failed", failure_protocol)
        _write_submission(config, case_id, payload)
        return payload

    # Only a complete passing review of this exact source and saved Markdown
    # admits it to the one formal client invocation.

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
            timeout=None,
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
