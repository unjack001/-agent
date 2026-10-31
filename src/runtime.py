"""AgentHack runtime that accepts the challenge's direct eight-field question JSON."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from agenthack import AgentRuntime
from agenthack.telemetry import error_type, task
from fastapi import Body, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.dispatch.loader import CASE_ID_PATTERN, FUNCTIONS
from src.llm.parsing import LLMError
from src.reproduction.errors import PlatformExecutionUnknownError, PlatformUnavailableError
from src.submission.schemas import FUNCTION_BY_CASE_PREFIX, OUTPUT_FIELDS

DEFAULT_TIMEOUT_MS = 180_000
TELEMETRY_FLUSH_TIMEOUT_S = 1.0
logger = logging.getLogger(__name__)


class TaskInputError(ValueError):
    """A malformed HTTP question, separate from errors while solving it."""

    def __init__(self, issues: list[dict[str, Any]]) -> None:
        super().__init__("invalid task input")
        self.issues = issues


def _issue(field: str, message: str, *, nested: bool = False) -> dict[str, Any]:
    location = ["body", "input", field] if nested else ["body", field]
    return {"loc": location, "msg": message, "type": "value_error"}


class FlexibleAgentRuntime(AgentRuntime):
    """Accepts both the raw question object and the SDK TaskRequest envelope."""

    def _setup_routes(self) -> None:
        @self.app.exception_handler(RequestValidationError)
        async def request_validation_error(_request: Request, exc: RequestValidationError):
            # FastAPI's default error body may echo the entire rejected input.
            # Return field locations and safe diagnostics without question data.
            return JSONResponse(status_code=422, content={"detail": [
                {"loc": list(item["loc"]), "msg": "invalid JSON request body",
                 "type": item["type"]}
                for item in exc.errors()
            ]})

        @self.app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "telemetry_configured": self.telemetry.export_enabled,
            }

        @self.app.post("/solve")
        async def solve(
            http_request: Request,
            payload: Any = Body(...),
        ):
            try:
                (
                    task_input,
                    task_id,
                    task_type,
                    timeout_ms,
                    session_id,
                    user_id_hash,
                ) = self._normalize(payload)
            except TaskInputError as exc:
                return JSONResponse(status_code=422, content={"detail": exc.issues})
            case_id = task_input.get("case_id")

            output: dict[str, Any] = {"case_id": case_id}
            try:
                with self.telemetry.session(
                    task_id,
                    task_type,
                    session_id,
                    user_id_hash,
                    http_request.headers,
                ):
                    with task() as task_operation:
                        result = await asyncio.wait_for(
                            self.agent.solve(task_input), timeout_ms / 1000
                        )
                        if not isinstance(result, dict):
                            raise RuntimeError("agent output must be a JSON object")
                        function = task_input.get("function")
                        if function in OUTPUT_FIELDS and (
                            set(result) != set(OUTPUT_FIELDS[function])
                            or result.get("case_id") != task_input["case_id"]
                        ):
                            raise RuntimeError("agent output does not match the assigned schema")
                        output = result
                        task_operation.result(output)
            except asyncio.TimeoutError:
                return JSONResponse(status_code=504, content={"detail": "task timed out"})
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # HTTP 422 is reserved for an invalid request at this route's
                # boundary. Internal report/material/JSON validation must not
                # be misreported as rejection of the platform's eight fields.
                status = (
                    503 if isinstance(exc, PlatformUnavailableError) else
                    504 if isinstance(exc, PlatformExecutionUnknownError) else
                    502 if isinstance(exc, LLMError) else 500
                )
                logger.error("/solve failed case_id=%s function=%s status=%s error_type=%s",
                             case_id, task_input.get("function"), status, type(exc).__name__)
                if isinstance(exc, PermissionError):
                    logger.error(
                        "Check non-root write access to /output/data/<RESEARCH_WORKSPACE_TEAM>; "
                        "a bind mount keeps host ownership and hides image directory permissions"
                    )
                return JSONResponse(status_code=status, content={"detail": error_type(exc)})
            finally:
                if os.getenv("AGENTHACK_FLUSH_ON_SOLVE", "true").lower() == "true":
                    try:
                        await asyncio.wait_for(
                            asyncio.to_thread(self.telemetry.flush),
                            TELEMETRY_FLUSH_TIMEOUT_S,
                        )
                    except asyncio.TimeoutError:
                        logger.warning("Telemetry flush timed out after solve")
                    except Exception as exc:
                        # Export failures must not override an already saved
                        # valid submission or the original task error response.
                        # Exception bodies may include sensitive SDK settings.
                        logger.warning("Telemetry flush failed (%s)", type(exc).__name__)

            # The challenge platform reads the case result itself, not the
            # SDK's task envelope, and compares it with the result file.
            return output

    @staticmethod
    def _normalize(payload: Any):
        if not isinstance(payload, dict):
            raise TaskInputError([_issue("body", "request body must be a JSON object")])

        direct = any(key in payload for key in ("case_id", "function", "output_schema_ref"))
        inner = payload.get("input")
        nested = isinstance(inner, dict) and any(
            key in inner for key in ("function", "output_schema_ref", "title", "materials", "constraints", "expected_output")
        )
        question = payload if direct else inner if nested else None

        if question is not None:
            _validate_question(question, nested=not direct)
            task_id = payload.get("task_id")
            if task_id is None:
                task_id = question["case_id"]
            elif not isinstance(task_id, str) or not task_id.strip():
                raise TaskInputError([_issue("task_id", "task_id must be a non-empty string")])
            return (
                question,
                task_id,
                question["function"],
                _timeout_ms(payload),
                _optional_metadata(payload, "session_id"),
                _optional_metadata(payload, "user_id_hash"),
            )

        # Retain the SDK envelope for explicit dispatch-reference clients.
        # The formal direct-question path above never needs dispatch.json.
        if isinstance(payload.get("task_id"), str) and payload["task_id"].strip() and isinstance(inner, dict):
            task_input = dict(inner)
            task_input.setdefault("task_id", payload["task_id"])
            return (
                task_input,
                payload["task_id"],
                _optional_metadata(payload, "task_type") or "qa",
                _timeout_ms(payload),
                _optional_metadata(payload, "session_id"),
                _optional_metadata(payload, "user_id_hash"),
            )

        raise TaskInputError([_issue("body", "expected the eight-field question JSON or an SDK task envelope")])


def _validate_question(question: dict[str, Any], *, nested: bool) -> None:
    issues: list[dict[str, Any]] = []
    for field in ("case_id", "function", "title", "input", "expected_output", "output_schema_ref"):
        value = question.get(field)
        if not isinstance(value, str):
            issues.append(_issue(field, f"{field} must be a string", nested=nested))
        elif field in ("case_id", "function", "input", "output_schema_ref") and not value.strip():
            issues.append(_issue(field, f"{field} must be non-empty", nested=nested))
    for field in ("materials", "constraints"):
        value = question.get(field)
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            issues.append(_issue(field, f"{field} must be an array of strings", nested=nested))
    case_id, function = question.get("case_id"), question.get("function")
    if isinstance(case_id, str) and not CASE_ID_PATTERN.fullmatch(case_id):
        issues.append(_issue("case_id", "case_id must match A-01 through E-100 style IDs", nested=nested))
    if isinstance(function, str) and function not in FUNCTIONS:
        issues.append(_issue("function", "function must be one of the five contest types", nested=nested))
    if isinstance(case_id, str) and CASE_ID_PATTERN.fullmatch(case_id) and function in FUNCTIONS:
        if FUNCTION_BY_CASE_PREFIX[case_id[0]] != function:
            issues.append(_issue("function", "function must match the case_id category", nested=nested))
    if isinstance(question.get("output_schema_ref"), str) and question["output_schema_ref"] != function:
        issues.append(_issue("output_schema_ref", "output_schema_ref must equal function", nested=nested))
    if issues:
        raise TaskInputError(issues)


def _optional_metadata(payload: dict[str, Any], field: str) -> str | None:
    value = payload.get(field)
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise TaskInputError([_issue(field, f"{field} must be a non-empty string when provided")])
    return value


def _timeout_ms(payload: dict[str, Any]) -> int:
    constraints = payload.get("constraints")
    if isinstance(constraints, dict):
        if "timeout_ms" in constraints:
            value = constraints["timeout_ms"]
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
            raise TaskInputError([_issue("constraints", "timeout_ms must be a positive integer")])
    return DEFAULT_TIMEOUT_MS
