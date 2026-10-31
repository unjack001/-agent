"""Generate run_experiment(request) via the platform Gateway, verbatim."""

from __future__ import annotations

import ast
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.dispatch.models import CaseQuestion
from src.handlers.base import LLMClient
from src.llm.parsing import LLMTruncatedError
from src.reproduction.errors import SourceGenerationError
from src.reproduction.materials import ExperimentBundle
from src.reproduction.model_source import diagnose, save_source


@dataclass(frozen=True)
class SourceAttempt:
    attempt: int
    response_path: Path
    body: dict[str, Any]
    source_path: Path | None
    source_sha256: str | None
    error: str | None


class SourceGenerator:
    def __init__(self, llm: LLMClient, work_dir: Path) -> None:
        self.llm = llm
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.attempt = 0

    async def generate(
        self,
        bundle: ExperimentBundle,
        question: CaseQuestion,
        *,
        timeout: float,
    ) -> SourceAttempt:
        return await self._call(
            self._messages(bundle, question), timeout, bundle.case_dir
        )

    async def repair(
        self,
        bundle: ExperimentBundle,
        question: CaseQuestion,
        previous: SourceAttempt,
        error: str,
        *,
        timeout: float,
    ) -> SourceAttempt:
        messages = self._messages(bundle, question)
        content = _response_content(previous.body)
        if not content and previous.source_path is not None:
            content = previous.source_path.read_text(encoding="utf-8")
        messages = messages + [
            {"role": "assistant", "content": (content or "(previous reply unavailable)")
             .encode("utf-8", "backslashreplace").decode("utf-8")},
            {"role": "user", "content": self._repair_prompt(error, bundle.case_id)},
        ]
        return await self._call(messages, timeout, bundle.case_dir)

    async def _call(
        self,
        messages: list[dict[str, str]],
        timeout: float,
        case_dir: Path,
    ) -> SourceAttempt:
        self.attempt += 1
        attempt = self.attempt
        try:
            _, body = await asyncio.wait_for(
                self.llm.complete_with_body(
                    messages, max_tokens=16000, temperature=0.0
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError as exc:
            raise SourceGenerationError("gateway generation timed out") from exc
        except LLMTruncatedError as exc:
            if not isinstance(exc.response_body, dict):
                raise SourceGenerationError("gateway response was truncated without diagnostic body") from exc
            body = exc.response_body
        except Exception as exc:
            raise SourceGenerationError(f"gateway generation failed: {exc}") from exc

        response_path = self.work_dir / f"generation-response-{attempt}.json"
        try:
            # JSON escaping preserves the response content and also lets us
            # retain diagnostic replies containing invalid Unicode scalars.
            text = json.dumps(body, ensure_ascii=True)
            response_path.write_text(text, encoding="utf-8")
            (self.work_dir / "generation-response.json").write_text(text, encoding="utf-8")
        except (TypeError, OSError) as exc:
            raise SourceGenerationError(f"cannot save model response: {exc}") from exc

        source_path = self.work_dir / "implementation.py"
        temporary_path = self.work_dir / f"implementation-{attempt}.py"
        try:
            sha = save_source(body, temporary_path)
        except (ValueError, OSError, SyntaxError, RecursionError) as exc:
            return SourceAttempt(
                attempt=attempt,
                response_path=response_path,
                body=body,
                source_path=None,
                source_sha256=None,
                error=f"{exc}; {diagnose(body)}",
            )

        imported = _non_stdlib_imports(
            temporary_path.read_text(encoding="utf-8"), case_dir
        )
        if imported:
            return SourceAttempt(
                attempt=attempt,
                response_path=response_path,
                body=body,
                source_path=None,
                source_sha256=None,
                error="non-stdlib imports: " + ", ".join(imported),
            )
        try:
            os.replace(temporary_path, source_path)
        except OSError as exc:
            raise SourceGenerationError(f"cannot install generated source: {exc}") from exc
        return SourceAttempt(
            attempt=attempt,
            response_path=response_path,
            body=body,
            source_path=source_path,
            source_sha256=sha,
            error=None,
        )

    @staticmethod
    def _messages(
        bundle: ExperimentBundle,
        question: CaseQuestion,
    ) -> list[dict[str, str]]:
        system = (
            "你是竞赛实验实现助手。你的回复会被平台审计，并原样作为正式实验源码。"
            "只返回一个完整的 ```python 代码块，包含全部实现；不要省略函数、不要返回补丁、"
            "不要拆成多段，也不要输出解释。"
        )
        user = (
            "请根据以下公开方法、协议和样例，完整实现 Python 3.9 标准库的 "
            "run_experiment(request)。不得联网、不得调用模型、不得读取隐藏标签、"
            "不得向标准输出打印日志、不得硬编码样例输出。必须遵守 protocol.json 的 "
            "request/response_schema、指标名称与单位、阶段和边界。\n\n"
            f"题目输入：\n{json.dumps(question.raw, ensure_ascii=False)}\n\n"
            f"===== method.md =====\n{bundle.method}\n\n"
            f"===== protocol.json =====\n"
            f"{json.dumps(bundle.protocol, ensure_ascii=False)}\n\n"
            f"===== sample.json =====\n"
            f"{json.dumps(bundle.sample, ensure_ascii=False)}\n\n"
            f"{_case_hints(bundle.case_id)}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def _repair_prompt(error: str, case_id: str | None = None) -> str:
        hint = ""
        if "length" in error.lower() or "truncat" in error.lower():
            hint += "\n上一回复截断，请压缩注释和重复辅助函数，返回紧凑但完整的实现，不要续写残缺源码。\n"
        if case_id == "D-05" and ("timeout" in error.lower() or "timed out" in error.lower()):
            hint += (
                "\n自测超时几乎总是表示模拟时间没有向前推进。重点检查 next_event："
                "不得反复选择空闲服务器的旧 free_at；下一事件必须严格大于当前 now，"
                "只取未来到达(arrival > now)或正在服务的完成事件(end > now)；"
                "没有未来事件时立即结束循环；while 循环必须设置硬上限。\n"
            )
        return (
            f"上一版实现未通过检查：\n{error[:4000]}\n{hint}\n"
            "请基于公开材料和上一版，返回一个完整、自洽、可直接执行、只依赖 Python 3.9 "
            "标准库的 run_experiment(request) 实现。只返回一个 ```python 代码块，"
            "不要输出补丁、解释或省略内容。"
        )


def _case_hints(case_id: str) -> str:
    common = (
        "===== 执行约束 =====\n"
        "所有循环必须有可证明的推进条件和硬上限；import 时只定义函数/常量，"
        "不得写文件或执行实验；不得向 stdout 打印。\n"
    )
    if case_id != "D-05":
        return common
    return common + (
        "===== D-05 事件循环硬性要求 =====\n"
        "1. 模拟时间 now 不得倒退；同一时刻允许按编号向多台空闲服务器依次派发；"
        "每轮循环必须推进 now、派发任务或丢弃任务，"
        "否则立即 raise RuntimeError。\n"
        "2. 没有可派发任务时，下一事件只能取 now 之后的下一次 arrival，"
        "或正在服务的任务完成时间 end；空闲服务器的旧 free_at 绝不能作为下一事件，"
        "不能原地重复选到同一时间。\n"
        "3. 既无未来 arrival 又无正在服务的完成事件时，立即结束循环。\n"
        "4. 三个主条件必须各自从空状态独立运行；全部条件完成后才汇总三项指标。\n"
    )


def _non_stdlib_imports(source: str, case_dir: Path) -> list[str]:
    allowed = set(sys.stdlib_module_names) | {"__future__"}
    try:
        tree = ast.parse(source, feature_version=(3, 9))
    except (OSError, SyntaxError):
        return []
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in allowed:
                    bad.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            root = (node.module or "").split(".")[0]
            if root and root not in allowed:
                bad.append(node.module or "")
    return sorted(set(bad))


def _response_content(body: dict[str, Any]) -> str:
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list):
        return ""
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
    return ""
