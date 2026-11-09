"""D-only bounded Markdown -> source -> review -> repair/review preparation."""
from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib

from src.llm.parsing import LLMError, LLMOutputError, LLMTruncatedError
from src.materials.models import MaterialError
from src.reproduction.errors import SourceGenerationError
from src.reproduction.logic_check import check_logic, review_state, output_schema_issues, literal_issues
from src.reproduction.name_checks import undefined_name_issues
from src.reproduction.source_checks import check_source_file
from src.reproduction.source_generation import SourceGenerator
from src.reproduction.reading_markdown import (
    reading_sources, generate_reading, save_reading, load_reading)

SOURCE_ATTEMPTS = 2
REVIEW_ATTEMPTS = 2
READING_ATTEMPTS = 2


def validate_version(attempt, bundle):
    """Every generated/repaired version gets the same deterministic checks."""
    check_source_file(attempt.source_path)
    source = attempt.source_path.read_text(encoding='utf-8')
    digest = hashlib.sha256(source.encode('utf-8')).hexdigest()
    if digest != attempt.source_sha256:
        raise ValueError('source changed after extraction')
    issues = undefined_name_issues(source) + literal_issues(source)
    issues += output_schema_issues(source, bundle.protocol)
    return source, issues


async def prepare_source(context, bundle, question, work_dir, diagnostics, evidence):
    diagnostics.record('preparation_policy', local_time_limit=None,
        source_attempts=SOURCE_ATTEMPTS, review_attempts_per_version=REVIEW_ATTEMPTS,
        reading_attempts=READING_ATTEMPTS)
    sources = reading_sources(bundle, evidence)
    document = None
    last_error = '没有完成Markdown阅读与代码审查'
    for index in range(READING_ATTEMPTS):
        diagnostics.record('reading_markdown_requested', attempt=index + 1, timeout_s=None)
        try:
            markdown = await generate_reading(context.llm, bundle, question, sources, timeout=None)
            document = save_reading(context.config, question, markdown)
            diagnostics.record('reading_markdown_saved', path=document.relative_path,
                markdown_sha256=document.sha256, model_authored=True, read_back_verified=True)
            break
        except (LLMError, MaterialError, ValueError, OSError, asyncio.TimeoutError) as exc:
            last_error = 'Markdown阅读未完成: ' + type(exc).__name__
            diagnostics.record('reading_markdown_incomplete', attempt=index + 1, reason=last_error)
            if isinstance(exc, (MaterialError, OSError)):
                break
            if isinstance(exc, LLMError) and not (exc.retryable or isinstance(exc, (LLMOutputError, LLMTruncatedError))):
                break
    if document is None:
        return None, last_error

    def current_bundle():
        return replace(bundle, paper_notes=load_reading(context.config, question, document))

    generator = SourceGenerator(context.llm, work_dir)
    previous = None
    for version in range(1, SOURCE_ATTEMPTS + 1):
        diagnostics.record('source_generation_requested', attempt=version, timeout_s=None,
            repair=previous is not None, markdown_path=document.relative_path,
            markdown_sha256=document.sha256)
        try:
            current_materials = current_bundle()
            if previous is None:
                attempt = await generator.generate(current_materials, question, timeout=None)
            else:
                attempt = await generator.repair(current_materials, question,
                    previous=previous, error=last_error, timeout=None)
        except SourceGenerationError as exc:
            last_error = str(exc)
            diagnostics.record('source_generation_error', attempt=version,
                reason=last_error, retryable=exc.retryable)
            if exc.retryable:
                continue
            break
        except (MaterialError, ValueError, OSError) as exc:
            return None, 'Markdown依据无法回读核验: ' + str(exc)
        if attempt.source_path is None:
            last_error = attempt.error or '未提取到完整源码'
            diagnostics.record('source_rejected', attempt=version, reason=last_error)
            previous = attempt
            continue
        try:
            source, issues = validate_version(attempt, bundle)
        except (ValueError, SyntaxError, OSError) as exc:
            issues = [str(exc)]
        if issues:
            last_error = '; '.join(issues)
            diagnostics.record('static_code_check', attempt=version, status='fail', issues=issues)
            previous = attempt
            continue
        diagnostics.record('static_code_check', attempt=version, status='pass',
                           source_sha256=attempt.source_sha256)
        state = 'incomplete'
        details = ['没有完成代码审查']
        for review_index in range(1, REVIEW_ATTEMPTS + 1):
            diagnostics.record('code_logic_check_requested', attempt=version,
                review_attempt=review_index, timeout_s=None,
                source_sha256=attempt.source_sha256, markdown_sha256=document.sha256)
            permanent = False
            try:
                current_materials = current_bundle()
                warnings, report = await check_logic(context.llm, current_materials,
                    question, source, timeout=None)
            except SourceGenerationError as exc:
                warnings = [str(exc)]
                report = {'source_sha256':attempt.source_sha256,
                          'review_unavailable':True, 'executed_locally':False}
                permanent = not exc.retryable
            except (MaterialError, ValueError, OSError) as exc:
                return None, 'Markdown依据无法回读核验: ' + str(exc)
            state, details = review_state(warnings, report, source)
            diagnostics.record('code_logic_check', attempt=version, review_attempt=review_index,
                review_status=state, issues=details, markdown_path=document.relative_path,
                markdown_sha256=document.sha256, **report)
            # Fail => one targeted code repair. Incomplete => retry the review
            # on the *same* source, never regenerate it as though a bug existed.
            if state in ('pass', 'fail') or permanent:
                break
        if state == 'pass':
            # Bind formal admission to the precise reviewed version and note.
            try:
                current_bundle()
                final_source, final_issues = validate_version(attempt, bundle)
            except (MaterialError, ValueError, SyntaxError, OSError) as exc:
                return None, '审查后代码或Markdown依据发生变化: ' + str(exc)
            if final_source != source or final_issues:
                return None, '审查后代码不再与通过版本一致'
            diagnostics.record('source_saved', attempt=version,
                source_sha256=attempt.source_sha256, review_status='pass',
                markdown_path=document.relative_path, markdown_sha256=document.sha256)
            return attempt, ''
        last_error = '; '.join(details)
        if state == 'incomplete':
            diagnostics.record('code_review_incomplete', reason=last_error,
                formal_client_attempted=False, regeneration_requested=False)
            return None, last_error
        previous = attempt
    return None, last_error
