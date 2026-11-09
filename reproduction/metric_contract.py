"""Current-task metric contracts and public numerical checks; never judge data."""
from __future__ import annotations

from copy import deepcopy
import math
import re

from src.materials.models import MaterialError


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def public_reference(spec):
    value = spec.get('reference')
    return value if finite(value) and spec.get('reference_verified') is not False else None


def input_contract(protocol, input_specs):
    """Use input names/units; retain metadata only for identical public metrics."""
    result = deepcopy(protocol or {})
    if len(input_specs) != 3:
        return result if isinstance(result.get('metrics'), list) and len(result['metrics']) == 3 else None
    existing = result.get('metrics', [])
    by_pair = {(s.get('name'), s.get('unit')): s for s in existing if isinstance(s, dict)} if isinstance(existing, list) else {}
    result['metrics'] = [dict(by_pair.get((s['name'], s['unit']), {}), name=s['name'], unit=s['unit']) for s in input_specs]
    return result


def resolve_contract(protocol, input_specs, question_text=''):
    result = input_contract(protocol, input_specs)
    if result is None:
        raise MaterialError('current public materials must independently declare exactly three metrics')
    specs = result['metrics']
    if any(not isinstance(s, dict) or not isinstance(s.get('name'), str) or not s['name'].strip()
           or not isinstance(s.get('unit'), str) or not s['unit'].strip() for s in specs):
        raise MaterialError('current task metrics require non-empty exact names and units')
    if len({(s.get('name'), s.get('unit')) for s in specs}) != 3:
        raise MaterialError('current task declares duplicate metric names/units')
    existing = (protocol or {}).get('metrics')
    if len(input_specs) == 3 and existing is not None:
        declared = {(s['name'], s['unit']) for s in input_specs}
        public = {(s.get('name'), s.get('unit')) for s in existing if isinstance(s, dict)}
        if declared != public:
            raise MaterialError('current input metric names/units conflict with the supplied public protocol; no formal run started')
    if question_text:
        compact = re.sub(r'\s+', '', question_text).lower()
        recognized = True
        if '100*abs(observed-reference)/abs(reference)' in compact or re.search(r'相对误差.{0,8}百分比|percentage relative error', question_text, re.I):
            result['relative_error_formula'] = 'percentage_relative_error'
        elif 'abs(observed-reference)/abs(reference)' in compact or 'abs((observed-reference)/reference)' in compact or re.search(r'绝对相对误差|absolute relative error', question_text, re.I):
            result['relative_error_formula'] = 'absolute_relative_error'
        elif 'abs(observed-reference)/reference' in compact:
            result['relative_error_formula'] = 'absolute_difference_signed_reference'
        elif '(observed-reference)/reference' in compact or 'observed/reference-1' in compact or 'signed relative error' in question_text.lower():
            result['relative_error_formula'] = 'signed_relative_error'
        elif re.search(r'(?:relative_error|相对误差)\s*=\s*abs\(observed\s*-\s*reference\)\s*(?:[;；。\n]|$)', question_text, re.I):
            result['relative_error_formula'] = 'absolute_error'
        else:
            recognized = False
        explicit = re.search(r'(?:relative_error|相对误差)\s*(?:=|定义为|计算为|公式为|按)\s*([^\n。;；]+)', question_text, re.I)
        if not recognized and explicit and re.search(r'observed|reference|实测|参考|平方|对数|双曲|custom|自定义', explicit[1], re.I):
            raise MaterialError('current input declares an unsupported custom error formula; cannot substitute the default convention')
    for spec in specs:
        try:
            expected_error(1.0, 1.0, spec, result)
        except ValueError as exc:
            raise MaterialError(str(exc)) from exc
    return result


def expected_error(reference, observed, spec, protocol):
    formula = spec.get('relative_error_formula', protocol.get('relative_error_formula', 'absolute_relative_error'))
    if not isinstance(formula, str):
        raise ValueError('declared relative-error convention must be a supported string')
    aliases = {'abs(observed-reference)/abs(reference)': 'absolute_relative_error',
               '(observed-reference)/reference': 'signed_relative_error',
               '100*abs(observed-reference)/abs(reference)': 'percentage_relative_error'}
    formula = aliases.get(formula, formula)
    if formula not in ('absolute_relative_error', 'signed_relative_error', 'percentage_relative_error', 'absolute_error', 'absolute_difference_signed_reference'):
        raise ValueError('unsupported declared relative-error convention; cannot verify returned error')
    if formula == 'absolute_error':
        return abs(observed - reference)
    if reference == 0:
        policy = spec.get('zero_reference_policy', protocol.get('zero_reference_policy', 'null'))
        if policy == 'zero_when_equal' and observed == 0:
            return 0.0
        return None
    if formula == 'signed_relative_error':
        return (observed - reference) / reference
    if formula == 'absolute_difference_signed_reference':
        return abs(observed - reference) / reference
    ratio = abs(observed - reference) / abs(reference)
    return 100 * ratio if formula == 'percentage_relative_error' else ratio


def metric_value_issues(metric, spec, protocol):
    issues = []
    known = public_reference(spec)
    reference, observed, error = (metric.get(k) for k in ('reference', 'observed', 'relative_error'))
    if known is not None and (not finite(reference) or not math.isclose(reference, known, rel_tol=1e-9, abs_tol=1e-12)):
        issues.append('reference differs from the current verifiable public reference')
    if reference is None or observed is None:
        if error is not None:
            issues.append('relative_error must be null without reference and observed values')
        return issues
    if not finite(reference) or not finite(observed):
        return issues  # The caller separately rejects non-finite values.
    try:
        expected = expected_error(reference, observed, spec, protocol)
    except ValueError as exc:
        return issues + [str(exc)]
    if expected is None:
        if error is not None:
            issues.append('relative_error is undefined at a zero reference under the declared convention')
    elif not finite(error) or not math.isclose(error, expected, rel_tol=1e-6, abs_tol=5.0001e-5):
        # Accept the four-decimal rounding used by the public output examples.
        issues.append('relative_error does not match the declared calculation')
    return issues
