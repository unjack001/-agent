"""B admission gate: complete criterion checks grounded in this paper's excerpts."""
from collections import Counter
import re

GLOBAL_RULE = re.compile(
    r'\d+\s*(?:篇|papers?)|[一二三四五六七八九十]+\s*篇|JSON|Markdown|'
    r'(?:输出|返回|保存)(?:格式|字段|路径|到|为)|排序|文件名|filename', re.I)


def compact(text):
    return ''.join(text.split()).casefold()


def requirements(question, criteria):
    # The full task remains in the prompt. This always-present check prevents
    # failed/incomplete criterion extraction from making admission vacuous.
    required = [{'id': 'T0', 'requirement': (
        '论文自身实际研究的核心对象、问题与方法符合原题。篇数、输出形式、排序和优先偏好不是单篇必要条件。')}]
    exclusions = []
    original = compact(question.title + '\n' + question.input + '\n' + '\n'.join(question.constraints))
    preferences = {compact(text) for text in criteria['focus']}
    ignored = []
    for field, target, prefix in (('mandatory', required, 'R'), ('exclude', exclusions, 'X')):
        seen = set()
        for text in criteria[field]:
            key = compact(text)
            if (not key or key not in original or key in seen or GLOBAL_RULE.search(text) or
                    field == 'mandatory' and (key in preferences or re.search(
                        r'优先|最好|尽量|可选|建议|\bprefer(?:red|ably)?\b|\boptional\b|\bideally\b', text, re.I))):
                ignored.append({'field': field, 'text': text})
                continue
            seen.add(key)
            target.append({'id': prefix + str(len(target)), 'requirement': text})
    return required, exclusions, ignored


def grounded(proof, doc):
    """Match a contiguous quote to the exact page of this supplied paper."""
    if not isinstance(proof, dict) or type(proof.get('page')) is not int:
        return False
    quote = proof.get('quote')
    if not isinstance(quote, str) or len(compact(quote)) < 8:
        return False
    return any(page['page'] == proof['page'] and compact(quote) in compact(page['text'])
               for page in doc['evidence']['pages'])


def assess(item, doc, required, exclusions):
    """Return a completed decision; invalid evidence becomes uncertain, never include."""
    if (not isinstance(item, dict) or item.get('decision') not in ('include', 'exclude', 'uncertain') or
            not isinstance(item.get('reason'), str) or not item['reason'].strip() or
            not isinstance(item.get('checks'), list)):
        return None
    expected = {r['id']: 'supported' for r in required}
    expected.update({r['id']: 'clear' for r in exclusions})
    counts = Counter(check.get('criterion') for check in item['checks']
                     if isinstance(check, dict) and isinstance(check.get('criterion'), str))
    satisfied = set()
    contradicted = set()
    issues = []
    for check in item['checks']:
        if not isinstance(check, dict):
            issues.append('invalid_check')
            continue
        key = check.get('criterion')
        if not isinstance(key, str) or key not in expected or counts[key] != 1:
            issues.append('unknown_or_duplicate_criterion')
            continue
        proofs = check.get('evidence')
        valid = (isinstance(proofs, list) and bool(proofs) and
                 all(grounded(proof, doc) for proof in proofs) and
                 isinstance(check.get('explanation'), str) and bool(check['explanation'].strip()))
        if not valid:
            issues.append('ungrounded:' + key)
            continue
        if check.get('status') == expected[key]:
            satisfied.add(key)
        elif check.get('status') == ('violated' if expected[key] == 'clear' else 'contradicted'):
            contradicted.add(key)
    missing = sorted(set(expected) - satisfied)
    role = item.get('study_role')
    own_research = role == 'actual_research' and grounded(item.get('role_evidence'), doc)
    if contradicted or role == 'background_only' and grounded(item.get('role_evidence'), doc):
        decision = 'exclude'
    elif (item['decision'] == 'include' and not missing and not issues and own_research):
        decision = 'include'
    else:
        decision = 'uncertain'
    if not own_research:
        issues.append('own_research_not_confirmed')
    return {'id': doc['id'], 'filename': doc['filename'], 'decision': decision,
            'reason': item['reason'].strip(), 'missing': missing, 'issues': issues,
            'checks': item['checks'], 'study_role': role, 'role_evidence': item.get('role_evidence')}
