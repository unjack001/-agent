"""Parse only explicit task quotas and independent scoping questions."""
from __future__ import annotations
import re


def _number(word):
    if word.isascii() and word.isdigit():
        return int(word)
    digits = dict(zip('零一二三四五六七八九', range(10)))
    if word in digits:
        return digits[word]
    if '十' in word:
        left, right = word.split('十', 1)
        return digits.get(left, 1) * 10 + digits.get(right, 0)
    return None


def task_text(question):
    return '\n'.join([question.input, *question.constraints, question.expected_output])


_N = r'([0-9]{1,3}|[一二三四五六七八九十]{1,3})'


def citation_minimum(question, paper_count):
    text = task_text(question)
    required = 1
    if re.search(r'(?:引用|比较|综述|讨论|涵盖).{0,8}(?:全部|所有|每[一]?篇).{0,8}(?:论文|文献)|(?:cite|cover|compare)\s+all\b', text, re.I):
        required = paper_count
    for pattern in [r'(?:引用|参考文献)[^。；;\n]{0,12}?(?:至少|不少于)\s*'+_N+r'\s*篇',
                    r'(?:至少|不少于)\s*引用\s*'+_N+r'\s*篇',
                    r'(?:列出|列示|附上)[^。；;\n]{0,12}?(?:至少|不少于)\s*'+_N+r'\s*篇(?:实际引用的|指定的|指定)*(?:论文|文献)',
                    r'cite\s+at\s+least\s+'+_N+r'\s+papers?']:
        for match in re.finditer(pattern, text, re.I):
            value = _number(match[1])
            if value is not None:
                required = max(required, value)
    for match in re.finditer(r'(?:引用|cite)\s*'+_N+r'\s*(?:篇|papers?)', text, re.I):
        if re.search(r'最多|至多|不超过|at most|up to', text[max(0, match.start()-10):match.start()], re.I):
            continue
        value = _number(match[1])
        if value is not None:
            required = max(required, value)
    return required


def survey_citation_minimum(question, paper_count):
    """Restore the passing C default of five, honoring explicit smaller quotas."""
    required = citation_minimum(question, paper_count)
    text = task_text(question)
    explicit_one = re.search(
        r'(?:引用|cite)\s*(?:至少|不少于|at least)?\s*(?:1|一)\s*(?:篇|paper)|'
        r'(?:至少|不少于)\s*(?:引用)?\s*(?:1|一)\s*篇', text, re.I)
    return required if required != 1 or explicit_one else min(5, paper_count)


def paper_count_bounds(question):
    text = task_text(question)
    lower, upper = 0, None
    for pattern, mode in [
        (r'(?:至少|不少于)\s*(?:选[出取择]?|返回|检索|筛选|推荐)?\s*'+_N+r'\s*篇', 'min'),
        (r'(?:最多|不超过|至多)\s*(?:选[出取择]?|返回|推荐)?\s*'+_N+r'\s*篇', 'max'),
        (r'(?:恰好|正好|exactly)\s*(?:选[出取择]?|返回)?\s*'+_N+r'\s*(?:篇|papers?)', 'exact'),
        (r'(?:at least)\s*'+_N+r'\s+papers?', 'min'),
        (r'(?:at most|up to)\s*'+_N+r'\s+papers?', 'max')]:
        for match in re.finditer(pattern, text, re.I):
            value = _number(match[1])
            if value is None:
                continue
            if mode in ('min', 'exact'):
                lower = max(lower, value)
            if mode in ('max', 'exact'):
                upper = min(upper, value) if upper is not None else value
    for match in re.finditer(r'(?:选择|选取|选出|筛选|返回|推荐|列出|return|select|retrieve)\s*'+_N+r'\s*(?:篇|papers?)', text, re.I):
        if re.search(r'至少|不少于|最多|不超过|至多|at least|at most|up to', text[max(0, match.start()-12):match.start()], re.I):
            continue
        value = _number(match[1])
        if value is not None:
            lower = max(lower, value)
            upper = min(upper, value) if upper is not None else value
    return lower, upper


def scoping_question_issues(value):
    # Quoted examples do not create additional questions. One compound scope
    # question remains valid; reject separate question sentences or list items.
    prose = re.sub(r'“[^”]*”|「[^」]*」|‘[^’]*’|"[^"\n]*"|(?<!\w)\x27[^\x27\n]+\x27(?!\w)|`[^`]*`', '', value)
    questions = re.findall(r'[^?？\n]+[?？]', prose)
    if len(questions) > 1:
        return ['clarifying_question 只能提出一个关键澄清问题，不能列出多个独立问句']
    lines = [line for line in prose.splitlines() if re.match(r'\s*(?:[-*]|\d+[.)、])\s*\S', line)]
    if len(lines) > 1:
        return ['clarifying_question 不能拆成多个独立问题条目']
    if not questions and not re.search(r'什么|哪|如何|是否|吗|能否|请.{0,12}(?:明确|限定|说明)|\b(?:what|which|who|where|when|how|whether|could|would|can|do|does|is|are)\b', prose, re.I):
        return ['clarifying_question 应提出一个用于限定研究范围的问题']
    return []
