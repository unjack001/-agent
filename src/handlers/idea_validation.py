"""Structural and explicit evidence-boundary checks for Idea reports.

These checks reject missing report elements and explicit claims that promote the
task's simulation to measured evidence. They do not prove arbitrary scientific
claims true; citation checks and the supplied PDF evidence remain necessary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.dispatch.models import CaseQuestion
from src.handlers.report_citations import cited_numbers, prose_only
from src.materials.models import PaperDigest

IDEA_SECTION_TITLES = (
    "研究问题与问题边界",
    "研究动机、已有方法局限和模拟观察的证据边界",
    "核心 Idea 与技术路线",
    "基线、数据集、实验设计、消融方案和量化指标",
    "指定论文的互补关系",
    "可行性、实施成本、失败边界和风险",
    "创新论证及可证伪方式",
)

_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$", re.M)
_PLACEHOLDER = re.compile(
    r"此处继续|继续完成不少于|示例占位|格式占位|"
    r"\b(?:TODO|TBD|lorem ipsum|to be filled)\b|"
    r"^\s*(?:[-*+]\s+)?(?:待补充|待填写|待完善)\s*[。.!！]?\s*$", re.I | re.M
)


@dataclass(frozen=True)
class _Section:
    title: str
    text: str


@dataclass(frozen=True)
class _Requirement:
    title: str
    headings: str
    concepts: tuple[tuple[str, str], ...]


_REQUIREMENTS = (
    _Requirement(IDEA_SECTION_TITLES[0], r"研究问题|问题边界|问题定义|research question|problem|scope", (
        ("研究问题", r"问题|question|problem"), ("范围边界", r"边界|范围|限定|scope|boundar|limit"))),
    _Requirement(IDEA_SECTION_TITLES[1],
                 r"动机|背景|缘起|动因|已有|现有|现存|证据|观察|相关工作|motivation|background|"
                 r"related work|evidence|observation|prior work", (
        ("研究动机或需求", r"动机|背景|需求|缘起|动因|出发点|为了|旨在|挑战|亟需|亟须|研究目标|"
                        r"motivat|background|need|aim|challenge"),
        ("已有方法", r"已有|现有|既有|现存|现行|传统|以往|前人|当前.{0,4}(?:方法|方案|研究)|"
                    r"prior|existing|previous|earlier"),
        ("已有方法的局限", r"局限|不足|缺陷|限制|瓶颈|难以|无法|受限|尚未解决|"
                         r"limit|gap|shortcoming|bottleneck|lack"),
        ("证据来源与范围", r"证据|摘录|原文|文献来源|论文(?:指出|提出|报告|显示|表明)|"
                         r"题目(?:给定|提供)|evidence|excerpt|source"))),
    _Requirement(IDEA_SECTION_TITLES[2], r"核心|技术路线|方法|方案|idea|method|approach|pipeline", (
        ("拟议机制", r"idea|思路|机制|方案|approach|mechanism"),
        ("技术路线", r"技术路线|流程|步骤|pipeline|workflow|method"))),
    _Requirement(IDEA_SECTION_TITLES[3],
                 r"实验|基线|数据|消融|指标|experiment|baseline|data|ablat|metric|evaluat", (
        ("基线", r"基线|对照|baseline"), ("数据来源", r"数据|dataset|data set"),
        ("实验步骤", r"实验|experiment"), ("消融", r"消融|ablat"), ("量化指标", r"指标|metric"))),
    _Requirement(IDEA_SECTION_TITLES[4], r"互补|论文关系|文献关系|complement|paper relationship|synthesis", (
        ("论文互补或材料不足说明", r"互补|协同|结合|complement|synthes|combin|仅.{0,8}一篇|单篇|only one|single paper"),)),
    _Requirement(IDEA_SECTION_TITLES[5], r"可行|成本|风险|失败|feasib|cost|risk|failure|limitation", (
        ("可行性", r"可行|feasib"), ("实施成本", r"成本|开销|cost|overhead"),
        ("失败情形", r"失败|失效|failure|fail"), ("适用边界", r"边界|条件|boundar|condition"),
        ("风险", r"风险|risk"))),
    _Requirement(IDEA_SECTION_TITLES[6], r"创新|证伪|novel|innovat|falsif", (
        ("创新主张", r"创新|novel|innovat"), ("可证伪条件", r"证伪|推翻|falsif|refut"))),
)


def idea_content_checklist() -> str:
    """Describe the same body requirements used by validation and repair."""
    return "\n".join(
        f"- {item.title}：正文说明" + "、".join(name for name, _ in item.concepts) + "。"
        for item in _REQUIREMENTS
    )


def _sections(body: str) -> tuple[str, list[_Section]]:
    text = prose_only(body)
    headings = list(_HEADING.finditer(text))
    sections: list[_Section] = []
    for index, match in enumerate(headings):
        depth = len(match.group(1))
        # A document title is not a content section. Subsequent first-level
        # headings are valid Markdown sections just like second-level ones.
        if depth == 1 and index == 0:
            continue
        end = next(
            (later.start() for later in headings[index + 1 :] if len(later.group(1)) <= depth),
            len(text),
        )
        content = _HEADING.sub("", text[match.end() : end]).strip()
        sections.append(_Section(match.group(2).strip(), content))
    return text, sections


def _has_content(text: str) -> bool:
    # Punctuation, citation numbers and empty list/table scaffolding are not prose.
    text = re.sub(r"\[[^\]\n]+\]", "", text)
    return len(re.findall(r"[\u4e00-\u9fffA-Za-z]", text)) >= 8


def _section_text(sections: list[_Section], heading_pattern: str) -> str:
    selected = [
        section for section in sections if re.search(heading_pattern, section.title, re.I)
    ]
    # Headings choose the relevant section; they cannot themselves satisfy the
    # required content elements, or a complete outline plus filler would pass.
    return "\n".join(section.text for section in selected)


def _boundary_issues(text: str, question: CaseQuestion) -> list[str]:
    task = question.input + "\n" + "\n".join(question.constraints)
    simulated = r"模拟(?:观察|结果|现象)|simulat(?:ed|ion) observations?"
    absent = (
        r"(?:未(?:提供|给出|附带|附上|随附)|没有(?:提供|给出|附带)?|"
        r"不(?:提供|包含|含有)|不含|无)"
        r"(?:任何|具体|相关|对应|可用|额外|明确|给定|的|一项|一组)*"
        rf"(?:{simulated})|\b(?:no|without)\s+(?:{simulated})"
    )
    # Mentioning that no observations were supplied must not invent a duty
    # to attribute nonexistent observations to the task. Positive observations
    # elsewhere in the same task remain subject to all existing checks.
    task_simulated = bool(re.search(simulated, re.sub(absent, "", task, flags=re.I), re.I))
    explicit_task_simulated = (
        rf"(?:题目|本题).{{0,12}}(?:{simulated})|"
        rf"(?:task|question).{{0,20}}(?:{simulated})"
    )
    if not task_simulated and not re.search(
        explicit_task_simulated, re.sub(absent, "", text, flags=re.I), re.I
    ):
        return []
    issues: list[str] = []
    pending = (
        r"待验证|待分析|未经验证|尚未验证|待检验|有待.{0,8}(?:检验|验证|分析)|"
        r"仍需.{0,8}(?:检验|验证|分析)|只是.{0,8}现象|仅.{0,8}现象|"
        r"unverified|unvalidated|hypothes[ie]s|"
        r"未经检验|尚待检验|尚待验证|缺乏实测|尚无.{0,10}实测|"
        r"仅.{0,10}(?:模拟设定|测试设定|题设)|"
        r"not.{0,12}(?:verified|validated)|to be (?:tested|verified|validated)"
    )
    # A clear explanation may span two sentences in the same paragraph.
    # Headings and unrelated paragraphs cannot supply the disclaimer.
    prose = _HEADING.sub("", text)
    pending_simulation = any(
        re.search(
            rf"(?:{simulated})[\s\S]{{0,300}}(?:{pending})|"
            rf"(?:{pending})[\s\S]{{0,300}}(?:{simulated})", paragraph, re.I,
        ) for paragraph in re.split(r"\n\s*\n", prose)
    )
    if task_simulated and not pending_simulation:
        issues.append("必须明确说明题目模拟观察只是待分析、待验证现象")
    if task_simulated and not re.search(
        r"(?:尚未|未曾|没有|尚无|还未|未)"
        r"(?:实际|直接|充分|系统|进行|开展|完成|经过|被|实验|任何|的){0,4}"
        r"(?:测试|验证|评估|证明)[^，。;；\n]{0,30}(?:替代|策略|改进|方案|效果)|"
        r"(?:替代|策略|改进|方案|效果)[^，。;；\n]{0,24}(?:尚未|没有|未)"
        r"(?:实际|进行|开展|被|得到|获得|实验|任何|的){0,4}(?:测试|验证|评估)|"
        r"(?:no|not|never|haven.t|hasn.t).{0,35}(?:tested|evaluated|validated|verified)"
        r".{0,40}(?:alternative|strateg|improvement|benefit)|"
        r"(?:alternative|strateg|improvement|benefit).{0,40}"
        r"(?:untested|unverified|unvalidated|not.{0,12}(?:tested|evaluated|validated))",
        text, re.I,
    ):
        issues.append("必须说明替代策略或所提方案的改进效果尚未测试、验证")

    # A disclaimer in another sentence/clause cannot excuse a positive false claim.
    clauses = re.split(r"[。！？!？\n；;，]|\b(?:but|however|yet)\b|但是|然而|可是|但", text)
    for clause in clauses:
        if re.search(r"不得|不能|不可|不应|并非|不是|尚未|未曾|没有|尚无|未经|未被|"
                     r"不(?:属于|等同于|等于|意味着|构成|代表|来自)|"
                     r"\b(?:not|never|cannot|no)\b", clause, re.I):
            continue
        simulation_subject = simulated if task_simulated else explicit_task_simulated
        if re.search(simulation_subject, clause, re.I) and re.search(
            rf"(?:{simulated}).{{0,20}}(?:就是|即为|属于|源自|来自|出自|取自|等同于|是|为)"
            r".{0,16}(?:论文|文献).{0,12}(?:实测|结论|实验结果|研究发现)|"
            rf"(?:{simulated}).{{0,16}}(?:已经|已被|已由|已通过)"
            r".{0,20}(?:实测|验证|实验结果|论文结论)|"
            r"(?:论文|文献).{0,12}(?:实测|实验结果|实验验证).{0,8}"
            rf"(?:证明|证实|验证|表明|支持).{{0,20}}(?:{simulated})|"
            r"(?:paper|publication).{0,25}(?:measured|confirmed|proved|validated)"
            rf".{{0,40}}(?:{simulated})|"
            rf"(?:{simulated}).{{0,20}}(?:is|are|was|were|constitute|come from)"
            r".{0,30}(?:paper|publication).{0,20}(?:measured|finding|experiment|result)",
            clause, re.I,
        ):
            issues.append("不得将题目模拟观察归为论文实测结论或已完成实验")
            break
    for clause in clauses:
        if re.search(r"尚未|未曾|没有|尚无|未经|不能|不应|不得|计划|拟验证|待验证|"
                     r"如果|假如|倘若|假设|若.{0,8}实验|"
                     r"(?:将|拟).{0,8}(?:检验|验证|评估|测试|证明|表明)|"
                     r"\b(?:not|never|cannot|no|will|planned|hypothes[ie]s|if|unless)\b",
                     clause, re.I):
            continue
        if re.search(
            r"(?:本研究|本次实验|我们|所提方案|提出的方案).{0,45}"
            r"(?:已经|已).{0,12}(?:证明|证实|验证|实测|实现|取得)|"
            r"(?:实验|评估|测试|实测)(?:结果|结论).{0,6}(?:表明|显示|证明|证实|验证)"
            r".{0,40}(?:所提(?:方案|方法|机制)|本(?:方案|方法)|提出的(?:方案|方法)|我们)|"
            r"(?:our|proposed|this experiment).{0,45}"
            r"(?:already|has|have).{0,15}(?:proved|confirmed|validated|measured|achieved)",
            clause, re.I,
        ):
            issues.append("所提方案的效果尚待验证，不得写成已完成实验或已取得改进")
            break
    return issues


def validate_idea_body(
    body: str, question: CaseQuestion, digests: list[PaperDigest]
) -> list[str]:
    """Return actionable issues; an empty list means the mechanical checks pass."""
    text, sections = _sections(body)
    issues: list[str] = []
    if not sections:
        issues.append("Idea 正文必须使用清晰的 Markdown 分节标题（## 或更深级别）")
    for section in sections:
        if not _has_content(section.text):
            issues.append(f"章节“{section.title}”缺少实质正文")
    if _PLACEHOLDER.search(text) or re.search(r"(?m)^\s*(?:…{2,}|\.{3,})\s*$", text):
        issues.append("报告仍含占位或待补充内容，请提交完整正文")

    for requirement in _REQUIREMENTS:
        section_text = _section_text(sections, requirement.headings)
        missing = [name for name, pattern in requirement.concepts
                   if not re.search(pattern, section_text, re.I)]
        if missing:
            issues.append(
                f"缺少分节内容或必要要素：{requirement.title}；"
                f"缺少的正文要素：{'、'.join(missing)}。请在该节正文补充实质说明，标题不能代替正文"
            )

    complement = _section_text(sections, _REQUIREMENTS[4].headings)
    if complement:
        if len(digests) == 1 and not re.search(
            r"仅.{0,8}一篇|只有.{0,8}一篇|单篇|single paper|only one", complement, re.I
        ):
            issues.append("仅有一篇指定论文时，应明确不能进行跨论文互补比较")
        elif len(digests) > 1:
            if len(cited_numbers(complement, len(digests))) < 2:
                issues.append("多篇论文互补关系须具体关联并引用至少两篇指定论文")
    issues.extend(_boundary_issues(text, question))
    return list(dict.fromkeys(issues))
