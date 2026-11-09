"""Load the bundled skill only for B/D/E; no tool loop or model call here."""
from functools import lru_cache
from pathlib import Path
import hashlib

ROOT=Path(__file__).resolve().parents[2]/'skills'/'read-paper-zr'
ACTIVE_ROLES=frozenset(('B','D','E'))
MARKER='[ACTIVE_SKILL: read-paper-zr'

def section(text,heading):
    lines=text.splitlines();start=next((i for i,s in enumerate(lines) if s==heading),None)
    if start is None:raise ValueError('missing skill section: '+heading)
    end=next((i for i in range(start+1,len(lines)) if lines[i].startswith('## ')),len(lines))
    return '\n'.join(lines[start:end])

@lru_cache(maxsize=3)
def instructions(role):
    if role not in ACTIVE_ROLES:raise ValueError('paper reading is enabled only for B/D/E')
    original=(ROOT/'SKILL.md').read_text(encoding='utf-8-sig')
    guide=(ROOT/'references/deep-reading-guide.md').read_text(encoding='utf-8-sig')
    # Load the user's actual mechanism/evidence rules, excluding its note-output
    # sections and the removed limitations/inspiration modules.
    excerpts=[section(original,'## 阅读与重建'),section(guide,'## 2. 从模块名到可执行机制'),
              section(guide,'## 4. 实验要能区分解释'),section(guide,'## 7. 交付前用正文回答')]
    policy=(ROOT/'contest-adapter.md').read_text(encoding='utf-8')
    profile=(ROOT/'profiles'/f'{role}.md').read_text(encoding='utf-8')
    return f'{MARKER} | role={role}]\n'+policy+'\n'+profile+'\n原技能核心阅读规则：\n'+'\n\n'.join(excerpts)

def apply_skill(messages,role):
    result=[dict(m) for m in messages]
    if role not in ACTIVE_ROLES:return result
    text=instructions(role)
    if result and result[0].get('role')=='system':
        if MARKER not in result[0]['content']:result[0]['content']+='\n\n'+text
    else:result.insert(0,{'role':'system','content':text})
    return result

def skill_metadata(role):
    text=instructions(role)
    return {'skill':'read-paper-zr','role':role,'adapter_version':'contest-20261106',
            'instructions_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'extra_model_calls':0,'removed_modules':['局限与启发']}
