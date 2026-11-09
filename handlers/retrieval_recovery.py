"""B-only token-limit recovery: complete JSON records, then smaller batches."""
from collections import deque
import asyncio,json,math,time
from src.llm.parsing import LLMError,LLMOutputError,LLMTruncatedError
from src.llm.structured import call_json,validate_json_text

def recover_review_prefix(body,criteria=None,*,array_key='decisions'):
    """Decode closed values only; never append guessed brackets or quote text."""
    result={array_key:[]}
    if criteria is not None:result['criteria']=criteria
    try:text=body['choices'][0]['message']['content'].lstrip()
    except (KeyError,IndexError,TypeError,AttributeError):return result
    if text.startswith('```'):
        parts=text.split('\n',1)
        if len(parts)<2:return result
        text=parts[1].lstrip()
    if not text.startswith('{'):return result
    def unique(pairs):
        obj={}
        for key,value in pairs:
            if key in obj:raise ValueError('duplicate JSON key')
            obj[key]=value
        return obj
    def constant(value):raise ValueError('nonfinite JSON constant')
    def finite(value):
        value=float(value)
        if not math.isfinite(value):raise ValueError('nonfinite JSON number')
        return value
    decoder=json.JSONDecoder(object_pairs_hook=unique,parse_constant=constant,parse_float=finite)
    def ws(i):
        while i<len(text) and text[i].isspace():i+=1
        return i
    i=1;seen=set()
    try:
        while True:
            i=ws(i)
            if i>=len(text) or text[i]=='}':break
            key,i=decoder.raw_decode(text,i)
            if not isinstance(key,str) or key in seen:return {array_key:[],'criteria':criteria or []}
            seen.add(key);i=ws(i)
            if i>=len(text) or text[i]!=':':break
            i=ws(i+1)
            if key==array_key:
                if i>=len(text) or text[i]!='[':break
                i+=1
                while True:
                    i=ws(i)
                    if i>=len(text):return result
                    if text[i]==']':i+=1;break
                    item,i=decoder.raw_decode(text,i)
                    if isinstance(item,dict):
                        try:validate_json_text(item)
                        except LLMOutputError:pass
                        else:result[array_key].append(item)
                    i=ws(i)
                    if i>=len(text):return result
                    if text[i]==',':i+=1;continue
                    if text[i]==']':i+=1;break
                    return result
            else:
                value,i=decoder.raw_decode(text,i)
                if key=='criteria' and criteria is None and isinstance(value,list):
                    validate_json_text(value);result['criteria']=value
            i=ws(i)
            if i<len(text) and text[i]==',':i+=1;continue
            break
    except (ValueError,TypeError,RecursionError,LLMOutputError):
        pass
    return result

async def review_in_batches(llm,question,papers,catalog,*,end,criteria=None,max_tokens=6000,on_progress=None):
    from src.handlers.retrieval_check import review_messages,assess_selection
    accepted=set();excluded=set();records={};events=[];frozen=criteria
    queue=deque([(list(papers),False)])
    while queue and end-time.monotonic()>0:
        batch,compact=queue.popleft();names={p['filename'] for p in batch}
        evidence=[e for e in catalog if e.filename in names]
        messages=review_messages(question,batch,evidence,frozen)
        messages.append({'role':'user','content':(
            '复核输出预算：criteria放在decisions之前且只写一次；已有required_criteria时可省略criteria。'
            '每篇只返回一个decision；逐条件引用最短充分的原文片段，explanation用一句话，不重复全文。'
            + ('上一复核达到token上限，本次只检查这个较小批次，独立返回完整JSON，不续写残片。' if compact else ''))})
        event={'filenames':[p['filename'] for p in batch],'compact':compact}
        truncated=False
        try:
            data=await call_json(llm,messages,timeout=max(0,end-time.monotonic()),max_tokens=max_tokens,temperature=0)
        except LLMTruncatedError as exc:
            truncated=True;data=recover_review_prefix(exc.response_body,frozen)
            event['status']='truncated'
        except (LLMError,ValueError,asyncio.TimeoutError) as exc:
            event.update(status='review_unavailable',error_type=type(exc).__name__)
            events.append(event);break
        else:event['status']='complete'
        if not isinstance(data,dict):data={}
        if frozen is not None:data=dict(data,criteria=frozen)
        good,bad,_,warnings=assess_selection(data,batch,evidence,question)
        if frozen is None and isinstance(data.get('criteria'),list):
            _,_,_,notes=assess_selection(data,[],[],question)
            if not any(n in ('条件编号重复','没有可核实的论文入选硬条件') for n in notes):frozen=data['criteria']
        accepted.update(good);excluded.update(bad)
        decisions=data.get('decisions',[]);completed=set();counts={}
        if isinstance(decisions,list):
            for item in decisions:
                if isinstance(item,dict) and isinstance(item.get('filename'),str):
                    counts[item['filename']]=counts.get(item['filename'],0)+1
            for item in decisions:
                if (isinstance(item,dict) and isinstance(item.get('filename'),str) and item['filename'] in names
                    and counts[item['filename']]==1 and item.get('decision') in ('include','exclude','uncertain')
                    and isinstance(item.get('checks'),list)):
                    completed.add(item['filename']);records[item['filename']]=item
        # Without real hard criteria even a complete record has no admission
        # basis. Recheck it instead of treating malformed metadata as proof.
        if frozen is None:completed=set()
        missing=[p for p in batch if p['filename'] not in completed|good|bad]
        event.update(accepted=sorted(good),excluded=sorted(bad),warnings=warnings,
                     recovered_complete_records=len(completed) if truncated else 0,
                     remaining_after_truncation=[p['filename'] for p in missing] if truncated else [])
        events.append(event)
        if on_progress is not None:
            on_progress({'accepted':sorted(accepted),'excluded':sorted(excluded),'event':event})
        if truncated and missing:
            if len(batch)==1:
                if not compact:queue.appendleft((missing,True))
            else:
                half=max(1,(len(missing)+1)//2)
                pieces=[missing[i:i+half] for i in range(0,len(missing),half)]
                for piece in reversed(pieces):queue.appendleft((piece,True))
    unresolved={p['filename'] for p in papers}-accepted-excluded
    return accepted,excluded,unresolved,{'criteria':frozen or [],'decisions':list(records.values()),
                                      'batch_events':events,'recovery':'complete_records_then_smaller_batches'}
