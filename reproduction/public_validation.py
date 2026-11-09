"""Deterministic comparison of explicit PUBLIC example results, never code execution."""
from __future__ import annotations
import math

def public_cases(sample):
    result=[]
    def visit(value,path):
        if isinstance(value,list):
            for i,item in enumerate(value):visit(item,f'{path}/{i}')
        elif isinstance(value,dict):
            if 'request' in value:
                for key in ('expected_response','expected','response','output'):
                    if key in value:
                        result.append({'id':path,'request':value['request'],'expected':value[key]});return
            for key in ('cases','examples','tests','samples'):
                if key in value:visit(value[key],f'{path}/{key}')
    visit(sample,'sample')
    return result

def compare_values(expected,actual,*,absolute=1e-8,relative=1e-6,path='$'):
    if isinstance(expected,bool) or expected is None:
        return [] if type(actual)==type(expected) and actual==expected else [path+': value/type mismatch']
    if isinstance(expected,(int,float)):
        return [] if (isinstance(actual,(int,float)) and not isinstance(actual,bool) and
            math.isfinite(actual) and math.isclose(expected,actual,abs_tol=absolute,rel_tol=relative)) else [path+': numerical mismatch']
    if isinstance(expected,dict):
        if not isinstance(actual,dict):return [path+': expected object']
        issues=[]
        for key,value in expected.items():
            if key not in actual:issues.append(path+'.'+key+': missing field')
            else:issues+=compare_values(value,actual[key],absolute=absolute,relative=relative,path=path+'.'+key)
        return issues
    if isinstance(expected,list):
        if not isinstance(actual,list) or len(expected)!=len(actual):return [path+': list length/type mismatch']
        return [issue for i,(e,a) in enumerate(zip(expected,actual)) for issue in compare_values(
            e,a,absolute=absolute,relative=relative,path=f'{path}[{i}]')]
    return [] if type(actual)==type(expected) and actual==expected else [path+': value/type mismatch']

def compare_public_traces(cases,data,source):
    traces=data.get('sample_traces',[]) if isinstance(data,dict) else []
    if not isinstance(traces,list):return [],['公开样例推演格式无效']
    byid={};duplicates=set()
    for item in traces:
        if isinstance(item,dict) and isinstance(item.get('id'),str):
            if item['id'] in byid:duplicates.add(item['id'])
            byid[item['id']]=item
    mismatches=[];warnings=[];normalized=' '.join(source.split())
    for case in cases:
        item=byid.get(case['id'],{});quote=item.get('code_evidence');steps=item.get('steps')
        if (case['id'] in duplicates or not isinstance(quote,str) or not quote.strip() or
            ' '.join(quote.split()) not in normalized or not isinstance(steps,str) or len(steps.strip())<12 or 'predicted' not in item):
            warnings.append(case['id']+': 无完整源码推演，未验证');continue
        for issue in compare_values(case['expected'],item['predicted']):
            mismatches.append(case['id']+': '+issue)
    return mismatches,warnings

def metric_range_issues(submission,protocol):
    specs={(s.get('name'),s.get('unit')):s for s in protocol.get('metrics',[]) if isinstance(s,dict)}
    issues=[]
    for metric in submission.get('metrics',[]):
        if not isinstance(metric,dict):continue
        value=metric.get('observed');spec=specs.get((metric.get('name'),metric.get('unit')),{})
        if not isinstance(value,(int,float)) or isinstance(value,bool):continue
        for key,violates in [('minimum',lambda x:value<x),('maximum',lambda x:value>x)]:
            bound=spec.get(key)
            if isinstance(bound,(int,float)) and not isinstance(bound,bool) and violates(bound):
                issues.append(f'{metric.get("name")}: observed violates public {key}={bound}')
    return issues
