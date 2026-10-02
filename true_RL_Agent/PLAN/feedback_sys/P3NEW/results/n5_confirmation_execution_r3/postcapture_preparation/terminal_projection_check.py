"""Independent field-level check of the frozen R3 terminal delivery projection.

Read literal whitelist declarations from the hash-verified frozen exporter; never
import/call that exporter, a report reducer, or a model. No source field is silently
ignored: retained leaves are compared exactly and omitted source fields are listed.
"""
from __future__ import annotations
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

CONSTANTS = ('STAIR_FACT_TERMINAL_FIELDS','PUSH_FACT_TERMINAL_FIELDS','PUSH_FACT_EVENT_FIELDS')

def load_contract(source_root, source_manifest):
    root, manifest_path = Path(source_root), Path(source_manifest)
    relative = 'PLAN/feedback_sys/P3NEW/export_confirmation_raw.py'
    path = root/relative
    data, manifest = path.read_bytes(), json.loads(manifest_path.read_bytes())
    expected = manifest[relative]
    sha = hashlib.sha256(data).hexdigest()
    if sha != expected['sha256'] or len(data) != expected['size_bytes']:
        raise ValueError('Frozen terminal whitelist source differs')
    tree = ast.parse(data.decode('utf-8'))
    fields = {}
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0],ast.Name):
            name = node.targets[0].id
            if name in CONSTANTS:
                value = ast.literal_eval(node.value)
                if not isinstance(value,tuple) or not all(isinstance(v,str) for v in value) or len(set(value)) != len(value):
                    raise ValueError('Unexpected terminal whitelist declaration')
                fields[name] = list(value)
    if set(fields) != set(CONSTANTS): raise ValueError('Missing terminal whitelist declaration')
    return {'fields':fields,'policy_source':{'path':str(path.resolve()),'sha256':sha,'size_bytes':len(data)},
            'source_manifest':{'path':str(manifest_path.resolve()),'sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest()}}

def verify_episode_projection(actual, source, *, task, contract):
    checks, excluded = [], []

    def record(path, observed, expected):
        # JSON native types must match: bool cannot pass as integer 0/1.
        equal = type(observed) is type(expected) and observed == expected
        checks.append({'path':path,'equal':equal,'actual':deepcopy(observed),'expected':deepcopy(expected),
                       'basis':'same_retained_source_field_no_numeric_transform'})

    def compare(path, observed, expected):
        if isinstance(expected,dict):
            if not isinstance(observed,dict):
                record(path+'/type',type(observed).__name__,'dict')
                return
            record(path+'/keys',sorted(observed),sorted(expected))
            for key,value in expected.items():
                if key in observed: compare(path+'/'+key,observed[key],value)
        elif isinstance(expected,list):
            if not isinstance(observed,list):
                record(path+'/type',type(observed).__name__,'list')
                return
            record(path+'/length',len(observed),len(expected))
            for index,(a,e) in enumerate(zip(observed,expected)): compare(path+'/'+str(index),a,e)
        else:
            record(path,observed,expected)

    if task not in ('stair','push') or not isinstance(actual,dict) or not isinstance(source,dict):
        raise ValueError('Requires one known-task actual/source episode mapping')
    record('/episode/source_task',source.get('task',task),task)
    expected = {k:deepcopy(v) for k,v in source.items() if k != 'direct_measurements'}
    if 'original_terminal_record' in source:
        terminal = source['original_terminal_record']
        if not isinstance(terminal,dict): raise ValueError('Source terminal is not a mapping')
        allowed = set(contract['fields']['STAIR_FACT_TERMINAL_FIELDS' if task == 'stair' else 'PUSH_FACT_TERMINAL_FIELDS'])
        projected = {k:deepcopy(v) for k,v in terminal.items() if k in allowed}
        excluded += [{'source_pointer':'/episode/original_terminal_record/'+k,'source_value':deepcopy(v),
                      'reason':'absent_from_frozen_terminal_whitelist'} for k,v in terminal.items() if k not in allowed]
        if task == 'push' and 'push_events' in terminal:
            events = terminal['push_events']
            if not isinstance(events,list) or not all(isinstance(e,dict) for e in events):
                raise ValueError('Source events are not an ordered list of mappings')
            event_fields = set(contract['fields']['PUSH_FACT_EVENT_FIELDS'])
            projected['push_events'] = []
            for index,event in enumerate(events):
                projected['push_events'].append({k:deepcopy(v) for k,v in event.items() if k in event_fields})
                excluded += [{'source_pointer':f'/episode/original_terminal_record/push_events/{index}/'+k,
                              'source_value':deepcopy(v),'reason':'absent_from_frozen_event_whitelist'}
                             for k,v in event.items() if k not in event_fields]
        expected['original_terminal_record'] = projected
    compare('/episode',actual,expected)
    failed = sum(not row['equal'] for row in checks)
    return {'schema_version':'p3new.r3_terminal_projection_source_check.v1',
            'status':'FAIL' if failed else 'PASS_EXACT_RETAINED_FIELDS', 'task':task,
            'policy_source':contract['policy_source'],'source_manifest':contract['source_manifest'],
            'checks':checks,'failed_count':failed,'excluded_source_fields':excluded,
            'direct_measurements':'Not an episode field in related_v2; checked separately as direct_arithmetic/columns.',
            'scope':'Exact source metadata and retained native terminal/event fields, including missing-key state and event order; no terminal semantics or physical reclassification.',
            'production_projection_called':False,'source_fields_added_to_facts':False}
