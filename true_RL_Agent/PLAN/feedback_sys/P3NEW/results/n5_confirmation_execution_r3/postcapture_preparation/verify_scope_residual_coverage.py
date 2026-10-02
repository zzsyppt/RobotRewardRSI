"""Read-only reuse of frozen independent raw/residual helpers on R2.

Checks a declared subset: all-pool counts, 12 focus membership, residuals and
numeric coverage. Does not import production reducers or author/reviewer output.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

B=Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
C=B/'n5_confirmation_20260930_r3'
S=B/'n5_confirmation_source_r3_final'
refs={}
checks=[]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def watch(path):
    path=Path(path).resolve()
    refs[str(path)]={'path':str(path),'sha256':digest(path),'bytes':path.stat().st_size}
    return refs[str(path)]


def load(path):
    watch(path)
    return json.loads(Path(path).read_text())


def same(label,actual,expected):
    normalize=lambda x:json.loads(json.dumps(x))
    a,e=normalize(actual),normalize(expected)
    record={'check':label,'status':'MATCH' if a==e else 'MISMATCH'}
    if a!=e:record.update(actual=a,expected=e)
    checks.append(record)


def module(path,name):
    watch(path)
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    raw=module(S/'PLAN/feedback_sys/P3R/confirmation_raw_material.py','independent_original_raw_reader')
    residual=module(S/'PLAN/feedback_sys/P3NEW/recompute_residuals_from_raw.py','independent_residual_coverage')
    private=load(C/'numeric_raw/_private/sources.json')
    packing=load(C/'numeric_materials/packing_receipt.json')
    fulls={t:load(C/t/'evaluation/public_dev/behavior_full.json') for t in ('stair','push')}
    aliases={(r['task'],r['original_id']):r['episode_id'] for r in private['anonymous_mapping']}
    source_paths={t:[] for t in fulls}
    source_manifests={}
    for r in private['sources']:
        task='stair' if r['source_id']=='stair_source' else 'push'
        path=Path(r['manifest_path'])
        if path in source_paths[task]:continue
        source_paths[task].append(path)
        manifest=load(path);source_manifests[path]=manifest
        same(f'{task}/{path.name}/bound manifest digest',digest(path),r['manifest_sha256'])
        for chunk in manifest['chunks']:
            p=path.parent/chunk['file']
            same(f'{task}/{path.parent.name}/{chunk["file"]}/chunk digest',watch(p)['sha256'],chunk['sha256'])
    rows_by_task={};results={}
    for task in ('stair','push'):
        full=fulls[task]
        if task=='stair':
            rows,curves,_=raw.stair_rows(source_paths[task][0],C/task/'evaluation/public_dev/behavior_full.json')
            rules=full['semantics']['thresholds']
            records=residual.stair_residuals(rows,curves,rules)
        else:
            rows,curves,_=raw.push_rows(source_paths[task])
            records=residual.push_residuals(rows)
        rows_by_task[task]=rows
        coverage=residual.coverage(rows,records,task)
        # Compare independent per-failure identities, states, signature bins and denominator operands.
        independent={r['episode_id']:r for r in records}
        production=(full['semantic']['residuals']['episodes'] if task=='stair' else full['residual']['episodes'])
        same(f'{task}/residual episode set',sorted(independent),sorted(r['episode_id'] for r in production))
        bins={0:'le_0.5s',1:'le_2s',2:'gt_2s'}
        for r in production:
            expected=independent[r['episode_id']]
            label=f'{task}/residual/{r["episode_id"]}'
            if task=='stair':
                same(label+'/state',r['status'],expected['state'])
                same(label+'/domain',[r['reason'],r['cell'],r['run']],expected['domain'])
                # Names of bins come from the frozen task contract; only labels differ.
                actual_signature=sorted(r['signature'])
                expected_signature=sorted([[k,bins[b]] for k,b in expected['signature']])
                same(label+'/signature',actual_signature,expected_signature)
                same(label+'/success_n',r['success_n'],expected['matched_success_n'])
                same(label+'/matching_success_n',r['matching_success_n'],expected['matching_signature_n'])
            else:
                states={'rule_positive':'positive','rule_negative':'negative','comparison_missing':'missing'}
                same(label+'/state',r['state'],states[expected['state']])
                a=[{k:w[k] for k in ('domain','state','signature','success_windows','matching_success_windows')} for w in r['windows']]
                e=[{'domain':w['domain'],'state':states[w['state']],'signature':w['signature'],
                    'success_windows':w['matched_success_n'],'matching_success_windows':w['matching_signature_n']} for w in expected['windows']]
                same(label+'/windows',a,e)
        sampled=full['semantic']['sampling'] if task=='stair' else full['coverage_sampling']
        same(f'{task}/sampling selected',sorted(sampled['selected_episode_ids']),sorted(r['episode_id'] for r in coverage['selected']))
        same(f'{task}/sampling supplement',sorted(sampled['supplemental_episode_ids'] if task=='stair' else sampled['residual_selected_episode_ids']),coverage['residual_supplement_ids'])
        same(f'{task}/sampling residual unselected',sorted(sampled['residual_unselected_episode_ids'] if task=='stair' else sampled['residual_unsampled_episode_ids']),coverage['unselected_residual_ids'])
        windows={}
        for w in sampled['windows']:
            windows.setdefault(w['episode_id'],[]).append([w['start'] if task=='stair' else w['start_step'],w['end'] if task=='stair' else w['end_step']])
        same(f'{task}/all numeric windows',{k:sorted(v) for k,v in windows.items()},
             {r['episode_id']:r['windows'] for r in coverage['selected']})
        same(f'{task}/time median',sampled['time_coverage_median'] if task=='stair' else sampled['sampled_time_fraction_median'],coverage['time_fraction_median'])
        same(f'{task}/time minimum',sampled['time_coverage_minimum'] if task=='stair' else sampled['sampled_time_fraction_minimum'],coverage['time_fraction_min'])
        counts=dict(Counter(r['outcome'] for r in rows))
        if task=='push':same('push/full terminal counts',full['profile']['all']['terminal_counts'],counts)
        else:
            summary=full['semantic']['comparison_summary']
            same('stair/summary N',summary['episode_n'],len(rows))
            for scope,stats in summary['strata'].items():
                group=[r for r in rows if r['scope']==scope]
                same(f'stair/{scope}/N-success',[stats['n'],stats['success']],[len(group),sum(bool(r['success']) for r in group)])
                for code in ('tilt','low_progress','unloaded_low_motion','up_z_negative'):
                    n=sum(code in residual.signature_stair(r,curves[r['id']],rules)[1] for r in group)
                    same(f'stair/{scope}/{code}',stats[code],n)
        results[task]={'pool_n':len(rows),'failure_n':len(records),'terminal_counts':counts,
                       'residual_state_counts':dict(Counter(r['state'] for r in records)),
                       'numeric_selected_n':len(coverage['selected']),'numeric_windows_n':sum(len(r['windows']) for r in coverage['selected']),
                       'residual_eligible_n':len(coverage['eligible_residual_ids']),
                       'residual_actual_sampled_n':len(coverage['union_residual_ids']),
                       'time_fraction_median':coverage['time_fraction_median'],'time_fraction_min':coverage['time_fraction_min']}
        del curves
    focus_results=[]
    for packet in (r for r in packing['packets'] if r['axis']=='focus'):
        task,scope,pid=packet['task'],packet['scope'],packet['packet_id']
        rows=rows_by_task[task]
        group=(rows if scope=='all' else [r for r in rows if r['run']==int(scope.removeprefix('logical_run_'))]
               if scope.startswith('logical_run_') else [r for r in rows if r['scope']==scope])
        path=C/'numeric_raw/focus'/f'{task}_{scope}.json';f=load(path)
        expected_members=sorted(aliases[task,r['id']] for r in group)
        same(pid+'/raw scope denominator',f['scope_denominator'],len(group))
        same(pid+'/raw scope members',sorted(r['episode_id'] for r in f['all_scope_rows']),expected_members)
        related=load(C/'numeric_materials/materials/related'/f'{pid}.json')
        idx=related['scope_row_columns'].index('episode_id')
        same(pid+'/related scope denominator',related['scope_denominator'],len(group))
        same(pid+'/related scope members',sorted(r[idx] for r in related['scope_rows']),expected_members)
        actual_pool=[]
        for r in related['complete_pool_groups']:
            mapping=dict(zip(related['complete_pool_group_columns'],r))
            for alias in mapping.pop('episode_ids'):
                actual_pool.append({'episode_id':alias,**mapping})
        expected_pool=[{'episode_id':aliases[task,r['id']],'run_group':f'run_{r["run"]:02d}','scope':r['scope'],
                        'cell':r['cell'],'original_success':r['success'],'original_outcome':r['outcome']} for r in rows]
        same(pid+'/related complete pool membership',sorted(actual_pool,key=lambda r:r['episode_id']),sorted(expected_pool,key=lambda r:r['episode_id']))
        focus_results.append({'packet_id':pid,'task':task,'scope':scope,'scope_n':len(group),
                              'terminal_counts':dict(Counter(r['outcome'] for r in group)),
                              'checked':'raw and related membership/denominators from original numeric pool'})
    unchanged=all(digest(p)==r['sha256'] for p,r in refs.items())
    same('inputs unchanged',unchanged,True)
    imported=[name for name in sys.modules if name=='true_rl_agent' or name.startswith('true_rl_agent.')]
    same('production module imports',imported,[])
    failures=[r for r in checks if r['status']=='MISMATCH']
    print(json.dumps({'schema_version':'p3new.focus_deterministic_supplement.v1','at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'PASS_DECLARED_SUBSET' if not failures else 'MISMATCH_DECLARED_SUBSET','checks':checks,'failed_checks':failures,
        'input_refs':list(refs.values()),'tasks':results,'focus_units':focus_results,'inputs_unchanged':unchanged,
        'model_calls':0,'simulation_calls':0,'source_writes':0,'production_imports':imported,
        'limits':['Subset, not complete focus formal-object coverage or semantic acceptance.',
                  'Still unchecked here: representative-selection and counterexample tie/median choices; exact report sentence arithmetic/rounding and actionability/ABI claims.',
                  'Original numeric terminal records are copied outcomes, not independent physical terminal reclassification.',
                  'No visual support review, necessary-fact judgments, author delivery, or original reviewer-result changes.']},ensure_ascii=False,indent=2))
    return 0 if not failures else 1


if __name__=='__main__':
    sys.exit(main())
