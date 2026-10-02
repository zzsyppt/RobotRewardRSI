"""Read-only independent raw-input check of frozen R2 selection arithmetic.

The ranking contract is restated here; production reducers are never imported.
This compares stored objects, not natural-language necessity or semantics.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

B=Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
C=B/'n5_confirmation_20261001_r5'
S=B/'n5_confirmation_source_r5_final'
refs={};checks=[]
TERMS=('tilt','low_progress','unloaded_low_motion','up_z_negative')
CONDITIONS=('planar_speed','angular_speed','roll','pitch','height')


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def watch(p):
    p=Path(p).resolve()
    refs[str(p)]={'path':str(p),'sha256':digest(p),'bytes':p.stat().st_size}
    return p


def read(p):return json.loads(watch(p).read_text())


def module(p,name):
    spec=importlib.util.spec_from_file_location(name,watch(p))
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
    return obj


def same(name,actual,expected):
    normalize=lambda v:json.loads(json.dumps(v))
    a,e=normalize(actual),normalize(expected)
    item={'check':name,'status':'MATCH' if a==e else 'MISMATCH'}
    if a!=e:item.update(actual=a,expected=e)
    checks.append(item)


def stair_pick(group,patterns):
    distribution=Counter(patterns[r['id']] for r in group)
    most=max(distribution.values())
    modal=min(k for k,v in distribution.items() if v==most)
    candidates=[r for r in group if patterns[r['id']]==modal]
    duration=sorted(r['steps'] for r in candidates)[(len(candidates)-1)//2]
    selected=min((r for r in candidates if r['steps']==duration),key=lambda r:(r['run'],r['env']))
    return {'episode_id':selected['id'],'duration':duration,'terms':dict(zip(TERMS,modal)),
            'pattern_n':most,'group_n':len(group),'tie':sum(v==most for v in distribution.values())>1,
            'majority':most*2>len(group),'modal_members':sorted(r['id'] for r in candidates)}


def main():
    raw=module(S/'PLAN/feedback_sys/P3R/confirmation_raw_material.py','independent_raw_reader')
    residual=module(S/'PLAN/feedback_sys/P3NEW/recompute_residuals_from_raw.py','independent_signature')
    # These frozen files are the declared ranking contract, never execution truth.
    for task in ('stair_climbing','push_recovery'):
        watch(S/f'src/true_rl_agent/tasks/{task}/process_report.py')
    watch(S/'src/true_rl_agent/tasks/push_recovery/behavior.py')
    private=read(C/'numeric_raw/_private/sources.json')
    packing=read(C/'numeric_materials/packing_receipt.json')
    source_paths={'stair':[],'push':[]};manifests={}
    for entry in private['sources']:
        task='stair' if entry['source_id']=='stair_source' else 'push'
        p=Path(entry['manifest_path'])
        if p in source_paths[task]:continue
        source_paths[task].append(p);m=read(p);manifests[p]=m
        same(str(p)+'/bound SHA',digest(p),entry['manifest_sha256'])
        for chunk in m['chunks']:
            p2=watch(p.parent/chunk['file'])
            same(str(p2)+'/chunk SHA',digest(p2),chunk['sha256'])
    aliases={(r['task'],r['original_id']):r['episode_id'] for r in private['anonymous_mapping']}
    fulls={task:read(C/task/'evaluation/public_dev/behavior_full.json') for task in source_paths}
    rows,curves,_=raw.stair_rows(source_paths['stair'][0],C/'stair/evaluation/public_dev/behavior_full.json')
    rows_by_task={'stair':rows}
    patterns={r['id']:tuple(t in residual.signature_stair(r,curves[r['id']],fulls['stair']['semantics']['thresholds'])[1]
                          for t in TERMS) for r in rows}
    failures=defaultdict(list);successes=defaultdict(list)
    for r in rows:
        (successes[(r['cell'],r['run'])] if r['success'] else failures[(r['outcome'],r['cell'],r['run'])]).append(r)
    reasons=Counter(r['outcome'] for r in rows if not r['success'])
    ordered=sorted(failures,key=lambda k:(-reasons[k[0]],k[0],not bool(successes[k[1:]]),-len(failures[k]),k[1],k[2]))
    primary=ordered[0]
    stair={name:stair_pick(group,patterns) for name,group in [('failure',failures[primary]),('success',successes[primary[1:]])]}
    stair['primary_domain']=primary;stair['reason_counts']=dict(reasons)
    for role,sid in [('failure','process-example'),('success','compatible-success')]:
        stored=next(r for r in fulls['stair']['semantic']['sentences'] if r['id']==sid)
        selected=stair[role]
        same('stair/'+sid+'/scope',stored['scope'],{'label':selected['episode_id'],'numerator':selected['pattern_n'],'denominator':selected['group_n']})
        same('stair/'+sid+'/window',stored['window'],{'start':1,'end':selected['duration'],'terminal':selected['duration']})
    # All protected group contrasts: numeric objects, not text or rounded wording.
    protected=[]
    for key in ordered:
        if not any(failures[k][0]['scope']==failures[key][0]['scope'] for k in protected):protected.append(key)
    for i,key in enumerate(protected):
        failed,control=failures[key],successes[key[1:]]
        counts=[sum(patterns[r['id']][j] for r in failed) for j in range(len(TERMS))]
        j=max(range(len(TERMS)),key=lambda j:(counts[j],-j));n=counts[j]
        m=sum(patterns[r['id']][j] for r in control)
        stored=next(r for r in fulls['stair']['semantic']['sentences'] if r['id']==f'contrast-{i}')
        same(f'stair/contrast-{i}/predicate',stored['predicate'],TERMS[j])
        same(f'stair/contrast-{i}/scope',stored['scope'],{'label':f'{key[0]}/{key[1]}/run{key[2]}','numerator':n,'denominator':len(failed)})
        same(f'stair/contrast-{i}/comparison counts',[stored['comparison']['left']['numerator'],stored['comparison']['left']['denominator'],stored['comparison']['right']['numerator'],stored['comparison']['right']['denominator']],[n,len(failed),m,len(control)])
    del curves
    rows,curves,_=raw.push_rows(source_paths['push']);rows_by_task['push']=rows
    condition_by_id={};index_by_id={}
    for path in source_paths['push']:
        manifest=manifests[path];run=int(manifest['identity']['run_index']);meta=manifest['metadata']
        same(f'push/run{run}/mechanism',meta['parameters']['mechanism'],'world_planar_velocity_increment')
        for env,terminal in enumerate(meta['runtime_episode_records']):
            ident=f'r{run}.e{env}';scope=terminal['stratum'];events=terminal['push_events']
            magnitude=0.0 if scope=='control' else meta['parameters']['velocity_magnitudes_mps']['medium' if scope=='repeated' else scope]
            condition_by_id[ident]={'stratum':scope,'nominal_velocity_or_impulse_per_mass_mps':magnitude,
                'planned_events':terminal['planned_push_count'],'executed_events':len(events),
                'verified_events':sum(e['injection_verified'] for e in events),'completed_events':len(events)}
            index_by_id[ident]=terminal['episode_index']
    groups=defaultdict(list)
    for r in rows:groups[json.dumps(condition_by_id[r['id']],sort_keys=True)].append(r)
    profiles=[]
    for key,group in sorted(groups.items()):
        for outcome,count in sorted(Counter(r['outcome'] or 'censored' for r in group if r['outcome'] not in ('success_recovered','no_push_control')).items(),key=lambda kv:(-kv[1],kv[0])):
            profiles.append({'conditions':json.loads(key),'failure_label':outcome,'failure_episodes':count,'condition_episodes':len(group)})
    profiles.sort(key=lambda p:(-p['failure_episodes'],-p['conditions']['executed_events'],p['conditions']['stratum'],p['failure_label']))
    actual=[{k:p[k] for k in ('conditions','failure_label','failure_episodes','condition_episodes')} for p in fulls['push']['profile']['matched_failure_profiles']]
    same('push/all compatible profile memberships and rank',actual,profiles)
    chosen=profiles[0]
    matched=[r for r in rows if condition_by_id[r['id']]==chosen['conditions']]
    failed=[r for r in matched if (r['outcome'] or 'censored')==chosen['failure_label']]
    def pattern(r):
        return tuple(t for t in CONDITIONS if any(a<r['steps'] for e in r['events'] if e['status']=='executed' for a,b in e['violation_intervals'][t]))
    distribution=Counter(pattern(r) for r in failed)
    modal=min(distribution,key=lambda k:(-distribution[k],k));candidates=[r for r in failed if pattern(r)==modal]
    duration=statistics.median(r['steps'] for r in candidates)
    tie=lambda r:(r['run']-1,r['env'],index_by_id[r['id']])
    example=min(candidates,key=lambda r:(abs(r['steps']-duration),*tie(r)))
    control=min((r for r in matched if r['outcome']=='success_recovered'),key=lambda r:(r['run']!=example['run'],abs(r['steps']-duration),*tie(r)))
    push={'status':'available','pattern':list(modal),'pattern_n':distribution[modal],'group_n':len(failed),
          'tied':sum(v==distribution[modal] for v in distribution.values())>1,'majority':distribution[modal]*2>len(failed),
          'failure_episode_id':example['id'],'context_episode_id':example['id'],'success_episode_id':control['id'],
          'success_matching':'same_event_conditions_same_run' if control['run']==example['run'] else 'same_event_conditions_other_run',
          'selected_episode_ids':[example['id'],control['id']]}
    same('push/representative selection',{k:fulls['push']['representative_selection'][k] for k in push},push)
    push.update(primary_profile=chosen,median_duration=duration,modal_members=sorted(r['id'] for r in candidates),compatible_success_n=sum(r['outcome']=='success_recovered' for r in matched))
    del curves
    focus=[]
    for packet in (p for p in packing['packets'] if p['axis']=='focus'):
        task,scope,pid=packet['task'],packet['scope'],packet['packet_id']
        group=[r for r in rows_by_task[task] if scope=='all' or (r['run']==int(scope[12:]) if scope.startswith('logical_run_') else r['scope']==scope)]
        raw_focus=read(C/'numeric_raw/focus'/f'{task}_{scope}.json')
        related=read(C/'numeric_materials/materials/related'/f'{pid}.json')
        outcomes=dict(Counter(r['outcome'] for r in group));n_success=sum(r['success'] for r in group)
        same(pid+'/scope outcome counts',raw_focus['scope_outcomes'],outcomes)
        same(pid+'/scope success count',raw_focus['scope_successes'],n_success)
        keys=('episode_id','run_group','cell','steps','dt_s','original_success','original_outcome')
        expected=[{'episode_id':aliases[task,r['id']],'run_group':f'run_{r["run"]:02d}','cell':r['cell'],'steps':r['steps'],'dt_s':r['dt'],'original_success':r['success'],'original_outcome':r['outcome']} for r in group]
        order=lambda xs:sorted(xs,key=lambda r:r['episode_id'])
        same(pid+'/raw per-row identity clock outcome',order([{k:r[k] for k in keys} for r in raw_focus['all_scope_rows']]),order(expected))
        relrows=[dict(zip(related['scope_row_columns'],r)) for r in related['scope_rows']]
        same(pid+'/related per-row identity clock outcome',order([{k:r[k] for k in keys} for r in relrows]),order(expected))
        focus.append({'packet_id':pid,'task':task,'scope':scope,'scope_n':len(group),'success_n':n_success,'terminal_counts':outcomes})
    unchanged=all(digest(p)==v['sha256'] for p,v in refs.items());same('inputs unchanged',unchanged,True)
    imports=[n for n in sys.modules if n=='true_rl_agent' or n.startswith('true_rl_agent.')];same('production imports',imports,[])
    failed=[c for c in checks if c['status']=='MISMATCH']
    print(json.dumps({'schema_version':'p3new.independent_representative_supplement.v1','at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'PASS_DECLARED_SUBSET' if not failed else 'MISMATCH_DECLARED_SUBSET','checks':checks,'failed_checks':failed,
        'input_refs':list(refs.values()),'stair':stair,'push':push,'focus_units':focus,'inputs_unchanged':unchanged,
        'model_calls':0,'simulation_calls':0,'source_writes':0,'production_imports':imports,
        'limits':['Recorded outcomes are not independent physical terminal reclassification.',
                  'This checks frozen ranking arithmetic and numeric sentence metadata, not semantic representativeness or necessary-fact completeness.',
                  'Focus packed interval/extrema operands beyond identity/time/outcome are not all independently recounted in this supplement.',
                  'No natural-language rounding, visual support, ABI/actionability, or final author delivery verification.']},ensure_ascii=False,indent=2))
    return 1 if failed else 0


if __name__=='__main__':sys.exit(main())
