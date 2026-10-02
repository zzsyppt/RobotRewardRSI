"""R2 read-only CPU audit: original Push archives -> supplied focus/full objects.

The only imported task helpers are frozen independent readers/verifiers.
No production reducer/exporter/packer is imported or used as expected truth.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import numpy as np

B = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
C, S, E = B/'n5_confirmation_20260930_r3', B/'n5_confirmation_source_r3_final', B/'n5_execution_20260930_r3'
refs, checks, helper_rows = {}, [], []
FOUR = ['true_steps','run_count','longest_consecutive_steps','first_start_step']
OPERANDS = ['planned_events','actual_events','whole_condition_false_stats','terminal_pose_motion','actual_event_rows',
            'post_all_required_latches_unstable_stats','terminal_conditions','missing_fields','requested_steps_not_observed']
EVENTS = ['event_index','anchor','requested_delta_v_xy','observed_delta_v_xy','original_verified','original_latch','original_censored',
          'observed_window_first_last_count','condition_false_count_runs_maxstreak_first','all_window_stable_stats',
          'arithmetic_first_dwell','pre_latch_stable_stats','interruptions_step_prior_dwell_failed_condition_mask',
          'anchor_first_response_first_speed_peak_window_last_full_pose_motion','post_latch_to_next_injection_unstable_count_runs_maxstreak_first']
LIMITS = [
    'Recorded pose/motion and runtime event/terminal receipts are authenticated operands, not independently reconstructed physical truth.',
    'Quaternion, full root angular velocity, absolute/reset positions and action vectors are not used to reconstruct pose/action metrics.',
    'Requested velocity changes/planned event schedules and outcome labels are source-receipt consistency; outcome meaning/priority and causal claims remain unverified.',
    'All frozen arithmetic helper unchecked fields are retained below; original reviewer answers and scores are untouched.',
    'Focus numeric objects and full transcript arithmetic are audited; report prose rounding/meaning, ABI, visual support and necessary-fact recall are not audited.',
]

def canonical(x): return json.dumps(x, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',',':')).encode()
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def watch(p):
    p=Path(p).resolve(); r={'path':str(p),'sha256':digest(p),'bytes':p.stat().st_size}
    if str(p) in refs and refs[str(p)] != r: raise ValueError('input changed: '+str(p))
    refs[str(p)]=r
    return r
def load(p): watch(p); return json.loads(Path(p).read_text())
def equal(a,b):
    if isinstance(b,dict): return isinstance(a,dict) and set(a)==set(b) and all(equal(a[k],v) for k,v in b.items())
    if isinstance(b,list): return isinstance(a,list) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    if isinstance(b,float): return type(a) in (int,float) and np.isfinite(a) and abs(a-b)<=1e-10
    return type(a) is type(b) and a==b
def check(path,actual,expected,kind='independent_arithmetic'):
    a=json.loads(json.dumps(actual)); e=json.loads(json.dumps(expected)); ok=equal(a,e)
    r={'path':path,'status':'MATCH' if ok else 'MISMATCH','kind':kind,
       'actual_sha256':hashlib.sha256(canonical(a)).hexdigest(),'expected_sha256':hashlib.sha256(canonical(e)).hexdigest()}
    if not ok:r.update(actual=a,expected=e)
    checks.append(r)
def mod(relative,name,source):
    p=S/relative; r=watch(p); expected=source[relative]
    if r['sha256']!=expected['sha256'] or r['bytes']!=expected['size_bytes']:raise ValueError('frozen helper differs')
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def array_id(a): return {'dtype':a.dtype.str,'shape':list(a.shape),'c_order_sha256':hashlib.sha256(a.tobytes()).hexdigest()}
def stats(spans,dwell=None):
    lengths=[b-a+1 for a,b in spans]
    out=[sum(lengths),len(spans),max(lengths,default=0),min((a for a,b in spans),default=None)]
    if dwell is not None:out += [min((a+dwell-1 for a,b in spans if b-a+1>=dwell),default=None),sum(x>=dwell for x in lengths)]
    return out
def decode_sample(sample):
    if sample is None:return None
    if len(sample)!=3:raise ValueError('packed sample shape differs')
    return [sample[0],*[None if v is None else array_id(np.asarray(v.split(','),dtype=np.float32)) for v in sample[1:]]]
def encoded_for_helper(a):
    shape=list(a.shape[1:]);n=len(a)
    if a.dtype.kind=='f':
        return {'dtype':a.dtype.str,'component_shape':shape,
                'by_c_order_component':[','.join(repr(float(v)) for v in col) for col in a.reshape(n,-1).T]}
    rows=[]
    for i in range(n):
        value=a[i].tolist()
        if rows and value==rows[-1][2]:rows[-1][1]=i+1
        else:rows.append([i+1,i+1,value])
    return {'dtype':a.dtype.str,'component_shape':shape,'inclusive_runs':rows}

def main():
    source=load(E/'runtime_source_manifest.json');card=load(E/'capture_card.json')
    check('capture/source manifest SHA',digest(E/'runtime_source_manifest.json'),card['source']['manifest']['sha256'],'source_binding')
    check('capture/root',card['output_root'],str(C),'source_binding')
    for relative in ('PLAN/feedback_sys/P3NEW/N5_FACT_REVIEW_PROTOCOL_ZH.md','PLAN/feedback_sys/P3NEW/export_confirmation_raw.py','PLAN/feedback_sys/P3NEW/prepare_confirmation_materials.py'):
        check('frozen source/'+relative,watch(S/relative)['sha256'],source[relative]['sha256'],'source_binding')
    raw=mod('PLAN/feedback_sys/P3R/confirmation_raw_material.py','independent_raw',source)
    arithmetic=mod('PLAN/feedback_sys/P3NEW/verify_push_confirmation_arithmetic.py','independent_push_arithmetic',source)
    private=load(C/'numeric_raw/_private/sources.json');packing=load(C/'numeric_materials/packing_receipt.json')
    full_path=C/'push/evaluation/public_dev/behavior_full.json';full=load(full_path)
    aliases={x['original_id']:x['episode_id'] for x in private['anonymous_mapping'] if x['task']=='push'}
    paths=[]; manifests={}; source_index={}
    for entry in private['sources']:
        if not entry['source_id'].startswith('push'):continue
        p=Path(entry['manifest_path'])
        if p in paths:continue
        paths.append(p);m=load(p);run=int(m['identity']['run_index']);manifests[run]=m
        check('source/'+str(run)+'/manifest',digest(p),entry['manifest_sha256'],'source_binding')
        positions=[i for i,v in enumerate(full['sources']) if v['manifest_sha256']==digest(p)]
        if len(positions)!=1:raise ValueError('full source binding absent or ambiguous')
        source_index[run]=positions[0]
        for chunk in m['chunks']:check('source/'+str(run)+'/'+chunk['file'],watch(p.parent/chunk['file'])['sha256'],chunk['sha256'],'source_binding')
    rows,curves,_=raw.push_rows(paths);by_alias={aliases[r['id']]:r for r in rows}
    check('unique raw episodes',len(rows),192,'population')
    arrays={}
    required=['valid','control_step','pose','planar_motion','conditions','episode_index','failure','timeout','event_steps','event_verified','observed_dwell','observed_recovery_step']
    for p in paths:
        m=manifests[int(load(p)['identity']['run_index'])];run=int(m['identity']['run_index'])
        arrays[run],_=raw.chunks(p,m,required)
    transcripts={(x['source'],x['env_id']):i for i,x in enumerate(full['transcripts'])}
    check('unique full transcripts',len(transcripts),192,'population')
    expected={};unverified=0;event_total=0
    for row in rows:
        run,env=row['run'],row['env'];ident=aliases[row['id']];meta=manifests[run]['metadata'];terminal=meta['runtime_episode_records'][env]
        source_arrays=arrays[run];valid=source_arrays['valid'][:,env]
        a={k:v[valid,env].copy() for k,v in source_arrays.items()}
        cv=curves[row['id']];steps,pose,motion,good=(cv[k] for k in ('steps','pose','motion','good'));stable=good.all(axis=1)
        T=len(steps);dwell=meta['time']['required_dwell_steps'];window=meta['time']['recovery_window_steps'];events=terminal['push_events']
        check(ident+'/source condition bytes',array_id(a['conditions']),array_id(good),'original_bytes_and_thresholds')
        indices=[e['event_index'] for e in events];anchors=[e['control_step'] for e in events]
        check(ident+'/event index order',indices,sorted(set(indices)),'source_binding')
        check(ident+'/event anchor order',anchors,sorted(set(anchors)),'source_binding')
        event_total+=len(events);unverified+=sum(not e['injection_verified'] for e in events)
        def sample(step):
            if step is None or step not in steps:return None
            ix=int(np.flatnonzero(steps==step)[0]);return [int(step),array_id(pose[ix]),array_id(motion[ix])]
        def ranges(mask):
            return {name:{'component_shape':list(values.shape[1:]),'c_order_components':[raw.extrema(col,steps[mask]) for col in values[mask].reshape(int(mask.sum()),-1).T]}
                    for name,values in (('pose',pose),('planar_motion',motion))} if mask.any() else {}
        event_rows=[];raw_events=[];landmarks=[];wanted={T};latches=[]
        for event in events:
            idx,anchor=event['event_index'],event['control_step'];mask=(steps>anchor)&(steps<=anchor+window);ws=steps[mask]
            spans=raw.spans(mask&stable,steps);latch=next((a+dwell-1 for a,b in spans if b-a+1>=dwell),None) if event['injection_verified'] else None
            pre=raw.spans(mask&stable&((steps<=latch) if latch is not None else np.ones(T,bool)),steps)
            peak=int(ws[np.argmax(pose[mask,3])]) if len(ws) else None
            stop=min((x for x in anchors if x>anchor),default=T)
            after=(steps>latch)&(steps<=stop)&~stable if latch is not None else None
            interruptions=[[b+1,b-a+1,(~good[b]).tolist()] for a,b in pre if len(ws) and b<int(ws[-1]) and (latch is None or b<latch)]
            er=[idx,anchor,event['requested_delta_v_world_mps'],event['observed_delta_v_world_mps'],event['injection_verified'],
                event['recovery_control_step'],event['recovery_censored'],[int(ws[0]),int(ws[-1]),len(ws)] if len(ws) else [None,None,0],
                [stats(raw.spans(mask&~good[:,j],steps)) for j in range(5)],stats(spans,dwell),latch,stats(pre,dwell),interruptions,
                [sample(s) for s in (anchor,anchor+1,peak,int(ws[-1]) if len(ws) else None)],None if after is None else stats(raw.spans(after,steps))]
            event_rows.append(er);latches.append(latch)
            raw_events.append({'event_index':idx,'anchor_step':anchor,'window_observed_steps':ws.tolist(),'stable_intervals':spans,
                'condition_false_intervals':{name:raw.spans(mask&~good[:,j],steps) for j,name in enumerate(meta['condition_fields'])},
                'first_dwell_end_step_by_arithmetic':latch,'pre_latch_stable_intervals':pre,'window_extrema':ranges(mask)})
            end=int(ws[-1]) if len(ws) else None
            landmarks.append({'event_index':idx,'anchor_step':anchor,'first_response_step':anchor+1,'observed_window_end_step':end,'window_first_speed_maximum_step':peak})
            wanted.update((anchor,anchor+1));wanted.update(s for s in (peak,end) if s is not None)
        all_latched=bool(latches) and len(latches)==terminal['planned_push_count'] and all(v is not None for v in latches)
        later=(steps>max(latches))&~stable if all_latched else None
        missing=[k for k in ('conditions','pose','planar_motion') if k not in a]
        expected[ident]={'packed':[terminal['planned_push_count'],len(events),[stats(raw.spans(~good[:,j],steps)) for j in range(5)],sample(T),event_rows,
                                  None if later is None else stats(raw.spans(later,steps)),good[-1].tolist(),missing,sorted(wanted-set(steps.tolist()))],
                         'raw_events':raw_events,'landmarks':landmarks,'sample_steps':sorted(wanted&set(steps.tolist())),
                         'raw_samples':[{'control_step':s,'pose':pose[s-1].tolist(),'planar_motion':motion[s-1].tolist()} for s in sorted(wanted&set(steps.tolist()))],
                         'curve_extrema':ranges(np.ones(T,bool)),'conditions':array_id(good),'terminal':terminal,'row':row}
        full_i=transcripts[source_index[run],env];transcript=full['transcripts'][full_i]
        check(ident+'/full source/env',[transcript['source'],transcript['env_id']],[source_index[run],env],'source_binding')
        numeric={'profile':'related_v1','episode':{'task':'push','episode_id':ident,'steps':T,'dt_s':row['dt'],'original_terminal_record':terminal},
                 'definitions':meta,'clock':{'first_step':1,'last_step':T,'stride':1},
                 'source_fields':{k:{'dtype':v.dtype.str,'shape':list(v.shape),'c_order_bytes_sha256':hashlib.sha256(v.tobytes()).hexdigest()} for k,v in a.items()},
                 'raw_columns':{k:encoded_for_helper(v) for k,v in a.items() if v.dtype.kind=='f'},
                 'raw_discrete':{k:encoded_for_helper(v) for k,v in a.items() if v.dtype.kind!='f'}}
        h=arithmetic.verify_push_confirmation_arithmetic(numeric,{'episode_id':ident,'complete_episode_derivation':transcript})
        helper_rows.append({'episode_id':ident,'original_id':row['id'],'full_pointer':'/transcripts/'+str(full_i),
                            'source_manifest':str(paths[list(manifests).index(run)]),'env_id':env,
                            'status':h['status'],'checked_field_paths':h['checked_field_paths'],'differences':h['differences'],
                            'unchecked_fields':h['unchecked_fields'],'input_error':h.get('input_error')})
        check(ident+'/independent full helper difference count',len(h['differences']),0,'independent_full_arithmetic')
        check(ident+'/independent full helper status',h['status'],'PASS_WITH_EXPLICIT_LIMITS','independent_full_arithmetic')
    focus_rows=[]
    for packet in packing['packets']:
        if packet['axis']!='focus' or packet['task']!='push':continue
        pid,scope=packet['packet_id'],packet['scope'];f=load(C/'numeric_raw/focus'/f'push_{scope}.json');related=load(C/'numeric_materials/materials/related'/f'{pid}.json')
        check(pid+'/packed operand columns',related['push_operand_columns'],OPERANDS,'format')
        check(pid+'/packed event columns',related['push_event_columns'],EVENTS,'format')
        check(pid+'/packed interval columns',related['interval_arithmetic_columns'][:4],FOUR,'format')
        expected_ids=sorted(aliases[r['id']] for r in rows if scope=='all' or (r['run']==int(scope[12:]) if scope.startswith('logical_run_') else r['scope']==scope))
        ids=[r['episode_id'] for r in f['all_scope_rows']]
        check(pid+'/source membership',sorted(ids),expected_ids,'population')
        check(pid+'/packed population',len(related['operands_in_same_row_order']),len(expected_ids),'population')
        for i,(raw_row,supplied) in enumerate(zip(f['all_scope_rows'],related['operands_in_same_row_order'])):
            ident=raw_row['episode_id'];e=expected[ident];prefix=pid+'/'+ident
            packed_row=dict(zip(related['scope_row_columns'],related['scope_rows'][i]));check(prefix+'/row identity',packed_row['episode_id'],ident,'source_binding')
            check(prefix+'/original anonymized terminal copy',raw_row['original_terminal_record'],{k:v for k,v in e['terminal'].items() if k not in ('env_id','episode_index')},'source_receipt_copy')
            measurements=raw_row['direct_measurements'];operands=measurements['focus_raw_operands']
            check(prefix+'/raw events',measurements['events'],e['raw_events'])
            check(prefix+'/raw whole extrema',measurements['curve_extrema'],e['curve_extrema'])
            check(prefix+'/raw latch receipt comparisons',measurements['arithmetic_latch_equals_original_record'],[x['first_dwell_end_step_by_arithmetic']==o['recovery_control_step'] for x,o in zip(e['raw_events'],e['terminal']['push_events'])])
            check(prefix+'/raw landmarks',operands['event_landmarks'],e['landmarks'])
            check(prefix+'/raw samples',operands['landmark_samples'],e['raw_samples'])
            decoded=np.empty((e['row']['steps'],5),bool);through=0
            for first,last,value in operands['conditions_rle']:
                if first!=through+1 or last<first or last>len(decoded):raise ValueError('conditions RLE not exact partition')
                decoded[first-1:last]=value;through=last
            check(prefix+'/raw condition RLE final step',through,len(decoded))
            check(prefix+'/raw condition RLE bytes',array_id(decoded),e['conditions'],'original_bytes_and_thresholds')
            check(prefix+'/raw clocks',[operands[k] for k in ('first_step','last_step','row_count','terminal_step')],[1,e['row']['steps'],e['row']['steps'],e['row']['steps']])
            check(prefix+'/raw missing fields',operands['missing_fields'],e['packed'][7],'source_field_presence')
            check(prefix+'/raw requested but unobserved',operands['requested_steps_not_observed'],e['packed'][8],'frozen_landmark_request')
            actual=json.loads(json.dumps(supplied));actual[3]=decode_sample(actual[3])
            for event in actual[4]:event[13]=[decode_sample(sample) for sample in event[13]]
            check(prefix+'/packed operand length',len(actual),len(OPERANDS),'format')
            for j,name in enumerate(OPERANDS):
                if name=='actual_event_rows':
                    check(prefix+'/packed event count',len(actual[j]),len(e['packed'][j]))
                    for k,(observed,wanted) in enumerate(zip(actual[j],e['packed'][j])):
                        check(prefix+'/event/'+str(k)+'/field count',len(observed),len(EVENTS),'format')
                        for n,column in enumerate(EVENTS):check(prefix+'/event/'+str(k)+'/'+column,observed[n],wanted[n],
                            'source_receipt_copy' if n in (2,3,4,5,6) else 'f32_source_bytes' if n==13 else 'independent_arithmetic')
                else:check(prefix+'/'+name,actual[j],e['packed'][j],'f32_source_bytes' if j==3 else 'independent_arithmetic')
        focus_rows.append({'packet_id':pid,'scope':scope,'rows':len(ids),'unique_original_ids':ids})
    imports=[name for name in sys.modules if name=='true_rl_agent' or name.startswith('true_rl_agent.')]
    check('production module imports',imports,[],'isolation')
    unchanged=all(digest(path)==ref['sha256'] for path,ref in refs.items());check('inputs unchanged',unchanged,True,'source_binding')
    failures=[x for x in checks if x['status']=='MISMATCH'];helper_failures=[x for x in helper_rows if x['differences'] or x['input_error']]
    result={'schema_version':'p3new.push_focus_operand_audit.v1','at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'PASS_DECLARED_SCOPE' if not failures and not helper_failures else 'MISMATCH_DECLARED_SCOPE',
        'original_episodes':len(rows),'original_actual_events':event_total,'unverified_actual_events':unverified,
        'focus_packet_count':len(focus_rows),'focus_projected_rows':sum(x['rows'] for x in focus_rows),'focus_units':focus_rows,
        'checks':checks,'failed_checks':failures,'full_arithmetic':helper_rows,'input_refs':list(refs.values()),'inputs_unchanged':unchanged,
        'model_calls':0,'gpu_calls':0,'remote_source_writes':0,'production_imports':imports,'semantic_support_proven':False,'limits':LIMITS}
    print(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2));return 0 if not failures and not helper_failures else 1

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as error:
        print(json.dumps({'status':'AUDIT_ERROR','exception_type':type(error).__name__,'error':str(error),
                          'completed_checks':checks,'completed_full_arithmetic':helper_rows,'input_refs':list(refs.values()),
                          'limits':LIMITS,'model_calls':0,'gpu_calls':0,'remote_source_writes':0},ensure_ascii=False,indent=2))
        traceback.print_exc(file=sys.stderr);sys.exit(2)
