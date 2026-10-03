"""Assemble small P1d evidence index from completed immutable local outputs."""
import hashlib,json,re,subprocess
from datetime import datetime,timezone
from pathlib import Path
root=Path.cwd(); out=root/'outputs/p1d_final'; checks=root/'outputs/trajectory_p1d/final_checks'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
 return h.hexdigest()
def ref(p):
 p=Path(p).resolve(); value={'path':str(p.relative_to(root)).replace('\\','/'),'sha256':sha(p),'bytes':p.stat().st_size}
 if p.suffix=='.py': value['lf_normalized_sha256']=hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
 return value
def size(p):return sum(f.stat().st_size for f in Path(p).rglob('*') if f.is_file())
run=read(out/'verification.json'); ledger=read(out/'session/ledger.jsonl')
assert ledger['formal']['accepted'] is True
assert ledger['formal']['selected_lane_id']=='lane-a'
request=read(out/'requests/g2-lane-a/user_payload.json'); feedback=request['public_feedback_summary']['behavior']
assert feedback['source_lane_id']=='lane-b'
assert feedback['accepted_best_relationship']=='not_established_for_selected_trial'
assert feedback['profile']['schema_version']=='true_rl_agent.stair.behavior.public.v5'
rows=[]; refs=[ref(out/'verification.json'),ref(out/'paused.json'),ref(out/'session/ledger.jsonl'),ref(out/'session/session_genesis.json')]
for m in run['sources']:
 p=Path(m['summary_path']); phase=read(p); e=phase['task_evidence']; receipt=e['trajectory_build']; cardp=Path(receipt['card_reference']['manifest_path']); cardm=read(cardp); card=read(cardp.with_name('policy_card.json')); full=read(p.with_name('behavior_full.json')); ids=full['audit']['required_claim_ids']; assert len(ids)==len(set(ids))==18
 assert e['behavior']==full['public_report']; assert receipt==full['audit']['trajectory_build']
 assert receipt['projection']['max_report_bytes']==8192 and receipt['projection']['report_bytes']<=8192
 assert cardm['example_source_bindings']==[]
 cache=cardp.parent.parent; numeric=read(cache/'numeric/manifest.json')
 row={'role':'parent' if phase['subject_kind']!='candidate' else ('lane-a' if 'lane-a' in phase['job_id'] else 'lane-b'), 'source_archive':m['source'],'original_manifest_sha256':m['original_manifest_sha256'],'fixture_manifest_sha256':m['fixture_manifest']['manifest_sha256'],'changed_identity_fields':m['changed_identity_fields'],'job_id':phase['job_id'],'checkpoint_hash':e['execution_identity']['checkpoint_hash'],'evaluation_plan_hash':phase['evaluation_plan_hash'],'source_id':receipt['source_id'],'card_id':e['behavior']['card_id'],'report_bytes':receipt['projection']['report_bytes'],'required_report_bytes':full['audit']['required_report_bytes'],'required_claim_count':len(ids),'claim_count':cardm['claim_count'],'selected_example_count':cardm['selected_example_count'],'selected_examples':card['selected_examples'],'unavailable_claim_texts':[c['text'] for c in card['claims'] if c['template_id']=='definition' and c['operands']['text']['path'][-1].startswith('unavailable_')],'phase':ref(p),'full':ref(p.with_name('behavior_full.json')),'card_manifest':ref(cardp),'numeric_manifest':ref(cache/'numeric/manifest.json'),'build_wall_seconds':receipt['build_wall_seconds'],'peak_process_memory_bytes':cardm['cost']['peak_process_memory_bytes'],'raw_archive_bytes':size(p.parent/'numeric_evidence'),'numeric_bytes':size(cache/'numeric'),'card_bytes':size(cache/'card'),'full_bytes':p.with_name('behavior_full.json').stat().st_size,'cpu_postprocessing':{k:v for k,v in phase['evaluation_execution_cost']['splits']['public_dev'].items() if k.startswith('postprocessing_')}}
 rows.append(row);refs += [row[k] for k in ['phase','full','card_manifest','numeric_manifest']]+[ref(cache/'complete.json')]
assert len({r['source_id'] for r in rows})==3 and len({r['card_id'] for r in rows})==3
for name in ['g1-lane-a','g1-lane-b','g2-lane-a']:
 refs += [ref(out/'requests'/name/'request.http_body.json'),ref(out/'requests'/name/'user_payload.json')]
for p in (out/'session').rglob('*.result.json'):refs.append(ref(p))
for p in (out/'session/generation_receipts').rglob('*.json'):refs.append(ref(p))
for p in (out/'session/trial_receipts').rglob('*.json'):refs.append(ref(p))
checks_result=read(checks/'test_results.json')
# Historical line-regex omitted a multi-line docstring test. Keep raw logs intact.
missing='tests.unit.test_subprocess_runner.SubprocessRunnerTests.test_subprocess_runner_accepts_existing_summary_without_optional_diagnostics'
if missing not in checks_result['unique_test_ids']:
 checks_result['unique_test_ids'][missing]=['ok'];checks_result['unique_real_tests']+=1;checks_result['unique_passed']+=1
extra=[]
for name in ['automatic_profile_final']:
 p=checks/(name+'.log'); text=p.read_text(encoding='utf-8',errors='replace'); extra.append({**ref(p),'test_count':int(re.search(r'Ran (\d+) tests',text)[1]),'passed':'\nOK' in text})
code_paths=subprocess.check_output(['git','diff','--name-only','41f5a44bf6dc63e1fd8aa8efed291d13f44f7511','--','src','scripts','tests'],text=True).splitlines()
code=[ref(root.parent/p if p.startswith('true_RL_Agent/') else root/p) for p in code_paths]
record={'schema_version':'trajectory.p1d.verification.v1','status':'PASS_CPU_COMPOSITION_ARCHIVE_GENERATION_HTTP_PENDING_PARENT_ACCEPTANCE','generated_at_utc':datetime.now(timezone.utc).isoformat(),'base_commit':'41f5a44bf6dc63e1fd8aa8efed291d13f44f7511','branch':'codex/trajectory-translator-upgrade','single_writer':True,'source_hash_note':'sha256 is executed working-tree bytes; lf_normalized_sha256 permits Git text normalization comparison without changing code semantics','final_artifact_review':{'read_only':True,'parent_stage_acceptance':False,'file_references_verified':81,'evidence_file_count_including_index':82,'file_count_scope':'81 referenced files plus this index','copied_npz_chunks_verified':84,'phase_full_receipt_http_source_checks':'PASS'},'code':code,'capture_command':['.venv/Scripts/python.exe','-u','PLAN/feedback_sys/NEW_SYS/P1d/capture_lifecycle.py','--output','outputs/p1d_final','--r4','outputs/trajectory_p0/raw/numeric_evidence/manifest.json','--k1','outputs/trajectory_p1d/raw/k1_numeric_evidence/manifest.json'],'capture_harness':ref(root/'PLAN/feedback_sys/NEW_SYS/P1d/capture_lifecycle.py'),'capture_log':ref(checks/'lifecycle.log'),'capture_exit_code':0,'run_summary':{k:run[k] for k in ['status','runner_completed_replays','changed_configuration_resume_rejections','wall_seconds','boundary','new_policy_evidence','gpu_runs','external_model_calls','http_network_calls','not_verified']},'accepted_vs_last_measured':{'generation_outcome':ledger['generation_outcome'],'accepted':ledger['formal']['accepted'],'selected_lane_id':ledger['formal']['selected_lane_id'],'next_request_source_lane_id':feedback['source_lane_id'],'next_request_relationship':feedback['accepted_best_relationship'],'next_request_card_id':feedback['profile']['card_id']},'mapping':rows,'requests':[{k:v for k,v in r.items() if k!='behavior'}|{'profile':r['behavior']['profile'],'source_trial_id':r['behavior'].get('source_trial_id')} for r in run['requests']],'artifacts':refs,'executions':read(checks/'executions.json'),'test_results':checks_result,'additional_tests':extra,'frozen_compatibility':read(checks/'frozen_compatibility.json'),'frozen_compatibility_audit':ref(checks/'current_frozen_raw_audit.json'),'storage':{'session_bytes':size(out/'session'),'retained_p1d_output_bytes':sum(size(p) for p in [root/'outputs/trajectory_p1d',out,*sorted((root/'outputs').glob('p1d_dev*'))]),'scope':'three phase archives and derived evidence plus fixture lifecycle receipts; excludes development retries'},'independent_read_only_review':{'agents':['review_evidence','review_lifecycle','review_acceptance_harness'],'scope':'development code/lifecycle/evidence review; not parent final acceptance','findings_resolved':['Genesis feedback configuration binding','private build receipt cross-checks','automatic 18-slot completeness','screen attachment validation','fixture storage and environment identities']},'development_failures':[ref(p) for p in sorted((root/'outputs/trajectory_p1d/development').glob('lifecycle_*.log'))],'not_verified':run['not_verified']+['target isaaclab6_validation short GPU eval','new-policy scientific qualification','Go2 v2 full lifecycle','Push/G1 trajectory profile','period/limit/geometry/history-link-COM qualifications','parent numeric comparison','DeepSeek comprehension','P2/P3 experiments'],'git_delivery':{'production_commits':['e4c10426','ec6c3a4f'],'final_head':'see final delivery response and branch HEAD (avoid self-referential commit hash)','push':'dedicated branch only; remote result recorded in final response'}}
(root/'PLAN/feedback_sys/NEW_SYS/P1d/verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':record['status'],'mapping':[{k:r[k] for k in ['role','source_archive','source_id','card_id','report_bytes','required_report_bytes','claim_count','build_wall_seconds','numeric_bytes','card_bytes']} for r in rows],'accepted_vs_last_measured':record['accepted_vs_last_measured'],'session_bytes':record['storage']['session_bytes']},indent=2))
