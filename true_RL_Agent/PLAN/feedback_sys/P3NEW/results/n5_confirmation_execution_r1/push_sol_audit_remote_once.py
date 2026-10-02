from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,os,subprocess
B=Path('/data/yuyangliu/AGENT/feedback_p3new_20260929');control=B/'n5_execution_20260930';stage=control/'sol_audit_stage_r1';source=B/'n5_selection_repair_source_r1';packet=B/'n5_confirmation_20260930/engineering_repair/observers/push'
manifest=json.loads((stage/'push_sol_audit_execution_manifest.json').read_bytes())
allowed={control/'audit_push_sol_engineering.py',control/'push_sol_audit_inputs.ready.json',packet/'e2/sol_response/raw_answer.txt',packet/'e2/sol_response/access_receipt.json'}
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2);f.write('\n')
rows=manifest['staged_files'];assert {Path(r['remote']) for r in rows}==allowed
copied=[];identical=[]
for row in rows:
 target=Path(row['remote']);original=stage/target.name
 assert digest(original)==row['sha256'] and original.stat().st_size==row['size_bytes']
 if target.exists():assert target.is_file() and digest(target)==row['sha256'];identical.append(str(target))
for row in rows:
 target=Path(row['remote'])
 if not target.exists():
  target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as f:f.write((stage/target.name).read_bytes())
  copied.append(str(target))
 assert digest(target)==row['sha256']
stdout=control/'push_sol_audit_r1.stdout.txt';stderr=control/'push_sol_audit_r1.stderr.txt';invocation=control/'push_sol_audit_r1.invocation.json';exit_path=control/'push_sol_audit_r1.exit.json'
assert not any(p.exists() for p in (stdout,stderr,invocation,exit_path))
argv=['/data/yuyangliu/envs/isaaclab6_validation/bin/python','-B',str(control/'audit_push_sol_engineering.py'),'--inputs',str(control/'push_sol_audit_inputs.ready.json')]
changes={'PYTHONPATH':str(source/'src')+':'+str(source/'PLAN/feedback_sys/P3NEW'),'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':''}
record={'status':'DEVELOPMENT_REPAIR_NOT_ORIGINAL_N5_CONFIRMATION','original_n5_status':'INCONCLUSIVE','started_at_utc':datetime.now(timezone.utc).isoformat(),'argv':argv,'cwd':str(control),'environment_set_process_local':changes,'script_sha256':digest(control/'audit_push_sol_engineering.py'),'ready_inputs_sha256':digest(control/'push_sol_audit_inputs.ready.json'),'files_copied_without_rewriting_bytes':copied,'existing_identical_files':identical,'stdout':str(stdout),'stderr':str(stderr),'automatic_retry':False,'model_calls':0,'GPU_jobs':0,'assemble_executed':False}
save(invocation,record)
exit_record=dict(record)
try:
 env=os.environ.copy();env.update(changes)
 with stdout.open('xb') as out,stderr.open('xb') as err:
  result=subprocess.run(argv,cwd=control,env=env,stdout=out,stderr=err,timeout=1800)
 exit_record['returncode']=result.returncode
except Exception as error:
 exit_record['returncode']=None;exit_record['exception_type']=type(error).__name__
finally:
 exit_record['finished_at_utc']=datetime.now(timezone.utc).isoformat();exit_record['stdout_sha256']=digest(stdout);exit_record['stderr_sha256']=digest(stderr);save(exit_path,exit_record)
print(json.dumps({'returncode':exit_record['returncode'],'exit_record':str(exit_path),'exit_record_sha256':digest(exit_path),'stdout_sha256':exit_record['stdout_sha256'],'stderr_sha256':exit_record['stderr_sha256'],'audit_stdout':stdout.read_text(encoding='utf-8'),'status':record['status']}))
