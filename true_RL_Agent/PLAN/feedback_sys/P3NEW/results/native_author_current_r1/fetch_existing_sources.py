from pathlib import Path
import sys,tarfile,json,hashlib,io
base=Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
files={}
for task,name in (('stair','stair_dual_view_dev_r3'),('push','push_body_grounding_dev_r2')):
 root=base/name
 for n in ('public_dev_result.json','public_dev_job.json'):
  files[task+'/'+n]=root/n
 phase=json.loads((root/'public_dev_result.json').read_bytes());bound=phase['task_evidence']['behavior_full']
 candidates=[root/bound['file'],root/'evaluation/public_dev'/bound['file'],root/'evaluation'/bound['file']]
 matches=[p for p in candidates if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==bound['sha256']]
 assert matches,(task,bound,[str(p) for p in candidates])
 files[task+'/behavior_full.json']=matches[0]
 if task=='stair':
  p=root/'numeric_evidence/manifest.json';m=json.loads(p.read_bytes());files['stair/numeric_evidence/manifest.json']=p
  for c in m['chunks']:
   assert Path(c['file']).name==c['file'];files['stair/numeric_evidence/'+c['file']]=p.parent/c['file']
shared=base/'text_author_smoke_r1/PLAN/feedback_sys/P3NEW/results/text_author_smoke_r1'
for p in (shared/'authoring').rglob('*.json'):
 files['shared_journal_readonly/'+p.relative_to(shared).as_posix()]=p
files['shared_journal_readonly/execution_stage.released.json']=shared/'execution_stage.released.json'
usage=base/'text_author_smoke_r1/PLAN/feedback_sys/P3NEW/results/native_author_smoke_r3/usage.json'
files['shared_journal_readonly/latest_native_usage.json']=usage
index={k:{'remote_path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size_bytes':p.stat().st_size} for k,p in files.items()}
raw=(json.dumps({'scope':'Read-only copies of existing artifacts; original shared journal remains authoritative; no simulation or model invocation','files':index},indent=2)+'\n').encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
 info=tarfile.TarInfo('remote_source_index.json');info.size=len(raw);t.addfile(info,io.BytesIO(raw))
 for rel,p in files.items():t.add(p,arcname=rel,recursive=False)
