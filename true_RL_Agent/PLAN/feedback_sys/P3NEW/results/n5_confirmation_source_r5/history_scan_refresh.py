from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
base=Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
job_patterns=['*/public_dev_job.json','*/*/public_dev_job.json','*/*/*/public_dev_job.json']
card_patterns=['*/capture_card*.json','*/*/capture_card*.json']
def paths(patterns):
 found=set()
 for pattern in patterns:found.update(base.glob(pattern))
 return sorted(found)
def hits(value,pointer=''):
 found=[]
 if isinstance(value,dict):
  for k,v in value.items():found.extend(hits(v,pointer+'/'+str(k).replace('~','~0').replace('/','~1')))
 elif isinstance(value,list):
  for i,v in enumerate(value):found.extend(hits(v,pointer+'/'+str(i)))
 elif value==1900000000 or (isinstance(value,str) and (value=='1900000000' or 'n5_r5_20261001' in value)):
  found.append({'pointer':pointer,'value':value})
 return found
rows,cards,errors=[],[],[]
for kind,listed in [('job',paths(job_patterns)),('card',paths(card_patterns))]:
 for path in listed:
  try:
   raw=path.read_bytes();data=json.loads(raw);item={'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw),'explicit_candidate_matches':hits(data)}
   if kind=='job':
    plan=data.get('spec',data)['evaluation_plan'];item.update(task_id=plan['task_id'],binding=plan['evaluation_binding']);rows.append(item)
   else:
    item.update(status=data.get('status'),tasks={task:{k:v for k,v in value.items() if k in ('diagnostic_cohort','eval_base_seed','runner_seeds','seed_namespace','seed_cohort_hash')} for task,value in data.get('tasks',{}).items() if isinstance(value,dict)});cards.append(item)
  except (OSError,ValueError,KeyError,TypeError) as error:errors.append({'kind':kind,'path':str(path),'error':str(error)})
print(json.dumps({'observed_utc':datetime.now(timezone.utc).isoformat(),'scope':'B public_dev_job metadata at 1/2/3 directory levels plus capture_card*.json at 1/2 levels; no images, NPZ, answers, giant archives or filesystem-wide search','patterns':{'jobs':job_patterns,'cards':card_patterns},'candidate':{'stair_cohort':'n5_r5_20261001','push_eval_base_seed':1900000000},'scanned_files':len(paths(job_patterns)),'scanned_cards':len(paths(card_patterns)),'files':rows,'cards':cards,'errors':errors,'new_roots_exist':{name:(base/name).exists() for name in ['n5_confirmation_source_r5_final','n5_confirmation_20261001_r5','n5_execution_20261001_r5']},'limits':['Explicit candidate matching is not a newly derived seed-hash comparison. Actual future CPU preflight must compare its seed_cohort_hash and runner seeds to these original bindings and R1-R4.','Bounded record scan does not prove global never-use and does not release source/capture/API.']},indent=2))
