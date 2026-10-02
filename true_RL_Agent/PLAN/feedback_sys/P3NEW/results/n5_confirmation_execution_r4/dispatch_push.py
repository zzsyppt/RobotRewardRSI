"""Start the existing frozen Push driver once, after actual Stair completion."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
remote = r'''
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,subprocess
c=Path('/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_execution_20261001_r4')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(c/'capture_card.json')=='811d7f140a5b3015f66a5093ba516c60bdc4a05c9ac827461d19bb05b8be3041'
assert sha(c/'capture_driver.py')=='a08d69b1f20adfc174d535afacb12bae05a54fccb2aab7bb2afcf290d86c68f2'
card=json.loads((c/'capture_card.json').read_bytes());cap=Path(card['output_root'])
previous=json.loads((c/'stair.exit.json').read_bytes())
assert previous['returncode']==0
assert json.loads((cap/'stair/result.json').read_bytes())['status']=='PASS'
assert json.loads((cap/'stair/public_dev_result.json').read_bytes())['evaluation_binding_hash']==card['tasks']['stair']['current_evaluation_binding']['evaluation_binding_hash']
assert not (cap/'push').exists()
assert not any((c/n).exists() for n in ('push.started.json','push.exit.json','push.stdout.log','push.stderr.log','push.driver.dispatch.json','push.driver.stdout.log','push.driver.stderr.log'))
free=shutil.disk_usage(c).free
assert free>=card['resources']['minimum_disk_before_task_bytes']['push']
gpu=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
rows=[[int(v.strip()) for v in line.split(',')] for line in gpu.stdout.splitlines()]
assert any(total>=32000 and used<1024 for _,used,total in rows)
argv=['/data/yuyangliu/envs/isaaclab6_validation/bin/python','-B',str(c/'capture_driver.py'),'push']
with (c/'push.driver.stdout.log').open('xb') as out,(c/'push.driver.stderr.log').open('xb') as err:
 proc=subprocess.Popen(argv,cwd=c,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
receipt={'status':'DRIVER_DISPATCHED_NOT_CAPTURE_COMPLETION','dispatched_at_utc':datetime.now(timezone.utc).isoformat(),
         'driver_pid':proc.pid,'argv':argv,'previous_stair_exit':previous,'gpu_before':rows,
         'disk_free_before':free,'capture_card_sha256':sha(c/'capture_card.json'),
         'driver_sha256':sha(c/'capture_driver.py'),'automatic_retries':0}
with (c/'push.driver.dispatch.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
'''
compile(remote, '<dispatch_push_remote>', 'exec')
with (HERE/'push_dispatch.stdout.json').open('xb') as out, (HERE/'push_dispatch.stderr.txt').open('xb') as err:
    result=subprocess.run(['ssh','humanoid-server','/data/yuyangliu/envs/isaaclab6_validation/bin/python','-B','-'],
                          input=remote.encode(),stdout=out,stderr=err)
assert result.returncode==0, 'Inspect remote state before any further action; never automatically redispatch.'
receipt=json.loads((HERE/'push_dispatch.stdout.json').read_bytes())
with (HERE/'push.driver.dispatch.json').open('x',encoding='utf-8') as f:
    json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
