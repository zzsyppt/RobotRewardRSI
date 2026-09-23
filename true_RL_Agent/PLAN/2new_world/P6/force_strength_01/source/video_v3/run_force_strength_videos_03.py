from pathlib import Path
import subprocess,json,time
from concurrent.futures import ThreadPoolExecutor
r=Path('/data/yuyangliu/AGENT/true_RL_Agent');d=r/'PLAN/2new_world/P6/force_strength_01'
commands=json.loads((d/'records/video_declaration_v3.json').read_text())['commands']
def lane(index):
 rows=[]
 for item in commands:
  if item['lane']!=index:continue
  with (d/'records'/('video_launch_v3_'+item['case_id']+'.log')).open('w') as log:
   start=time.time();z=subprocess.run(item['argv'],cwd=r,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
  rows.append({'case_id':item['case_id'],'returncode':z.returncode,'start':start,'end':time.time()});(d/'records'/f'video_jobs_v3_lane{index}.json').write_text(json.dumps(rows,indent=2))
  if z.returncode:break
 return rows
with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(lane,[0,1]))
(d/'records/video_jobs_v3.json').write_text(json.dumps([r for rows in results for r in rows],indent=2))
