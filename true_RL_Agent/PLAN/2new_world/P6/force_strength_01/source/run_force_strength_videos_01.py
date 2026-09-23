from pathlib import Path
import subprocess,json,time
r=Path('/data/yuyangliu/AGENT/true_RL_Agent');d=r/'PLAN/2new_world/P6/force_strength_01'; rows=[]
for item in json.loads((d/'records/video_declaration.json').read_text())['commands']:
 with (d/'records'/('video_launch_'+item['case_id']+'.log')).open('w') as log:
  start=time.time(); z=subprocess.run(item['argv'],cwd=r,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
 rows.append({'case_id':item['case_id'],'returncode':z.returncode,'start':start,'end':time.time()});(d/'records/video_jobs.json').write_text(json.dumps(rows,indent=2))
 if z.returncode:break
