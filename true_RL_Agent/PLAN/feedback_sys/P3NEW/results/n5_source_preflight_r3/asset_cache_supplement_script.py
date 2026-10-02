from pathlib import Path
import json,hashlib,os
p=Path('/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_preflight_r3/asset_cache_check.json')
a=json.loads(p.read_bytes());expected=a['expected'];sizes={e['size_bytes'] for e in expected}
roots=[Path.home()/'.cache/ov',Path.home()/'.local/share/ov']
rows=[];count=0
for root in roots:
 if not root.exists():continue
 for folder,dirs,names in os.walk(root):
  for name in names:
   p=Path(folder)/name
   try:
    if p.stat().st_size not in sizes:continue
    count+=1;h=hashlib.sha256(p.read_bytes()).hexdigest()
    for e in expected:
     if h==e['sha256']:rows.append({'path':str(p),'sha256':h,'identity':e['identity'],'size_bytes':p.stat().st_size})
   except (OSError,PermissionError):pass
print(json.dumps({'cache_roots':[str(x) for x in roots],'matching_assets':rows,'size_candidate_files_hashed':count,'boundary':'Read-only cached asset bytes; runtime resolver/load remains unexecuted.'}))
