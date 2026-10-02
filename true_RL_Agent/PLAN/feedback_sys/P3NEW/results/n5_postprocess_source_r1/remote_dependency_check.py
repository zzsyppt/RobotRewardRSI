from pathlib import Path
import hashlib,json
from datetime import datetime,timezone
root=Path('/data/yuyangliu/AGENT/feedback_p3new_20260929/text_author_smoke_r1')
rels=['PLAN/feedback_sys/P3R/confirmation_context.py', 'PLAN/feedback_sys/P3R/author_ceiling_probe.py', 'PLAN/feedback_sys/P3/author_experiment.py']
rows=[]
for r in rels:
 p=root/r
 raw=p.read_bytes()
 rows.append(dict(repository_relative_path=r,path=str(p),sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw)))
print(json.dumps(dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),target_exists=Path('/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_postprocess_source_r1').exists(),files=rows)))
