from pathlib import Path
import hashlib,json,os,sys
base=Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
source=base/'n5_selection_repair_source_r1'
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(source/'src'))
from true_rl_agent.evaluation import visual_review
module=Path(visual_review.__file__).resolve()
assert module == source/'src/true_rl_agent/evaluation/visual_review.py'
p=base/'n5_confirmation_20260930/engineering_repair/observers/push/combined_audit.json'
b=p.read_bytes()
print(json.dumps({'status':'DEVELOPMENT_REPAIR_NOT_ORIGINAL_N5_CONFIRMATION','original_n5_status':'INCONCLUSIVE','combined_audit':{'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)},'visual_review_binding':visual_review.visual_review_binding(json.loads(b)),'visual_review_module':{'path':str(module),'sha256':hashlib.sha256(module.read_bytes()).hexdigest(),'size_bytes':module.stat().st_size},'new_model_calls':0,'semantic_review_performed':False}))