from pathlib import Path
import sys,json,hashlib,importlib.util
from datetime import datetime,timezone
root=Path('/data/yuyangliu/AGENT/feedback_p3new_20260929/text_author_smoke_r1')
sys.path.insert(0,'/data/yuyangliu/AGENT/feedback_p3new_20260929/source_observer_v3_r8/src')
sys.path.insert(0,str(root/'PLAN/feedback_sys/P3NEW'))
file=root/'PLAN/feedback_sys/P3NEW/send_native_author_smoke_after_recovery.py'
spec=importlib.util.spec_from_file_location('historical_n4_readonly',file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
shared=m.SHARED
paths=sorted(shared.rglob('*.json'))
before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
card=json.loads((root/'PLAN/feedback_sys/P3NEW/results/native_author_smoke_r3/stage.runtime_recovered.json').read_bytes())
usage=m.accounting(card,shared);m.check_usage(usage,card,expected_count=6)
assert before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
slots={str(p.relative_to(shared)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (shared/'authoring').rglob('attempt_result.json')}
new=shared/'authoring/p3new-text-smoke-20260929-generation-004'
print(json.dumps({'status':'PASS_READONLY_EXISTING_SHARED_JOURNAL','checked_at_utc':datetime.now(timezone.utc).isoformat(),'shared_journal':str(shared),'reader_source':str(file),'reader_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'all_prior_json_hashes_unchanged':True,'existing_json_count':len(paths),'attempts':usage['attempts'],'usd_peak_actual_or_reserved':usage['usd_peak_actual_or_reserved'],'violations':usage['violations'],'attempt_result_sha256':slots,'generation004_exists':new.exists(),'usage':usage,'model_calls':0,'writes_to_shared_journal':0}))
