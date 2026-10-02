ROOT='/data/yuyangliu/AGENT/feedback_p3new_20260929/text_author_smoke_r1/PLAN/feedback_sys/P3NEW/results/native_author_current_r1'
DRIVER='/data/yuyangliu/AGENT/feedback_p3new_20260929/text_author_smoke_r1/PLAN/feedback_sys/P3NEW/results/native_author_current_r1/driver/send_native_author_compat.py'
EXPECTED='1de27eed62ae1da55fec9f9daf76b2e18549866cd70ce0e90e813430c37c9760'

from pathlib import Path
import hashlib,json
root=Path(ROOT);draft=root/'stage.draft.json';raw=draft.read_bytes();assert hashlib.sha256(raw).hexdigest()==EXPECTED
v=json.loads(raw);assert v['status']=='PREPARED_NOT_RELEASED'
assert v['budget']['historical_actual_calls']==6 and v['episode_budget']['total_if_run']==8
assert v['budget']['new_actual_calls_cap']==2 and v['budget']['max_attempts_per_new_episode']==1
assert v['budget']['total_with_two_peak_reservations_usd']<4
path=Path(DRIVER);ref={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'size_bytes':path.stat().st_size}
v['status']='FROZEN_API_RELEASED';v['pricing_status']='VERIFIED_FOR_AUTHOR_MODEL';v['execution_script']=ref
if ref not in v['frozen_inputs']:v['frozen_inputs'].append(ref)
out=root/'stage.released.json'
with out.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,sort_keys=True,separators=(',',':'))
print(json.dumps({'status':v['status'],'stage_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'driver_sha256':ref['sha256'],'new_call_cap':2,'episode_limit':8}))
