import hashlib,json,tarfile,datetime
from pathlib import Path
s=Path('/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_confirmation_source_r5_final');c=Path('/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_execution_20261001_r5/source_control')
expected={'source.tar.gz': {'sha256': 'dfdbeaa9c6e35b1b7eb1ed929a5e617fdea00a91315c22beb6126f0cb0cf2638', 'size_bytes': 1794807}, 'source_manifest.json': {'sha256': 'ca9e11f079200d26545f75a2550bcf0901afd59c02adee3d418695cb4ccfa2bd', 'size_bytes': 85723}, 'source_preparation.json': {'sha256': '19ab5797e71ccad80d139f3429ccf5fbba0254c6d0f3eef5ff61b96fd4810372', 'size_bytes': 4727}, 'source_delta.json': {'sha256': 'ef3f9fa33b8026ddf22eb6b99712f0b007dfed5257a0cb8c89aeee0d7e02be78', 'size_bytes': 5791}, 'commands.candidate.json': {'sha256': '914cac21a33c0525f08082080b2f87decdd8147466d3b92344692b902a593f77', 'size_bytes': 9939}, 'history_cohorts.before.json': {'sha256': 'ecaaaffc023d0210acc2fd0c6c6ac2768c0049e0d3b35033cf2cdaa336d797f4', 'size_bytes': 41852}, 'r4_capture_card.readonly.json': {'sha256': '811d7f140a5b3015f66a5093ba516c60bdc4a05c9ac827461d19bb05b8be3041', 'size_bytes': 46242}, 'remote_cpu.py': {'sha256': 'e6f2a57cce7c67ad541868f054b0c173984a53005635fa50b607a3c7fb64b543', 'size_bytes': 8660}, 'source_review_v2.json': {'sha256': '352c238d3cb885fbd74b13e53e4814a8a6b39bfeec445e070c03dff187bff71c', 'size_bytes': 99932}, 'root_source_cpu_authorization.json': {'sha256': 'd30d43977aaa4a3ac36af843f4cb039e316718ddba1298436ecb0e18f8d1759d', 'size_bytes': 744}}
def item(p):return {'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size_bytes':p.stat().st_size}
assert {n:item(c/n) for n in expected}==expected
assert not list(s.iterdir())
manifest=json.loads((c/'source_manifest.json').read_bytes())
with tarfile.open(c/'source.tar.gz','r:gz') as archive:
 members=archive.getmembers()
 assert len(members)==len(manifest) and {m.name for m in members}==set(manifest)
 for m in members:
  assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
 archive.extractall(s)
actual={p.relative_to(s).as_posix():item(p) for p in s.rglob('*') if p.is_file()}
assert actual==manifest
receipt={'status':'SOURCE_DEPLOYED_HASH_VERIFIED_CPU_PENDING','observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':str(s),'source_files':len(actual),'transferred':expected,'capture_root_absent':not Path('/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_confirmation_20261001_r5').exists(),'gpu_or_capture_or_model_calls':0}
with (c/'deployment_receipt.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps({'source_files':len(actual),'deployment_receipt':item(c/'deployment_receipt.json')}))
