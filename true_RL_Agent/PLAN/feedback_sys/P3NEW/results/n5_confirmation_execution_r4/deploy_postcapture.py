"""Deploy the reviewed R4 CPU controls only; never execute capture or derivation."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
PREP = HERE / 'postcapture_preparation'
manifest = json.loads((PREP / 'preparation_manifest.json').read_bytes())
sha = lambda data: hashlib.sha256(data).hexdigest()
addition = (
    "    require(card['review_assignment']['sha256'] == ref(ASSIGNMENT)['sha256']\n"
    "            and card['review_assignment']['remote_path'] == str(ASSIGNMENT), 'Assignment/card mismatch')\n"
)
for row in manifest['controls']:
    parent = Path(row['parent']['path']).read_bytes()
    current = (PREP / row['file']).read_bytes()
    assert sha(parent) == row['parent']['sha256']
    assert sha(current) == row['prepared']['sha256']
    expected = parent.decode('utf-8')
    for change in row['changes']:
        if 'before' in change:
            assert expected.count(change['before']) == change['occurrences']
            expected = expected.replace(change['before'], change['after'])
        else:
            assert row['file'] == 'postcapture_cpu.py'
            expected = expected.replace('    return card\n', addition + '    return card\n')
    assert expected.encode('utf-8') == current, row['file']
names = [row['file'] for row in manifest['controls']] + [
    'preparation_manifest.json', 'static_verification.json', 'README_ZH.md']
files = {}
for name in names:
    data = (PREP / name).read_bytes()
    files[name] = {'base64': base64.b64encode(data).decode(), 'sha256': sha(data), 'size_bytes': len(data)}
payload = {
    'control_root': manifest['execution_control_root'],
    'capture_card_sha256': manifest['capture_card']['sha256'],
    'source_manifest_sha256': manifest['source_manifest']['sha256'],
    'files': files,
}
remote = r'''
import base64,hashlib,json,pathlib,sys
from datetime import datetime,timezone
p=json.loads(sys.stdin.buffer.read())
c=pathlib.Path(p['control_root']); h=c/'postcapture_preparation'
sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha((c/'capture_card.json').read_bytes())==p['capture_card_sha256']
assert sha((c/'runtime_source_manifest.json').read_bytes())==p['source_manifest_sha256']
assert not h.exists(), 'Preserve existing deployment'
decoded={}
for name,item in p['files'].items():
 assert pathlib.PurePath(name).name==name
 data=base64.b64decode(item['base64'],validate=True)
 assert len(data)==item['size_bytes'] and sha(data)==item['sha256']
 decoded[name]=data
h.mkdir()
for name,data in decoded.items():
 with (h/name).open('xb') as f:f.write(data)
 assert sha((h/name).read_bytes())==p['files'][name]['sha256']
r={'status':'DEPLOYED_NOT_EXECUTED','recorded_at_utc':datetime.now(timezone.utc).isoformat(),
   'root':str(h),'capture_card_sha256':p['capture_card_sha256'],
   'source_manifest_sha256':p['source_manifest_sha256'],
   'files':{n:{k:v for k,v in i.items() if k!='base64'} for n,i in p['files'].items()},
   'derivation_executed':False,'model_api_gpu_calls':0,'r4_raw_or_images_read':False}
with (h/'deployment_receipt.json').open('x',encoding='utf-8') as f:json.dump(r,f,indent=2)
print(json.dumps(r))
'''
compile(remote, '<remote_deployment>', 'exec')
# Pass program and JSON separately: the shell receives only a base64 Python program.
program = base64.b64encode(remote.encode()).decode()
command = "/data/yuyangliu/envs/isaaclab6_validation/bin/python -B -c \"import base64;exec(base64.b64decode('" + program + "'))\""
with (HERE / 'postcapture_deployment.stdout.json').open('xb') as out, (HERE / 'postcapture_deployment.stderr.txt').open('xb') as err:
    result = subprocess.run(['ssh', 'humanoid-server', command], input=json.dumps(payload).encode(), stdout=out, stderr=err)
receipt = {'returncode': result.returncode, 'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
           'script_sha256': sha(Path(__file__).read_bytes()), 'reviewed_controls': len(manifest['controls']),
           'parent_plus_recorded_changes_equal': True,
           'stdout_sha256': sha((HERE/'postcapture_deployment.stdout.json').read_bytes()),
           'stderr_sha256': sha((HERE/'postcapture_deployment.stderr.txt').read_bytes())}
with (HERE / 'postcapture_deployment.json').open('x', encoding='utf-8') as f:
    json.dump(receipt, f, indent=2)
print(json.dumps(receipt))
raise SystemExit(result.returncode)
