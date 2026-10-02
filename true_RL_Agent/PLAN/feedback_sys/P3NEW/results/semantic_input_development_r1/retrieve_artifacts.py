"""Copy completed remote receipts and reviewable JSON; preserve both attempts."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

D = Path(__file__).resolve().parent
attempt = sys.argv[1] if len(sys.argv) == 2 else 'attempt_01'
if not attempt.startswith('attempt_') or not attempt[8:].isdigit():
    raise ValueError('explicit attempt_NN required')
destination = D / attempt / 'artifacts'
if destination.exists():
    raise ValueError('refuse overwriting retrieved artifacts')
remote = '/data/yuyangliu/AGENT/feedback_p3new_20260929/semantic_input_development_r1/' + attempt
script = '''import io,pathlib,sys,tarfile
root=pathlib.Path(ROOT)
names=['receipt.json','failure.json','numeric_verification.json','text_capacity.json','executed_source_refs.json','original_raw_manifest.json','raw_development_view/manifest.json','numeric_materials/packing_receipt.json']
paths=[root/name for name in names if (root/name).is_file()]
for pattern in ('numeric_materials/materials/related/*.json','numeric_materials/packets/*/facts.manifest.json','media_materials/*/related_numeric.json','media_materials/*/facts.manifest.json','media_materials/*/verification.json'):
    paths.extend(sorted(root.glob(pattern)))
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
    for path in paths: archive.add(path,arcname=str(path.relative_to(root)),recursive=False)
'''
source = ('ROOT = ' + repr(remote) + '\n' + script).encode()
completed = subprocess.run(['ssh', 'humanoid-server', '/data/yuyangliu/envs/isaaclab6_validation/bin/python', '-B', '-'],
    input=source, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
if completed.returncode:
    raise RuntimeError(completed.stderr.decode())
destination.mkdir()
records = []
with tarfile.open(fileobj=io.BytesIO(completed.stdout), mode='r:gz') as archive:
    for member in archive:
        path = destination / member.name
        if not member.isfile() or '..' in Path(member.name).parts or Path(member.name).is_absolute():
            raise ValueError('invalid artifact member')
        raw = archive.extractfile(member).read()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
        records.append({'relative_file': member.name, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)})
receipt = {'remote_root': remote, 'files': records, 'stream_sha256': hashlib.sha256(completed.stdout).hexdigest(),
           'remote_writes': 0, 'model_calls': 0, 'gpu_calls': 0}
with (D / attempt / 'retrieval.json').open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, indent=2)
print(json.dumps({'files_retrieved': len(records), 'destination': str(destination)}))
