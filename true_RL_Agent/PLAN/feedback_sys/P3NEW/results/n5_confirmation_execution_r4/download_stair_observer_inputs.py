"""Stream only the two prepared Stair observer packages; retain original bytes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import struct
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
DEST = HERE / 'stair_observer_inputs'
EXPECTED = {'e1':'40383b22d1237c7fd8816e8d07d8a69a7c6725b1a375aff52a5cb376085a8ffc',
            'e2':'0647c5e36b3938cfac3ed5309a9297b0d58196312ee7ea1409a7f98c7635ced5'}
assert not DEST.exists()
remote = r'''
import pathlib,json,hashlib,tarfile,sys
b=pathlib.Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
r=b/'n5_confirmation_20261001_r4/observers/stair'
expected={'e1':'40383b22d1237c7fd8816e8d07d8a69a7c6725b1a375aff52a5cb376085a8ffc','e2':'0647c5e36b3938cfac3ed5309a9297b0d58196312ee7ea1409a7f98c7635ced5'}
files=[]
for alias,sha in expected.items():
 blind=r/alias/'blind_inputs';raw=(blind/'materials.json').read_bytes()
 assert hashlib.sha256(raw).hexdigest()==sha
 m=json.loads(raw)['materials'];assert len(m)==13
 files.append((blind/'materials.json',alias+'/blind_inputs/materials.json'))
 for item in m:
  p=blind/item['path'];assert p.resolve().is_relative_to(blind.resolve())
  assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
  files.append((p,alias+'/blind_inputs/'+item['path']))
 files.append((r/alias/'carrier_preparation.json',alias+'/carrier_preparation.json'))
files.extend([(r/'sol_carrier_preparation_receipt.json','control/sol_carrier_preparation_receipt.json'),
 (b/'n5_execution_20261001_r4/postcapture/stair/receipt.json','control/postcapture_task_receipt.json')])
assert len(files)==32
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as t:
 for p,name in files:t.add(p,arcname=name,recursive=False)
'''
compile(remote, '<remote_package_stream>', 'exec')
DEST.mkdir()
names=set()
with (HERE/'stair_observer_transfer.stderr.txt').open('xb') as err:
    proc=subprocess.Popen(['ssh','humanoid-server','/data/yuyangliu/envs/isaaclab6_validation/bin/python','-B','-'],
                          stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err)
    proc.stdin.write(remote.encode());proc.stdin.close()
    print(json.dumps({'status':'TRANSFER_RUNNING','ssh_pid':proc.pid}),flush=True)
    with tarfile.open(fileobj=proc.stdout,mode='r|') as t:
        for member in t:
            rel=PurePosixPath(member.name)
            assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
            assert rel.parts[0] in ('e1','e2','control') and member.name not in names
            target=DEST.joinpath(*rel.parts)
            assert target.resolve().is_relative_to(DEST.resolve())
            target.parent.mkdir(parents=True,exist_ok=True)
            with t.extractfile(member) as src,target.open('xb') as dst:
                while chunk:=src.read(1024*1024):dst.write(chunk)
            assert target.stat().st_size==member.size
            names.add(member.name)
    assert proc.wait()==0 and len(names)==32
records=[]
for alias,expected in EXPECTED.items():
    blind=DEST/alias/'blind_inputs';raw=(blind/'materials.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==expected
    materials=json.loads(raw)['materials']
    assert {str(f.relative_to(blind)).replace('\\','/') for f in blind.rglob('*') if f.is_file()}=={'materials.json',*(r['path'] for r in materials)}
    images=[]
    for row in materials:
        data=(blind/row['path']).read_bytes();assert hashlib.sha256(data).hexdigest()==row['sha256']
        if row['role']=='original_image':
            assert data[:8]==b'\x89PNG\r\n\x1a\n'
            assert list(struct.unpack('>II',data[16:24]))==row['image_size_px']==[1920,1080]
            images.append(row['path'])
    assert len(images)==12
    records.append({'alias':alias,'materials_sha256':expected,'files':14,'original_pngs':12,
                    'total_bytes':sum(f.stat().st_size for f in blind.rglob('*') if f.is_file())})
control=DEST/'control/sol_carrier_preparation_receipt.json'
assert hashlib.sha256(control.read_bytes()).hexdigest()=='561d50615c6f310cb4186d83dd6ee83714eeb6a1a14ae6b47ef4fa67095e0e33'
receipt={'status':'EXACT_BLIND_PACKAGES_TRANSFERRED_NOT_CALLED','recorded_at_utc':datetime.now(timezone.utc).isoformat(),
         'packages':records,'files_total':32,'source_card_sha256':'811d7f140a5b3015f66a5093ba516c60bdc4a05c9ac827461d19bb05b8be3041',
         'archive_files_created':0,'model_calls':0}
with (DEST/'transfer_receipt.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt),flush=True)
