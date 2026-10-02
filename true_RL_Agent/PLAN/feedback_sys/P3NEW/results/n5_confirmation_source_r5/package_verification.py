from pathlib import Path
import hashlib,json,tarfile,datetime
c=Path('/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_execution_20261001_r5/source_control');p=Path('/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_execution_20261001_r5/cpu_preflight')
assert json.loads((c/'verification.json').read_bytes())['status']=='CPU_STATIC_PREFLIGHT_PASS_NOT_CAPTURE_RELEASE'
entries=[(q,'control/'+q.name) for q in sorted(c.iterdir()) if q.is_file() and q.suffix in ('.json','.txt','.py')]
entries += [(q,'preflight/'+q.relative_to(p).as_posix()) for q in sorted(p.rglob('*')) if q.is_file()]
assert all(not q.is_symlink() for q,n in entries)
assert len({n for q,n in entries})==len(entries)
assert sum(q.stat().st_size for q,n in entries)<30*1024**2
members={n:{'sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'size_bytes':q.stat().st_size} for q,n in entries}
with tarfile.open(c/'verification_results.tar','x') as a:
 for q,n in entries:a.add(q,arcname=n,recursive=False)
receipt={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'archive_sha256':hashlib.sha256((c/'verification_results.tar').read_bytes()).hexdigest(),'archive_size_bytes':(c/'verification_results.tar').stat().st_size,'members':members,'scope':'original CPU controls/receipts/logs and both complete preflight directories; no capture/model'}
with (c/'verification_archive_receipt.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps({'members':len(members),'sha256':receipt['archive_sha256'],'bytes':receipt['archive_size_bytes']}))
