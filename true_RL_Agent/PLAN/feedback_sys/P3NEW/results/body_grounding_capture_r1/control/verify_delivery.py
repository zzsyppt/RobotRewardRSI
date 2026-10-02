"""Verify the single transferred run-1 bundle before exposing its original files."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import tarfile

root = Path(__file__).resolve().parent.parent
archive = root / 'receipt/run01_review_bundle.tar.gz.download'
expected = '29ab4e1373ec70342cec0357e8d13cbd36119d616104076654ad72da72ab13e8'
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
if digest != expected:
    raise ValueError('downloaded bundle hash mismatch')
with tarfile.open(archive, 'r:gz') as bundle:
    index_bytes = bundle.extractfile('receipt/download_index.json').read()
    index = json.loads(index_bytes)
    if set(bundle.getnames()) != set(index['files']) | {'receipt/download_index.json'}:
        raise ValueError('bundle inventory differs')
    for name, metadata in index['files'].items():
        data = bundle.extractfile(name).read()
        if len(data) != metadata['bytes'] or hashlib.sha256(data).hexdigest() != metadata['sha256']:
            raise ValueError('original evidence hash mismatch: ' + name)
        target = (root / name).resolve()
        if not target.is_relative_to(root):
            raise ValueError('bundle target outside intended directory')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != data:
                raise ValueError('existing local evidence differs: ' + name)
        else:
            with target.open('xb') as stream:
                stream.write(data)
    with (root / 'receipt/download_index.json').open('xb') as stream:
        stream.write(index_bytes)
final_archive = archive.with_suffix('')
if final_archive.exists() or not final_archive.resolve().is_relative_to(root):
    raise ValueError('unexpected final archive target')
archive.rename(final_archive)
receipt = {'status': 'pass', 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
           'bundle_sha256': digest, 'original_evidence_files_verified': len(index['files']),
           'run1_original_png_count': index['run1_image_count'], 'machine_verification_status': index['verification_status'],
           'independent_image_review': 'pending', 'provider_requests': 0, 'gpu_capture_attempts': 1}
with (root / 'receipt/local_delivery_verification.json').open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, indent=2)
print(json.dumps(receipt))
print(str(root / 'materials/push_run01/video'))
