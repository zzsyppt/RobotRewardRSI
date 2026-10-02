"""Join saved CPU audit and codec source hashes; no data recomputation or writes to inputs."""
from pathlib import Path
import hashlib
import json

BASE = Path(__file__).resolve().parent
AUDIT = BASE / 'run_01/receipt.json'
CODEC = BASE.parent / 'attempt_01/artifacts/receipt.json'
MANIFEST = BASE / 'source_numeric_raw_manifest.json'
OUT = BASE / 'source_header_binding.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


audit, codec, manifest = (json.loads(p.read_text(encoding='utf-8')) for p in (AUDIT, CODEC, MANIFEST))
sources = {r['file']: (i, r) for i, r in enumerate(codec['original_input_refs'])}
numeric = {r['file']: (i, r) for i, r in enumerate(manifest['readable_materials'])}
headers = [(i, r) for i, r in enumerate(audit['checked_inputs']) if r['path'].endswith('/episode.json')]
manifest_source = next(r for r in audit['checked_inputs'] if r['path'].endswith('/numeric_raw/manifest.json'))
checks = [dict(check='exact_original_numeric_manifest_bytes', equal=sha(MANIFEST) == manifest_source['before_sha256'] == manifest_source['after_sha256']),
          dict(check='audit_pass_with_limits', equal=audit['status'] == 'PASS_WITH_EXPLICIT_LIMITS'),
          dict(check='codec_original_inputs_unchanged', equal=codec['original_inputs_unchanged'] is True),
          dict(check='all_copied_raw_nonmanifest_files_unchanged', equal=any(c['check'] == 'all copied raw nonmanifest files unchanged' and c['status'] == 'PASS' for c in codec['checks'])),
          dict(check='all_original_source_files_unchanged', equal=any(c['check'] == 'all original source files unchanged' and c['status'] == 'PASS' for c in codec['checks'])),
          dict(check='header_count_28', equal=len(headers) == 28)]
rows = []
for index, record in headers:
    path = record['path']
    codec_index, original = sources[path]
    row = {'file': path, 'sha256': record['before_sha256'],
           'audit_pointer': '/checked_inputs/' + str(index),
           'codec_original_pointer': '/original_input_refs/' + str(codec_index),
           'audit_to_codec_equal': record['before_sha256'] == record['after_sha256'] == original['sha256']}
    if '/numeric_raw/' in path:
        relative = path.split('/numeric_raw/', 1)[1]
        manifest_index, declared = numeric[relative]
        row.update(numeric_manifest_pointer='/readable_materials/' + str(manifest_index),
                   numeric_manifest_equal=declared['sha256'] == record['before_sha256'])
    else:
        row['media_scope'] = 'Header bound to codec pre-rebuild original-file SHA and unchanged check; no separate historical media raw-manifest header declaration is asserted.'
    checks.append({'check': path, 'equal': row['audit_to_codec_equal'] and row.get('numeric_manifest_equal', True)})
    rows.append(row)
result = {'status': 'PASS' if all(c['equal'] for c in checks) else 'FAIL',
          'scope': 'Source-header correspondence only; joins existing evidence without rerunning arithmetic or claiming semantic qualification.',
          'inputs': [{'file': str(p), 'sha256': sha(p)} for p in (AUDIT, CODEC, MANIFEST, Path(__file__).resolve())],
          'checks': checks, 'headers': rows, 'header_count': len(rows),
          'numeric_manifest_bound_count': sum('numeric_manifest_equal' in r for r in rows),
          'media_codec_snapshot_bound_count': sum('media_scope' in r for r in rows),
          'model_calls': 0, 'gpu_calls': 0}
with OUT.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ('status', 'header_count', 'numeric_manifest_bound_count', 'media_codec_snapshot_bound_count')}))
raise SystemExit(0 if result['status'] == 'PASS' else 1)
