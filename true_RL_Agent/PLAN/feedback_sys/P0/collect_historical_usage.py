"""Bind saved authoring and metering receipts to their actual prompt bytes."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows = []
    roots = [OUT / 'inventory_remote_raw', ROOT / 'PLAN/2new_world/P5/provider_live_01']
    for root in roots:
        for path in sorted(root.rglob('attempt_result.json')):
            raw = json.loads(path.read_text(encoding='utf-8'))
            attempt = raw.get('provider_attempt', {})
            prompt = path.parent / 'prompt_payload.json'
            if not prompt.exists():
                continue
            frame = json.loads(prompt.read_text(encoding='utf-8'))
            usage = attempt.get('provider_usage')
            row = {'receipt_path': path.relative_to(ROOT).as_posix(), 'receipt_sha256': sha(path),
                   'prompt_path': prompt.relative_to(ROOT).as_posix(), 'prompt_sha256': sha(prompt),
                   'receipt_prompt_checksum': raw.get('prompt_payload_checksum'),
                   'prompt_file_bytes': prompt.stat().st_size, 'prompt_schema': frame.get('schema_version'),
                   'frame_kind': frame.get('authoring_context_frame', {}).get('frame_kind', 'legacy_unspecified'),
                   'provider_status': attempt.get('provider_status'), 'usage': usage,
                   'provider_receipt': attempt.get('provider_response_receipt'),
                   'kind': 'actual_historical_authoring_attempt', 'rewired_current_request': False}
            row['prompt_hash_matches_receipt'] = row['receipt_prompt_checksum'] == 'sha256:' + row['prompt_sha256']
            if usage and usage.get('request_units'):
                row['file_bytes_per_prompt_token'] = row['prompt_file_bytes'] / usage['request_units']
            rows.append(row)
    p5 = ROOT / 'PLAN/2new_world/P5'
    raw = json.loads((p5 / 'audit_capacity/result.json').read_text())
    for name, value in raw['frames'].items():
        path = p5 / 'audit_capacity' / (name + '.json')
        rows.append({'kind': 'historical_one_output_token_metering', 'name': name, **value,
                     'actual_copy_sha256': sha(path), 'prompt_hash_matches_receipt': sha(path) == value['source_sha256']})
    report = {'rows': rows, 'notes': ['Raw bytes of old prompt schemas are not current compact wire bytes.',
              'source_followup capacity branch was constructed; no natural historical source request is claimed.',
              'minimum-core live usage is measured by P0, no historical receipt was found for that exact branch.'],
              'actual_authoring_count': sum(x['kind'] == 'actual_historical_authoring_attempt' for x in rows),
              'hash_match_count': sum(x['prompt_hash_matches_receipt'] for x in rows),
              'row_count': len(rows)}
    (OUT / 'historical_prompt_usage.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('rows', 'notes')}))


if __name__ == '__main__':
    main()
