"""Read-only P5 evidence excerpt; never copy source bodies or private evaluation."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/data/yuyangliu/AGENT/true_RL_Agent/outputs/formal_go2_stair_15round_20260901_c6ab8ba')
rows = []
for record in map(json.loads, (ROOT / 'ledger.jsonl').read_text().splitlines()):
    if record['generation_index'] not in (1, 5, 6, 7, 10, 11, 13, 14, 15):
        continue
    row = {'generation': record['generation_index'], 'outcome': record['generation_outcome'], 'lanes': []}
    for trial in record['trials']:
        lane = {'lane': trial['lane_id'], 'status': trial['trial_status'], 'reason': trial['status_reason'],
                'train_completed': trial['train'] is not None, 'public_dev_completed': trial['public_dev'] is not None}
        artifact = next((item for item in record['private_artifacts']
                         if item['kind'] == trial['lane_id'] + '_authoring_evidence'), None)
        if artifact:
            path = Path(artifact['uri']).resolve()
            assert path.is_relative_to(ROOT)
            blob = path.read_bytes()
            assert artifact['checksum'] == 'sha256:' + hashlib.sha256(blob).hexdigest()
            evidence = json.loads(blob)
            lane['evidence_checksum'] = artifact['checksum']
            lane['attempts'] = []
            for attempt in evidence['attempts']:
                data = attempt.get('provider_attempt', {})
                lane['attempts'].append({key: data.get(key) for key in
                                        ('attempt_id', 'candidate_id', 'verifier_status', 'verifier_reason',
                                         'component_smoke', 'provider_failure')})
            prompts = sorted((ROOT / 'authoring' / record['generation_id'] / trial['lane_id'] /
                              'authoring_attempts').glob('attempt_*/prompt_payload.json'))
            lane['prompts'] = []
            for path in prompts:
                blob = path.read_bytes()
                prompt = json.loads(blob)
                feedback = prompt.get('public_feedback_summary', {})
                repair = prompt.get('provider_or_runtime_repair_feedback', {})
                lane['prompts'].append({
                    'attempt': path.parent.name, 'sha256': hashlib.sha256(blob).hexdigest(),
                    'stored_bytes': len(blob),
                    'compact_bytes': len(json.dumps(prompt, ensure_ascii=False, separators=(',', ':')).encode()),
                    'feedback_candidate_ids': [entry.get('candidate_id') for entry in feedback.get('feedback_entries', [])],
                    'runtime_candidate_id': prompt.get('runtime_feedback_summary', {}).get('source_candidate_id'),
                    'repair_category': repair.get('failure_category'), 'repair_candidate_id': repair.get('previous_candidate_id'),
                    'source_checksums': {item['logical_id']: item.get('content_sha256')
                                         for item in prompt.get('source_payloads', [])},
                })
        row['lanes'].append(lane)
    rows.append(row)
print(json.dumps({'source': str(ROOT), 'read_only': True, 'generations': rows}, ensure_ascii=True, indent=2))
