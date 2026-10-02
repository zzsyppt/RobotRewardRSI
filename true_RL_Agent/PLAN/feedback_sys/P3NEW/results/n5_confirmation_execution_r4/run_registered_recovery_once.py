"""One explicitly registered DS episode-13 call; no continuation or automatic retry."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys

C = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_execution_20261001_r4')
S = C.parent / 'n5_confirmation_source_r4_final'
OUT = C / 'amendment_r1_recovery'

def read(p): return json.loads(Path(p).read_bytes())
def ref(p):
    p = Path(p)
    raw = p.read_bytes()
    return {'path': str(p), 'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}
def save(p, value):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')
def now(): return datetime.now(timezone.utc).isoformat()

def main():
    assert not OUT.exists(), 'The single registered attempt already has a record'
    assert not (C / 'STOP_FACTS_DISPATCH_AMENDMENT_R1').exists()
    card_path = C / 'runtime_limit_amendment_r1.json'
    assert ref(card_path)['sha256'] == 'c08ace063b3ae6a34300308cba856e82294bbe25fdfc0573865a192557f1bf34'
    card = read(card_path)
    assert card['recovery_gate']['maximum_new_attempts'] == 1
    prep_receipt = C / 'amendment_r1_facts_preparation/preparation_receipt.json'
    assert ref(prep_receipt)['sha256'] == '2090a5ee711207b31b56cd77e907821b38137f201908b87858199f882d5df9f0'
    row = next(r for r in read(prep_receipt)['slots'] if r['packet_id'] == 'episode-13')
    for key in ('original_request', 'new_prepare', 'new_request', 'reference_catalog'):
        assert ref(row[key]['path']) == row[key]
    prepared = Path(row['new_prepare']['path']).parent
    old, new = read(row['original_request']['path']), read(row['new_request']['path'])
    assert old['max_tokens'] == 32768 and new == {**old, 'max_tokens': 65536}
    assert not any((prepared / n).exists() for n in ('receipt.json', 'raw_answer.txt', 'raw_response.json'))
    runner = S / 'PLAN/feedback_sys/P3NEW/run_confirmation_review.py'
    assert ref(runner)['sha256'] == '1aa1c6c6dce603d06d2e6cf2c7eeeb9f31dad8e65148781d86d10d0bfa4f574f'
    budget = read(C / 'capture_card.json')['budget']
    proposal = read(C / 'facts_dispatch_preparation/deepseek_reservation_proposal.json')
    reserve = next(r['suggested_input_token_reservation'] for r in proposal['rows'] if r['packet_id'] == 'episode-13')
    assert reserve == 102346 and (reserve * 0.3 + 65536 * 1.2) / 1e6 <= 0.20
    ledger_path = C / 'budget_ledger.working.json'
    release = read(prepared / 'release.draft.json')
    assert release['output_token_reservation'] == 65536
    release.update(status='RELEASED_FOR_ONE_CALL', input_token_reservation=reserve,
                   reserved_usd=0.20, prices_usd_per_million=budget['prices_usd_per_million'],
                   pricing_source=budget['pricing_source'])
    release['budget'] = {'ledger_file': str(ledger_path), 'ledger_sha256': ref(ledger_path)['sha256'],
                         'budget_id': budget['budget_id'], 'limit_usd': budget['api_phase_limit_usd']}
    OUT.mkdir()
    save(OUT / 'only_max_tokens_changed.json', {
        'status': 'PASS_ONLY_MAX_TOKENS_CHANGED', 'amendment': ref(card_path),
        'old_request_sha256': row['original_request']['sha256'], 'new_request_sha256': row['new_request']['sha256'],
        'old_max_tokens': 32768, 'new_max_tokens': 65536,
        'prepared': row['new_prepare'], 'request': row['new_request'], 'original_request': row['original_request'],
        'whole_messages_images_and_catalog_unchanged': True, 'cpu_preparation': ref(prep_receipt)})
    release_path = OUT / 'release.json'
    save(release_path, release)
    argv = [sys.executable, '-B', str(runner), 'run', '--prepared', str(prepared),
            '--release', str(release_path), '--secret', '/data/yuyangliu/secrets/deepseek.env']
    save(OUT / 'dispatch.json', {'started_at_utc': now(), 'script': ref(__file__), 'amendment': ref(card_path),
                               'argv': argv, 'release': ref(release_path), 'scope': 'one_registered_recovery_only'})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', PYTHONPATH=os.pathsep.join(map(str, (S / 'src', S, runner.parent))))
    with (OUT / 'stdout.log').open('xb') as stdout, (OUT / 'stderr.log').open('xb') as stderr:
        code = subprocess.run(argv, cwd=S, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr).returncode
    receipt_path = prepared / 'receipt.json'
    receipt = read(receipt_path) if receipt_path.exists() else {}
    result = {'finished_at_utc': now(), 'returncode': code, 'runner_receipt': ref(receipt_path) if receipt else None,
              'complete': receipt.get('complete') is True, 'automatic_retries': 0, 'facts_seals': 0, 'report_calls': 0,
              'next_batch_started': False, 'structure_audit_still_required': True}
    save(OUT / 'execution_receipt.json', result)
    print(json.dumps(result))
    return 0 if code == 0 and result['complete'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
