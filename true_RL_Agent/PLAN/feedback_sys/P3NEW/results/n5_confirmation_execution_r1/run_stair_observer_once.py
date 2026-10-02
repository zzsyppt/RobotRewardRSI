"""N5 execution adapter: global ledger around the unchanged two-slot sender."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

BASE = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
SOURCE = BASE / 'n5_postprocess_source_r1'
CONTROL = BASE / 'n5_execution_20260930'
REPORT = BASE / 'n5_confirmation_20260930/observers/stair'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
sys.path[:0] = [str(SOURCE / p) for p in (
    'src', 'PLAN/feedback_sys/P3NEW', 'PLAN/feedback_sys/P3R', 'PLAN/feedback_sys/P3')]
import run_confirmation_review as R
import send_native_observer as S

parser = argparse.ArgumentParser()
parser.add_argument('alias', choices=('e1', 'e2'))
alias = parser.parse_args().alias
ledger_path = CONTROL / 'budget_ledger.working.json'
receipt_path = REPORT / alias / 'receipt.json'
request_sha = hashlib.sha256((REPORT / alias / 'request.json').read_bytes()).hexdigest()
with R.ledger_lock(ledger_path):
    if not ledger_path.exists():
        initial = CONTROL / 'budget_ledger.initial.json'
        assert hashlib.sha256(initial.read_bytes()).hexdigest() == '0940e2218e6b20b1cfdd9cb86c52d7a2b7332e78db80d01ae41fd966cc806655'
        ledger_path.write_bytes(initial.read_bytes())
    ledger = R.read(ledger_path)
    assert ledger['budget_id'] == 'p3new-20260929-total-api'
    assert not receipt_path.exists()
    assert not any(e.get('request_sha256') == request_sha for e in ledger['entries'])
    assert sum(e['charged_usd'] for e in ledger['entries']) + .05 <= ledger['limit_usd'] == 15
    entry = dict(request_sha256=request_sha, judge_id='deepseek', packet_id='stair/' + alias,
                 stage='observer', receipt_file=str(receipt_path), status='reserved',
                 reserved_usd=.05, charged_usd=.05, charge_basis='reservation_unknown_actual')
    ledger['entries'].append(entry)
    R.save(ledger_path, ledger)
    try:
        result = S.send_report(REPORT, CONTROL / 'stair_observer_release.json',
                               Path('/data/yuyangliu/secrets/deepseek.env'), episode_alias=alias,
                               timeout_s=180)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        if receipt_path.exists():
            receipt = R.read(receipt_path)
            cost = R.usage_cost(receipt.get('usage'), {'cache_hit': .006, 'cache_miss': .3, 'output': 1.2})
            entry.update(status=receipt['status'], charged_usd=.05 if cost is None else cost,
                         charge_basis='reservation_unknown_actual' if cost is None else 'usage_at_registered_rates_not_invoice',
                         receipt_sha256=R.ref(receipt_path)['sha256'])
        R.save(ledger_path, ledger)
        print(json.dumps({'packet': 'stair/' + alias, 'budget_charged_usd': entry['charged_usd'],
                          'budget_basis': entry['charge_basis'], 'ledger_sha256': R.ref(ledger_path)['sha256']}), flush=True)
