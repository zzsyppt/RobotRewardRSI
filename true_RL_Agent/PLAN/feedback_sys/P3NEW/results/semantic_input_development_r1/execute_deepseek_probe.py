"""One registered episode-02 development call through the frozen existing runner."""
from pathlib import Path
import hashlib
import json
import sys

BASE = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
DEV = BASE / 'semantic_input_development_r1'
ATTEMPT = DEV / 'attempt_01'
CONTROL = DEV / 'probes'
sys.path.insert(0, str(ATTEMPT / 'source_snapshot/PLAN/feedback_sys/P3NEW'))
import run_confirmation_review as runner


def ref(path):
    return {'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def read(path):
    return json.loads(path.read_bytes())


engineering = read(ATTEMPT / 'receipt.json')
capacity = read(ATTEMPT / 'text_capacity.json')
arithmetic_path = DEV / 'arithmetic_input_audit/run_01/receipt.json'
arithmetic = read(arithmetic_path)
binding_path = CONTROL / 'source_header_binding.json'
binding = read(binding_path)
assert engineering['status'] == 'PASS_ENGINEERING_ONLY_NOT_RELEASED'
assert engineering['original_inputs_unchanged'] and engineering['prepared_requests'] == 80
assert arithmetic['status'] == 'PASS_WITH_EXPLICIT_LIMITS' and arithmetic['difference_count'] == 0
assert binding['status'] == 'PASS' and binding['header_count'] == 28
assert ref(arithmetic_path)['sha256'] == '5814b10b2298053f1fa92ba812606eb3c080b7b3e1ef9f49e5cfaeea91955825'
assert ref(binding_path)['sha256'] == '88ea141609fd300b709225d474eea051fe9a1c974a81ea1675430e00451b4f54'
prepared = ATTEMPT / 'numeric_materials/prepared/episode-02/deepseek'
measured, = [r for r in capacity['records'] if r['packet_id'] == 'episode-02' and r['judge_id'] == 'deepseek']
assert measured['png_count'] == 0 and measured['serialized_text_tokens_deepseek_v4'] < 360000
assert measured['body_output_cap'] == 32768
assert measured['request']['sha256'] == ref(prepared / 'request.http_body.json')['sha256']
assert measured['prepare']['sha256'] == ref(prepared / 'prepare.json')['sha256']
ledger_path = BASE / 'n5_execution_20260930_r2/budget_ledger.working.json'
ledger = read(ledger_path)
release = read(prepared / 'release.draft.json')
release.update(status='RELEASED_FOR_ONE_CALL', input_token_reservation=360000,
    reserved_usd=0.15, prices_usd_per_million={'cache_hit': 0.006, 'cache_miss': 0.3, 'output': 1.2},
    pricing_source='https://api-docs.deepseek.com/quick_start/pricing/',
    budget={'ledger_file': str(ledger_path), 'ledger_sha256': ref(ledger_path)['sha256'],
            'budget_id': ledger['budget_id'], 'limit_usd': 15},
    development_only=True, semantic_score=None,
    input_gates=[ref(p) for p in (ATTEMPT / 'receipt.json', ATTEMPT / 'text_capacity.json',
        arithmetic_path, binding_path, CONTROL / 'PROBE_REGISTRATION_ZH.md', Path(__file__))])
release_path = CONTROL / 'episode-02.release.json'
runner.save(release_path, release, exclusive=True)
print(json.dumps({'status': 'DISPATCHING_ONE_DEVELOPMENT_CALL', 'packet': 'episode-02',
    'registered_reserve_usd': 0.15, 'ledger_used_before_usd': sum(e['charged_usd'] for e in ledger['entries'])}), flush=True)
receipt = runner.run(prepared, release_path, Path('/data/yuyangliu/secrets/deepseek.env'))
print(json.dumps({k: receipt.get(k) for k in ('status', 'complete', 'incomplete_reasons',
    'usage_cost_at_registered_rates_usd', 'budget_used_after_usd')}), flush=True)
raise SystemExit(0 if receipt['complete'] else 1)
