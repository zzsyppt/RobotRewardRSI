"""Seal the single new-carrier calibration; compare all frozen items once."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIVE = Path('D:/humanoid_agent/artifacts/p3new_clean_calibration_20260930_gpt61')
ROLLOUT = Path('C:/Users/lover/.codex/sessions/2026/09/30/rollout-2026-09-30T21-48-06-01a0f292-9038-7f10-acbe-f578bb3795a5.jsonl')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(name, obj):
    data = (json.dumps(obj, ensure_ascii=False, indent=2) + '\n').encode()
    with (HERE / name).open('xb') as stream:
        stream.write(data)
    return digest(data)


def main():
    registration = json.loads((HERE / 'registration.json').read_text())
    for key in ('input', 'original_request', 'comparison_reference'):
        ref = registration[key]
        assert digest(Path(ref['file']).read_bytes()) == ref['sha256'], key
    originals = {}
    for name in ('answer.json', 'access_log.json'):
        data = (LIVE / name).read_bytes()
        with (HERE / name).open('xb') as stream:
            stream.write(data)
        originals[name] = digest(data)
    answer = json.loads((HERE / 'answer.json').read_text())
    reference = json.loads(Path(registration['comparison_reference']['file']).read_text())
    request = json.loads(Path(registration['original_request']['file']).read_text())
    expected = [x['id'] for x in json.loads(request['messages'][1]['content'])['items.json']]
    actual = {x['id']: x for x in answer['items']}
    ds = {x['id']: x for x in reference['items']}
    assert len(actual) == len(answer['items']) == len(expected) == 20
    assert set(actual) == set(ds) == set(expected)
    records = [json.loads(line) for line in ROLLOUT.read_text(encoding='utf-8').splitlines()]
    contexts = [r['payload'] for r in records if r['type'] == 'turn_context']
    assert all(x['model'] == 'gpt-6.1-sol' and x['effort'] == 'high' for x in contexts)
    calls = [{'timestamp': r.get('timestamp'), **r['payload']} for r in records
             if r['type'] == 'response_item'
             and r['payload'].get('type') in ('function_call', 'custom_tool_call')]
    trace_hash = save('observable_tool_trace.json', calls)
    usage = [r['payload'].get('info') for r in records if r['type'] == 'event_msg'
             and r['payload'].get('type') == 'token_count' and r['payload'].get('info')]
    receipt = {
        'status': 'ORIGINAL_ANSWER_SEALED_BEFORE_COMPARISON',
        'sealed_utc': datetime.now(timezone.utc).isoformat(),
        'model': 'gpt-6.1-sol', 'reasoning_effort': 'high', 'fork_turns': 'none',
        'platform_task_id': records[0]['payload']['id'],
        'input_sha256': registration['input']['sha256'], 'originals': originals,
        'observable_tool_trace_sha256': trace_hash, 'tool_call_count': len(calls),
        'access_audit': {
            'reviewer': 'root; nonblind carrier integrity audit, not a semantic regrade',
            'observed_reads': 'Only the allowlisted input, parsed with standard-library Python; no other evidence paths.',
            'other_judgments_or_memory_read': False, 'network_or_model_calls': False,
            'answer_writes': 1, 'answer_revised_after_save': False,
            'limits': 'Shared filesystem, not OS isolation. Truncated initial output followed by bounded rereads; no claim every scalar was simultaneously in context.'},
        'desktop_usage_last_record': usage[-1] if usage else None,
        'desktop_cost_usd': None, 'api_calls': 0, 'fingerprint': None,
    }
    seal_hash = save('carrier_receipt.json', receipt)
    rows = [{'id': ident, 'deepseek_kind': ds[ident]['kind'], 'gpt_kind': actual[ident]['kind'],
             'deepseek_verdict': ds[ident]['verdict'], 'gpt_verdict': actual[ident]['verdict'],
             'verdict_agrees': ds[ident]['verdict'] == actual[ident]['verdict'],
             'kind_agrees': ds[ident]['kind'] == actual[ident]['kind']} for ident in expected]
    n = len(rows)
    verdict = sum(r['verdict_agrees'] for r in rows)
    kind = sum(r['kind_agrees'] for r in rows)
    result = {
        'status': 'RAW_CALIBRATION_COMPARISON_NOT_ADJUDICATED',
        'compared_utc': datetime.now(timezone.utc).isoformat(), 'formal_confirmation': False,
        'originals_sealed_before_comparison': True, 'carrier_receipt_sha256': seal_hash,
        'gpt_sha256': originals['answer.json'],
        'deepseek_sha256': registration['comparison_reference']['sha256'],
        'n': n, 'verdict_agreement_count': verdict, 'kind_agreement_count': kind,
        'verdict_agreement': verdict/n, 'kind_agreement': kind/n,
        'verdict_disagreement_ids': [r['id'] for r in rows if not r['verdict_agrees']],
        'kind_disagreement_ids': [r['id'] for r in rows if not r['kind_agrees']],
        'minimum_verdict_agreement': .90, 'verdict_agreement_gate_passed': verdict/n >= .90,
        'missing_response_counts_as_disagreement': True,
        'scope': 'One new-carrier calibration using all 20 unchanged old-development items. Agreement is not accuracy; no new formal N5 outcome.',
        'new_deepseek_calls': 0, 'desktop_cost_usd': None, 'rows': rows,
    }
    save('raw_comparison.json', result)
    print(json.dumps({k: result[k] for k in ('n', 'verdict_agreement', 'kind_agreement', 'verdict_agreement_gate_passed')}))


if __name__ == '__main__':
    main()
