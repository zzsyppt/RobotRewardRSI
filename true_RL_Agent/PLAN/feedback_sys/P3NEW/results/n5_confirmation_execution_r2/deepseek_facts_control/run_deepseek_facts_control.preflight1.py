"""R2 fixed forty facts: read-only preflight by default; --execute only on root start.

This is control glue around the frozen one-call runner. It never edits materials,
creates facts seals, reads review votes, executes report, retries, or reclaims locks.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

B = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
S, C = B / 'n5_confirmation_source_r2', B / 'n5_execution_20260930_r2'
CAP = B / 'n5_confirmation_20260930_r2'
P = S / 'PLAN/feedback_sys/P3NEW'
PY = Path('/data/yuyangliu/envs/isaaclab6_validation/bin/python')
RUNNER = P / 'run_confirmation_review.py'
LEDGER = C / 'budget_ledger.working.json'
SECRET = Path('/data/yuyangliu/secrets/deepseek.env')
OUT = C / 'control/deepseek_facts_r2'
CARD_SHA = '6bea98ed63e1f1e516631e58205e3da93406311d1b87466e445fe754b0c22794'
SOURCE_SHA = '497d14b6a133d1ba3afa97e15c39b00515479692e528aac648970851245f3b76'
RUNNER_SHA = 'd20765ea0e3c96b2070a63e7c5c79adfe28b402f964914e58faffbe24e97114e'
VISION_DOC_SHA = '5654a198302edd80aea443fb123d14b14d675ccc76730959dd55414d9baee1e2'
VISION_DOC = C / 'deepseek_facts_control/vision_docs.20260930.html'
IMAGE_TOKEN_BOUND = 1024  # Official Vision page, independently counted per image.


def now():
    return datetime.now(timezone.utc).isoformat()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def ref(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return {'file': str(path), 'sha256': h.hexdigest(), 'size_bytes': path.stat().st_size}


def save_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def fixed_packets():
    numeric = CAP / 'numeric_materials/prepared'
    rows = [(f'episode-{n:02d}', numeric / f'episode-{n:02d}/deepseek') for n in range(1, 25)]
    rows += [(f'focus-{n:02d}', numeric / f'focus-{n:02d}/deepseek') for n in range(1, 13)]
    rows += [(f'media-{n:02d}', CAP / f'media_materials_control_corrected/media-{n:02d}/prepared_facts/deepseek')
             for n in range(1, 5)]
    return rows


def preflight():
    require(ref(C / 'capture_card.json')['sha256'] == CARD_SHA, 'capture card drift')
    require(ref(C / 'runtime_source_manifest.json')['sha256'] == SOURCE_SHA, 'source manifest drift')
    require(ref(RUNNER)['sha256'] == RUNNER_SHA, 'frozen runner drift')
    require(ref(VISION_DOC)['sha256'] == VISION_DOC_SHA, 'official vision evidence drift')
    card = read(C / 'capture_card.json')
    require(card['status'] == 'FROZEN' and card['output_root'] == str(CAP), 'wrong frozen round')
    require(card['budget']['reserved_usd_per_review'] == .15
            and card['budget']['api_phase_limit_usd'] == 15
            and card['budget']['prices_usd_per_million'] == {'cache_hit': .006, 'cache_miss': .3, 'output': 1.2},
            'budget contract drift')
    require(card['materials']['input_token_proxy_limit'] == 360000
            and card['materials']['output_reserved'] == 32768, 'capacity contract drift')
    require(SECRET.is_file(), 'specified secret file absent')  # Never read it in control code.
    require(not Path(str(LEDGER) + '.lock').exists(), 'ledger lock requires manual reconciliation')
    ledger = read(LEDGER)
    require(ledger['schema_version'] == 'p3new.review_budget.v1' and ledger['limit_usd'] == 15,
            'unexpected ledger contract')
    require(not any(e.get('stage') in ('facts', 'report') for e in ledger['entries']),
            'formal attempt already present; no resume or retry')
    require(abs(sum(e['charged_usd'] for e in ledger['entries']) - .468395796) < 1e-12,
            'historical ledger differs; reconcile before this first facts batch')
    require(not OUT.exists(), 'batch output already exists; no resume or retry')
    counts, count_refs = {}, []
    for name in ('numeric', 'media_control_corrected'):
        path = C / f'postcapture/{name}/prepared_text_counts.json'
        count_refs.append(ref(path))
        for row in read(path)['records']:
            if row['judge_id'] == 'deepseek':
                key = row['packet_id']
                require(key not in counts, 'duplicate static count')
                counts[key] = row
    require(set(counts) == {p for p, _ in fixed_packets()}, 'static counts do not cover exactly forty facts')
    rows = []
    for packet, prepared in fixed_packets():
        meta, count = read(prepared / 'prepare.json'), counts[packet]
        prepared_ref, request_ref = ref(prepared / 'prepare.json'), ref(prepared / 'request.http_body.json')
        require(meta['stage'] == 'facts' and meta['packet_id'] == packet and meta['judge_id'] == 'deepseek'
                and meta['status'] == 'PREPARED_NOT_RELEASED' and meta['max_output_tokens'] == 32768
                and meta['own_facts_seal'] is None and meta.get('capacity_facts_fixture') is None,
                packet + ': incorrect prepared identity or stage')
        require(prepared_ref['sha256'] == count['prepared']['sha256']
                and request_ref['sha256'] == count['request']['sha256'] == meta['request_sha256'],
                packet + ': static count/request drift')
        require(meta['runner']['sha256'] == RUNNER_SHA and ref(meta['transport']['file'])['sha256'] == meta['transport']['sha256'],
                packet + ': runner or transport drift')
        require(ref(meta['manifest']['file'])['sha256'] == meta['manifest']['sha256'], packet + ': manifest drift')
        require(not any((prepared / n).exists() for n in
                        ('receipt.json', 'raw_response.json', 'raw_answer.txt', 'raw_reasoning.txt')),
                packet + ': previous attempt artifacts; no retry')
        require(count['image_count'] == meta['png_count'] and count['images_with_unknown_bytes'] == 0,
                packet + ': image inventory mismatch')
        require(meta['png_count'] == (12 if packet.startswith('media-') else 0), packet + ': image count differs')
        require(request_ref['size_bytes'] <= 48 * 1024 * 1024, packet + ': HTTP capacity exceeded')
        for material in meta['materials']:
            require(ref(material['file'])['sha256'] == material['sha256'], packet + ': material drift')
        text_tokens = count['serialized_text_tokens_deepseek_v4']
        image_tokens = meta['png_count'] * IMAGE_TOKEN_BOUND
        estimated_input = text_tokens + image_tokens
        reserved_total = estimated_input + 32768 + 32768 + 8192
        require(estimated_input <= 360000 and reserved_total < 450000, packet + ': static capacity exceeded')
        rows.append({'packet_id': packet, 'prepared_directory': str(prepared), 'prepared': prepared_ref,
                     'request': request_ref, 'text_tokens_proxy': text_tokens, 'image_tokens_upper_bound': image_tokens,
                     'estimated_input_before_chat_overhead': estimated_input, 'reserved_total_proxy': reserved_total,
                     'input_token_reservation': 360000, 'reserved_usd': .15})
    module_receipts, missing = {}, []
    for task in ('stair', 'push'):
        path = CAP / f'observers/{task}/root_module_review_receipt.json'
        if not path.is_file():
            missing.append(task + ':module_review_receipt_missing')
            continue
        # Select only provenance/time/barrier metadata. Never load the referenced review or answers.
        data = read(path)
        timestamp = datetime.fromisoformat(data['recorded_at_utc'].replace('Z', '+00:00'))
        require(timestamp.tzinfo is not None and timestamp <= datetime.now(timezone.utc), 'invalid module seal time')
        require(data['formal_facts_calls_before_seal'] == 0 and data['formal_report_calls_before_seal'] == 0
                and data['original_raw_answers_preserved'] is True, task + ': module barrier metadata invalid')
        review_ref = ref(Path(data['review']))
        require(review_ref['sha256'] == data['review_sha256'], task + ': sealed review file drift')
        module_receipts[task] = {'receipt': ref(path), 'recorded_at_utc': data['recorded_at_utc'],
                                 'review_ref_hash_only': review_ref}
    worst_cost = (360000 * .3 + 32768 * 1.2) / 1e6
    require(worst_cost <= .15 and sum(e['charged_usd'] for e in ledger['entries']) + 80 * .15 <= 15,
            'whole registered reservation exceeds budget')
    return {'status': 'READY_PENDING_ROOT_START' if not missing else 'BLOCKED_PRECALL', 'created_at_utc': now(),
            'missing_gates': missing, 'stage': 'facts', 'judge_id': 'deepseek', 'packet_count': 40,
            'capture_card': ref(C / 'capture_card.json'), 'source_manifest': ref(C / 'runtime_source_manifest.json'),
            'runner': ref(RUNNER), 'controller': ref(Path(__file__)), 'budget_before': ref(LEDGER),
            'static_count_receipts': count_refs, 'module_receipts': module_receipts, 'packets': rows,
            'vision_bound_source': {'url': 'https://api-docs.deepseek.com/guides/vision/', 'snapshot': ref(VISION_DOC),
                                    'per_image_upper_bound': IMAGE_TOKEN_BOUND},
            'max_estimated_input': max(x['estimated_input_before_chat_overhead'] for x in rows),
            'max_reserved_total_proxy': max(x['reserved_total_proxy'] for x in rows),
            'capacity_note': 'Proxy plus documented image bound; chat/internal variation remains covered by reservation and actual usage checks.',
            'calls_by_preflight': 0, 'facts_seals_created': 0, 'reports_supported': False}


def execute(plan):
    require(not plan['missing_gates'], 'root module receipt barrier not complete')
    OUT.mkdir(parents=True, exist_ok=False)
    save_new(OUT / 'start.json', plan)
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', PYTHONPATH=os.pathsep.join(map(str, (S / 'src', S, P))))
    completed = 0
    expected_ledger_sha = plan['budget_before']['sha256']
    for row in plan['packets']:
        packet, prepared = row['packet_id'], Path(row['prepared_directory'])
        ledger_ref, ledger = ref(LEDGER), read(LEDGER)
        require(ledger_ref['sha256'] == expected_ledger_sha, 'ledger changed outside this single-writer batch')
        require(not Path(str(LEDGER) + '.lock').exists(), 'do not reclaim ledger lock')
        require(ref(prepared / 'prepare.json')['sha256'] == row['prepared']['sha256'], 'prepared drift since preflight')
        for item in plan['module_receipts'].values():
            require(ref(item['receipt']['file'])['sha256'] == item['receipt']['sha256'], 'module receipt drift')
        release = read(prepared / 'release.draft.json')
        release.update(status='RELEASED_FOR_ONE_CALL', input_token_reservation=row['input_token_reservation'],
                       reserved_usd=.15, prices_usd_per_million={'cache_hit': .006, 'cache_miss': .3, 'output': 1.2},
                       pricing_source='https://api-docs.deepseek.com/quick_start/pricing/',
                       budget={'ledger_file': str(LEDGER), 'ledger_sha256': ledger_ref['sha256'],
                               'budget_id': ledger['budget_id'], 'limit_usd': 15})
        release_path = OUT / f'{packet}.release.json'
        save_new(release_path, release)
        argv = [str(PY), '-B', str(RUNNER), 'run', '--prepared', str(prepared),
                '--release', str(release_path), '--secret', str(SECRET)]
        save_new(OUT / f'{packet}.dispatch.json', {'started_utc': now(), 'argv': argv,
                 'release': ref(release_path), 'budget_before': ledger_ref,
                 'static_capacity': {k: row[k] for k in ('text_tokens_proxy', 'image_tokens_upper_bound',
                    'estimated_input_before_chat_overhead', 'reserved_total_proxy', 'input_token_reservation')}})
        with (OUT / f'{packet}.stdout.log').open('xb') as stdout, (OUT / f'{packet}.stderr.log').open('xb') as stderr:
            result = subprocess.run(argv, cwd=S, env=env, stdout=stdout, stderr=stderr, check=False)
        receipt_path = prepared / 'receipt.json'
        receipt = read(receipt_path) if receipt_path.is_file() else {}
        complete = result.returncode == 0 and receipt.get('complete') is True and receipt.get('status') == 'received'
        record = {'finished_utc': now(), 'packet_id': packet, 'returncode': result.returncode, 'complete': complete,
                  'receipt': ref(receipt_path) if receipt else None, 'budget_after': ref(LEDGER),
                  'incomplete_reasons': receipt.get('incomplete_reasons', []), 'status': receipt.get('status', 'NO_RECEIPT_PRECALL_FAILURE'),
                  'exception_type': receipt.get('exception_type'), 'automatic_retry': False}
        save_new(OUT / f'{packet}.exit.json', record)
        expected_ledger_sha = record['budget_after']['sha256']
        print(json.dumps({k: record[k] for k in ('packet_id', 'returncode', 'complete', 'status')}), flush=True)
        if not complete:
            save_new(OUT / 'stopped.json', {**record, 'completed_facts': completed,
                     'action': 'STOP_KEEP_ALL_ORIGINALS_NO_RETRY_NO_SEAL_NO_REPORT'})
            return 1
        completed += 1
    save_new(OUT / 'complete.json', {'finished_utc': now(), 'completed_facts': completed,
             'facts_seals_created': 0, 'report_calls': 0, 'scope': 'Mechanical runner completion only; no semantic qualification.'})
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--execute', action='store_true', help='Use only after root explicitly starts this fixed facts batch.')
    parser.add_argument('--preflight-output', type=Path, help='Optional new control JSON; never a release or a facts seal.')
    args = parser.parse_args()
    try:
        plan = preflight()
        if args.preflight_output:
            save_new(args.preflight_output, plan)
        print(json.dumps({k: plan[k] for k in ('status', 'packet_count', 'missing_gates', 'max_estimated_input', 'max_reserved_total_proxy')}), flush=True)
        if not args.execute:
            return 0 if not plan['missing_gates'] else 1
        return execute(plan)
    except Exception as error:
        record = {'status': 'CONTROL_STOP_NO_RETRY', 'error_type': type(error).__name__, 'message': str(error), 'utc': now()}
        if args.execute and OUT.is_dir() and not (OUT / 'stopped.json').exists():
            save_new(OUT / 'stopped.json', record)
        print(json.dumps(record), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
