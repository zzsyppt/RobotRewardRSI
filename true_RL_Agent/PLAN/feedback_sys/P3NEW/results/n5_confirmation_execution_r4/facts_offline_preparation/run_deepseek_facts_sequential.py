"""Eight frozen R4 DeepSeek facts slots; default preflight, never seal/report."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

B = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
C = B / 'n5_execution_20261001_r4'
CAP = B / 'n5_confirmation_20261001_r4'
S = B / 'n5_confirmation_source_r4_final'
RUNNER = S / 'PLAN/feedback_sys/P3NEW/run_confirmation_review.py'
CARD = '811d7f140a5b3015f66a5093ba516c60bdc4a05c9ac827461d19bb05b8be3041'
SOURCE = '1b7cb40e5b1d1780011e1748a9e6adae88517dd880b74c8b0b5f23df66a1a645'
PROPOSAL = 'f1c780dc86006fe4f0c022b9bc8ae1bbdd8a80ce2b8f343540cca184f232f39e'
EXPORT = 'ad03463e684ca0a1d92f047a485e6e25b0e914c2f7f0a84c85273bc5edaa82b2'
RUNNER_SHA = '1aa1c6c6dce603d06d2e6cf2c7eeeb9f31dad8e65148781d86d10d0bfa4f574f'
DELIVERY = 'ffe1e077b43bbd5e7213ae95135fb7a8ab77c08b11cac590cd15a3c3e6c462f8'
MODULES = {
    'stair': '179701aa9a6e5eefff528eb3521d1e41d411df250de1e644ae1c9497237f6459',
    'push': 'f804849ab5d2670edf23bd199619e15b0c99a05f1b478cb1fc6d0e68c3edc736',
}
SLOTS = ('episode-01', 'episode-13', 'focus-01', 'focus-07',
         'media-01', 'media-02', 'media-03', 'media-04')
STOP = C / 'STOP_FACTS_DISPATCH'
OUT = C / 'postcapture/deepseek_facts_dispatch'


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def ref(path):
    path = Path(path)
    data = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(),
            'size_bytes': len(data)}


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def verify(item):
    path = item.get('path', item.get('file'))
    actual = ref(path)
    require(actual['sha256'] == item['sha256'], 'SHA mismatch: ' + str(path))
    if 'size_bytes' in item:
        require(actual['size_bytes'] == item['size_bytes'], 'size mismatch: ' + str(path))
    return actual


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def preflight(proposal_path):
    require(ref(proposal_path)['sha256'] == PROPOSAL, 'proposal changed')
    proposal = load(proposal_path)
    export_path = C / 'facts_offline_preparation/export_receipt.json'
    require(ref(export_path)['sha256'] == EXPORT, 'export changed')
    exported = load(export_path)
    require(proposal['export_receipt_sha256'] == EXPORT and
            proposal['capture_card_sha256'] == CARD, 'proposal identity')
    for path, sha in ((C / 'capture_card.json', CARD),
                      (C / 'runtime_source_manifest.json', SOURCE), (RUNNER, RUNNER_SHA)):
        require(ref(path)['sha256'] == sha, 'frozen identity: ' + str(path))
    card = load(C / 'capture_card.json')
    delivery_path = C / 'postcapture/delivery_original_r1/execution_receipt.json'
    require(ref(delivery_path)['sha256'] == DELIVERY, 'delivery changed')
    delivery = load(delivery_path)
    require(delivery['execution_status'] == 'COMPLETED' and
            delivery['status'] == 'offline_serialized_not_sent' and
            delivery['returncode'] == 0, 'delivery not completed offline')
    bound = {x['sha256'] for x in delivery['actual_module_and_trace_inputs']}
    for task, sha in MODULES.items():
        module_path = CAP / 'observers' / task / 'root_module_review_receipt.json'
        require(ref(module_path)['sha256'] == sha and sha in bound,
                task + ' module seal/delivery binding')
    for mode in ('numeric', 'media', 'validate', 'audit'):
        receipt = load(C / 'postcapture' / mode / 'receipt.json')
        require(receipt['status'] == 'COMPLETED_CPU_ONLY_NOT_RELEASED' and
                verify(receipt['capture_card'])['sha256'] == CARD and
                verify(receipt['source_manifest'])['sha256'] == SOURCE,
                mode + ' CPU identity/status')
    records = {x['packet_id']: x for x in exported['records'] if x['judge_id'] == 'deepseek'}
    rows = proposal['rows']
    require(tuple(x['packet_id'] for x in rows) == SLOTS and len(records) == 8,
            'fixed eight-slot roster changed')
    for row in rows:
        packet = row['packet_id']
        require(row['judge_id'] == 'deepseek' and row['output_token_reservation'] == 32768,
                packet + ' proposal role/output')
        require(isinstance(row['suggested_input_token_reservation'], int) and
                row['suggested_input_token_reservation'] > 0, packet + ' input reservation')
        for key, export_key in (('prepared', 'remote_prepare'), ('request', 'remote_request')):
            actual = verify(row[key])
            require(actual == records[packet][export_key], packet + ' export ' + key)
        prepared_dir = Path(row['prepared']['path']).parent
        prepared = load(prepared_dir / 'prepare.json')
        draft = load(prepared_dir / 'release.draft.json')
        require(prepared['stage'] == 'facts' and prepared['judge_id'] == 'deepseek' and
                prepared['packet_id'] == packet and prepared['status'] == 'PREPARED_NOT_RELEASED'
                and prepared['max_output_tokens'] == 32768 and
                not prepared.get('capacity_facts_fixture') and not prepared.get('own_facts_seal'),
                packet + ' prepared facts identity')
        require(draft['prepared_sha256'] == row['prepared']['sha256'] and
                draft['request_sha256'] == row['request']['sha256'] and
                draft['manifest_sha256'] == verify(records[packet]['remote_manifest'])['sha256'] and
                draft['runner_sha256'] == RUNNER_SHA and draft['automatic_retries'] == 0,
                packet + ' release draft identity')
    return card, rows


def dispatch(card, rows, result):
    ledger_path = C / 'budget_ledger.working.json'
    budget = card['budget']
    for row in rows:
        if STOP.exists():
            result['status'] = 'STOPPED_BEFORE_NEW_DISPATCH'
            return
        packet = row['packet_id']
        directory = Path(row['prepared']['path']).parent
        require(not any((directory / name).exists() for name in
                        ('receipt.json', 'raw_response.json', 'raw_answer.txt', 'raw_reasoning.txt')),
                packet + ' already has invocation artifacts; no retry')
        ledger_ref = ref(ledger_path)
        ledger = load(ledger_path)
        release = copy.deepcopy(load(directory / 'release.draft.json'))
        release.update(status='RELEASED_FOR_ONE_CALL',
                       input_token_reservation=row['suggested_input_token_reservation'],
                       reserved_usd=budget['reserved_usd_per_review'],
                       prices_usd_per_million=budget['prices_usd_per_million'],
                       pricing_source=budget['pricing_source'])
        release['budget'] = {'ledger_file': str(ledger_path), 'ledger_sha256': ledger_ref['sha256'],
                             'budget_id': budget['budget_id'], 'limit_usd': budget['api_phase_limit_usd']}
        require(ledger['budget_id'] == budget['budget_id'], 'ledger budget identity')
        release_path = OUT / 'releases' / (packet + '.release.json')
        write_new(release_path, release)
        if STOP.exists():
            result['status'] = 'STOPPED_BEFORE_NEW_DISPATCH'
            return
        argv = [sys.executable, '-B', str(RUNNER), 'run', '--prepared', str(directory),
                '--release', str(release_path), '--secret', '/data/yuyangliu/secrets/deepseek.env']
        record = {'packet_id': packet, 'started_at_utc': now(), 'argv': argv,
                  'release': ref(release_path), 'ledger_before': ledger_ref}
        result['dispatches'].append(record)
        write_new(OUT / (packet + '.started.json'), record)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', PYTHONPATH=os.pathsep.join(
                       map(str, (S / 'src', S, RUNNER.parent))))
        with (OUT / (packet + '.stdout.log')).open('xb') as stdout, \
                (OUT / (packet + '.stderr.log')).open('xb') as stderr:
            code = subprocess.run(argv, cwd=S, env=env, stdin=subprocess.DEVNULL,
                                  stdout=stdout, stderr=stderr).returncode
        record.update(returncode=code, finished_at_utc=now())
        receipt_path = directory / 'receipt.json'
        receipt = load(receipt_path) if receipt_path.exists() else {}
        record['runner_receipt'] = ref(receipt_path) if receipt_path.exists() else None
        record['complete'] = receipt.get('complete') is True
        write_new(OUT / (packet + '.exit.json'), record)
        require(code == 0 and record['complete'], packet + ' runner incomplete; stop without retry')
    result['status'] = 'COMPLETED_FACTS_CALLS_NOT_SEALED_NO_REPORTS'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal', type=Path, default=Path(__file__).with_name('deepseek_reservation_proposal.json'))
    parser.add_argument('--execute-facts', action='store_true')
    args = parser.parse_args()
    card, rows = preflight(args.proposal)
    result = {'status': 'PREFLIGHT_ONLY_NOT_SENT', 'started_at_utc': now(),
              'proposal': ref(args.proposal), 'slots': list(SLOTS), 'dispatches': [],
              'facts_sealed': 0, 'reports_created': 0, 'automatic_retries': 0,
              'all_80_facts_seal_gate_preserved': True}
    if not args.execute_facts:
        print(json.dumps(result, ensure_ascii=False))
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    # The persistent exclusive marker prohibits a second scheduler or automatic resume.
    write_new(OUT / 'single_writer.json', {'pid': os.getpid(), 'started_at_utc': now()})
    try:
        dispatch(card, rows, result)
    except Exception as exc:
        result.update(status='FAILED_STOPPED_NO_RETRY', error=str(exc))
        if not STOP.exists():
            try:
                write_new(STOP, {'carrier': 'deepseek', 'recorded_at_utc': now(), 'reason': str(exc)})
            except FileExistsError:
                pass
    finally:
        result['finished_at_utc'] = now()
        write_new(OUT / 'execution_receipt.json', result)
    print(json.dumps({'status': result['status'], 'completed_slots': sum(
        x.get('complete') is True for x in result['dispatches']), 'receipt': str(OUT / 'execution_receipt.json')}))
    return 0 if result['status'] == 'COMPLETED_FACTS_CALLS_NOT_SEALED_NO_REPORTS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
