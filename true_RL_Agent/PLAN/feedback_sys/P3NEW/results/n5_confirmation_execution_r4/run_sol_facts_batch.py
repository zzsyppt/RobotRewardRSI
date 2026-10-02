"""R4 facts-only 72-slot controller: at most two CLI calls, no retries/seals/reports."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

import prepare_facts_offline as prep

E = Path(__file__).resolve().parent
F = E / 'facts_offline_preparation'
RUN = E / 'sol_facts_batch_r1'
SOURCE = prep.LOCAL_SOURCE
P = SOURCE / 'PLAN/feedback_sys/P3NEW'
PINS = {
    prep.CLI: 'cbafb6422bca005b94c12d105b1a16a0474219e24ea4893f85846409c464f5a1',
    E / 'prepare_facts_offline.py': '9f47b226b8b1da6f5aceb6ed8d7e02fe74610b58b8a7ac17d136deb693ec0424',
    F / 'pending_sol_launch_commands.json': '2411827c8340145763629412e10bf7d77bb7ccb78cfa2c4c93e9f6b140c0128b',
    F / 'local_preparation_receipt.json': '8f34ec0d5935efdf321b6b33b3b04bdfe05cfe78fc083414db5db7492ccbe699',
    F / 'export_receipt.json': 'ad03463e684ca0a1d92f047a485e6e25b0e914c2f7f0a84c85273bc5edaa82b2',
    F / 'capacity_summary.json': 'c0fc2395612aa32c20b398d1a73bc59fa230e43a840edfc960e389a31d279a9a',
    E.parent / 'n5_confirmation_source_r4/source_freeze_receipt.json': '881cbf9babfaf4732694d4d27b4db881c0bddce4c26a796968585a05f3411501',
}


def now(): return datetime.now(timezone.utc).isoformat()
def option(argv, name): return Path(argv[argv.index(name) + 1])
def check(reference, path=None):
    path = Path(path or reference.get('path') or reference['file'])
    actual = prep.ref(path)
    prep.require(actual['sha256'] == reference['sha256'], 'Bound file changed: ' + str(path))
    return actual


def prerequisites(execution_path, assembly_path):
    prep.require(not RUN.exists(), 'This fixed batch already has an attempt; never restart it')
    for path, expected in PINS.items():
        prep.require(prep.ref(path)['sha256'] == expected, 'Preparation changed: ' + str(path))
    frozen = prep.verify_source(SOURCE, E / 'runtime_source_manifest.json')
    prep.require(prep.ref(E / 'capture_card.json')['sha256'] == prep.CARD_SHA, 'Local capture card changed')
    cpu = prep.read(E.parent / 'n5_confirmation_source_r4/source_freeze_receipt.json')
    prep.require(cpu['preflight_exit_codes'] == {'stair': 0, 'push': 0}, 'CPU preflight incomplete')
    check(cpu['verification'])
    capacity = prep.read(F / 'capacity_summary.json')
    prep.require(capacity['all_72_text_proxy_checks_pass'] is True, 'Frozen text proxy gate failed')
    execution, assembly = prep.read(execution_path), prep.read(assembly_path)
    check(execution['assembly_receipt'], assembly_path)
    prep.require(execution['scope'] == 'n5_confirmation' and execution['execution_status'] == 'COMPLETED' and
                 execution['status'] == assembly['status'] == 'offline_serialized_not_sent' and
                 execution['capture_card']['sha256'] == prep.CARD_SHA and
                 execution['source_manifest']['sha256'] == prep.SOURCE_SHA and
                 assembly['model_calls'] == assembly['new_author_calls'] == assembly['gpu_calls'] == 0,
                 'R4 offline assembly is not complete/bound')
    gates = [prep.ref(execution_path), prep.ref(assembly_path)]
    for task in ('stair', 'push'):
        root = E / (task + '_observer_inputs')
        seal_path = root / 'root_module_review_receipt.json'
        seal = prep.read(seal_path)
        gates.extend([prep.ref(seal_path), check(seal['review'], root / 'root_module_review.json'),
                      check(seal['combined_audit'], root / 'combined_audit.json')])
        row = assembly['tasks'][task]
        check(row['as_of_cutoff_ref'], seal_path)
        prep.require(row['as_of_cutoff'] == seal['recorded_at_utc'] == execution['tasks'][task]['as_of_cutoff'] and
                     row['nonvisual_unchanged'] is True, 'Actual module seal/assembly cutoff differs: ' + task)
    local, exported = prep.read(F / 'local_preparation_receipt.json'), prep.read(F / 'export_receipt.json')
    contexts = {(r['packet_id'], r['judge_id']): r for r in local['contexts']}
    records = {(r['packet_id'], r['judge_id']): r for r in exported['records']}
    calls = prep.read(F / 'pending_sol_launch_commands.json')['calls']
    prep.require(len(calls) == len(contexts) == 72 and
                 {(r['packet_id'], r['judge_id']) for r in calls} == set(contexts), 'Fixed 72 slots differ')
    bound_source = [prep.ref(P / name) for name in ('run_sol_review.py', 'prepare_desktop_review.py',
                    'run_confirmation_review.py', 'confirmation_review_coverage.py',
                    'N5_SOL_CLI_POLICY_R4.json', 'N5_REVIEW_ASSIGNMENT_R4.json')]
    for row in calls:
        key = row['packet_id'], row['judge_id']
        context, original, argv = contexts[key], records[key], row['argv']
        prep.require(argv[-1] == '--execute-one-call' and Path(argv[2]) == P / 'run_sol_review.py', 'Registered launch changed')
        prepared, output, runtime = (option(argv, flag) for flag in ('--prepared', '--output', '--runtime-home'))
        prep.require(not output.exists() and not runtime.exists(), 'Slot output/runtime already exists: ' + str(key))
        check(original['remote_prepare'], prepared / 'prepare.json')
        check(original['remote_request'], prepared / 'request.http_body.json')
        transport_ref = check(context['transport_receipt'])
        transport = prep.read(Path(transport_ref['path']))
        check(context['stdin'])
        for item in transport['child_read_whitelist']: check(item)
        card = prep.read(prepared / 'prepare.json')
        prep.require(card['stage'] == 'facts' and card['judge_id'] == key[1] and card['packet_id'] == key[0] and
                     card['max_output_tokens'] == 32768 and card['own_facts_seal'] is None and
                     card['capacity_facts_fixture'] is None, 'Not the original facts slot')
    return calls, contexts, records, bound_source, gates


def finish(active, review):
    process, row, slot, stdout, stderr = active
    code = process.wait()
    stdout.close()
    stderr.close()
    output = option(row['argv'], '--output')
    result = {'packet_id': row['packet_id'], 'judge_id': row['judge_id'], 'returned_at_utc': now(),
              'pid': process.pid, 'returncode': code, 'status': 'FAILED_STOP_NEW_DISPATCH',
              'scope': 'facts_only', 'answer_rewritten': False, 'facts_sealed': False,
              'stdout': prep.ref(slot / 'stdout.log'), 'stderr': prep.ref(slot / 'stderr.log')}
    try:
        receipt_path, answer_path = output / 'receipt.json', output / 'raw_answer.json'
        result['original_driver_receipt'] = prep.ref(receipt_path)
        receipt = prep.read(receipt_path)
        result['raw_answer'] = prep.ref(answer_path)
        prep.require(code == 0 and receipt['status'] == 'COMPLETED_TRANSPORT_VERIFIED_NOT_RELEASED' and
                     receipt['exit_code'] == 0, 'Frozen CLI transport did not complete')
        check(receipt['raw_answer'], answer_path)
        check(receipt['access_trace'])
        prepared = option(row['argv'], '--prepared')
        expected = prep.read(prepared / 'prepare.json')
        actual_transport = prep.read(output / 'transport_receipt.json')
        prep.require(actual_transport['source_request']['sha256'] == expected['request_sha256'] and
                     prep.ref(output / 'stdin.txt')['sha256'] ==
                     prep.ref(F / 'transport' / row['packet_id'] / row['judge_id'] / 'stdin.txt')['sha256'],
                     'Executed original input differs')
        answer, request = prep.read(answer_path), prep.read(prepared / 'request.http_body.json')
        audit = review.coverage.audit_answer(answer, request, stage='facts')
        audit.update(request=prep.ref(prepared / 'request.http_body.json'), answer=result['raw_answer'],
                     coverage_helper=prep.ref(Path(review.coverage.__file__)), scope='mechanical_facts_only_not_semantic')
        prep.save(slot / 'coverage_audit.json', audit)
        result['coverage_audit'] = prep.ref(slot / 'coverage_audit.json')
        prep.require(review.envelope(answer, row['judge_id'], row['packet_id'], 'facts') and
                     answer['complete'] is True and audit['mechanical_valid'] and audit['review_complete'],
                     'Facts envelope/completeness/reference coverage failed')
        check(result['raw_answer'], answer_path)
        result['status'] = 'COMPLETED_TRANSPORT_AND_MECHANICAL_ONLY_NOT_SEALED'
    except Exception as error:
        result['error'] = {'type': type(error).__name__, 'message': str(error)}
    prep.save(slot / 'return.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assembly-execution-receipt', type=Path, required=True)
    parser.add_argument('--assembly-receipt', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='Explicitly start the registered model calls after all gates')
    args = parser.parse_args()
    prep.require(args.execute, 'Preparation only; --execute must be supplied by the operator')
    calls, contexts, records, bound_source, gates = prerequisites(args.assembly_execution_receipt, args.assembly_receipt)
    _, review, _ = prep.modules(SOURCE)
    RUN.mkdir(exist_ok=False)  # Fixed exclusive attempt also prevents a second controller.
    prep.save(RUN / 'start.json', {'scope': 'facts_only', 'started_at_utc': now(), 'script': prep.ref(Path(__file__)),
        'commands': prep.ref(F / 'pending_sol_launch_commands.json'), 'prerequisite_receipts': gates,
        'preparation_pins': [{'path': str(p), 'sha256': h} for p, h in PINS.items()],
        'source': bound_source, 'slots': 72, 'max_parallel': 2, 'application_retries': 0,
        'hidden_transport_retries': 'Unobserved; frozen CLI defaults unchanged',
        'capacity_scope': 'Frozen text proxy passes; actual SOL/image tokens and runtime peak are checked only from returned evidence',
        'facts_seals_created': 0, 'reports_created': 0, 'semantic_admission_claimed': False})
    active, results, next_slot, stop = [], [], 0, None
    try:
        while active or (stop is None and next_slot < len(calls)):
            for item in list(active):
                if item[0].poll() is not None:
                    result = finish(item, review)
                    results.append(result)
                    active.remove(item)
                    if result['status'] != 'COMPLETED_TRANSPORT_AND_MECHANICAL_ONLY_NOT_SEALED':
                        stop = stop or {'reason': 'returned_call_failed', 'packet_id': result['packet_id'], 'judge_id': result['judge_id']}
            while stop is None and len(active) < 2 and next_slot < len(calls):
                if (E / 'STOP_FACTS_DISPATCH').exists():
                    stop = {'reason': 'operator_stop_marker', 'path': str(E / 'STOP_FACTS_DISPATCH'), 'observed_at_utc': now()}
                    break
                if any(item[0].poll() is not None for item in active): break
                row = calls[next_slot]
                key = row['packet_id'], row['judge_id']
                for reference in bound_source: check(reference)
                prepared = option(row['argv'], '--prepared')
                check(records[key]['remote_request'], prepared / 'request.http_body.json')
                check(records[key]['remote_prepare'], prepared / 'prepare.json')
                prep.require(not option(row['argv'], '--output').exists() and not option(row['argv'], '--runtime-home').exists(), 'Slot is no longer fresh')
                slot = RUN / row['packet_id'] / row['judge_id']
                slot.mkdir(parents=True, exist_ok=False)
                prep.save(slot / 'dispatch.json', {'scope': 'facts_only', 'dispatched_at_utc': now(),
                    'packet_id': key[0], 'judge_id': key[1], 'argv': row['argv'],
                    'original_prepare': records[key]['remote_prepare'], 'original_request': records[key]['remote_request'],
                    'prepared_transport': contexts[key]['transport_receipt'], 'attempt': 1})
                next_slot += 1  # The slot is spent even if process creation fails.
                stdout, stderr = (slot / 'stdout.log').open('xb'), (slot / 'stderr.log').open('xb')
                try:
                    process = subprocess.Popen(row['argv'], stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                except BaseException:
                    stdout.close()
                    stderr.close()
                    raise
                active.append((process, row, slot, stdout, stderr))
                prep.save(slot / 'process.json', {'pid': process.pid, 'registered_at_utc': now()})
            if active: time.sleep(1)
    except BaseException as error:
        stop = stop or {'reason': 'controller_exception', 'type': type(error).__name__, 'message': str(error)}
    # No terminate/cancel/retry: already dispatched calls always finish and retain evidence.
    for item in active:
        results.append(finish(item, review))
    complete = len(results) == 72 and all(r['status'] == 'COMPLETED_TRANSPORT_AND_MECHANICAL_ONLY_NOT_SEALED' for r in results)
    prep.save(RUN / 'final.json', {'scope': 'facts_only', 'finished_at_utc': now(),
        'status': 'COMPLETED_72_FACTS_MECHANICAL_ONLY_NOT_SEALED' if complete else 'STOPPED_NO_NEW_DISPATCH',
        'stop_reason': stop, 'attempted_slots': next_slot, 'returned_calls': len(results),
        'unstarted_slots': [{'packet_id': r['packet_id'], 'judge_id': r['judge_id']} for r in calls[next_slot:]],
        'results': results, 'application_retries': 0, 'cancellations': 0, 'facts_seals_created': 0,
        'reports_created': 0, 'semantic_admission_claimed': False,
        'limit': 'Mechanical references and declared complete are not semantic completeness. A hard OS/process crash requires manual reconciliation; never rerun this fixed batch.'})
    return 0 if complete else 1


if __name__ == '__main__':
    raise SystemExit(main())
