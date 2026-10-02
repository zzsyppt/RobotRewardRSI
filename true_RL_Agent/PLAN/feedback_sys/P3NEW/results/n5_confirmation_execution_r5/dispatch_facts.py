"""R5 fixed facts-only dispatch. Default preflight; explicit --execute-facts only."""
import argparse
import copy
import os
from pathlib import Path
import subprocess
import sys
import time

import prepare_facts_runtime as prep

STOP_NAME = 'STOP_FACTS_DISPATCH'


def option(argv, name): return Path(argv[argv.index(name) + 1])


def remote_stop(write=False):
    # One shared remote marker lets DS failure stop SOL and SOL failure stop DS.
    # Only marker state/content is transferred; no answer, auth or environment.
    target = prep.C + '/' + STOP_NAME
    command = (f"umask 077; set -C; printf '%s\\n' 'SOL controller stopped new dispatch' > '{target}' 2>/dev/null || test -e '{target}'"
               if write else f"if test -e '{target}'; then printf STOP; else printf OPEN; fi")
    result = subprocess.run(['ssh', 'humanoid-server', command], capture_output=True, timeout=30)
    prep.require(result.returncode == 0, 'Shared STOP SSH unavailable; no new dispatch')
    return result.stdout.strip() == b'STOP' if not write else True


def stop_new(local_stop, carrier, reason):
    if not local_stop.exists():
        try: prep.save(local_stop, {'carrier': carrier, 'recorded_utc': prep.now(), 'reason': reason})
        except FileExistsError: pass
    if carrier == 'sol':
        try: remote_stop(write=True)
        except Exception as error: return 'Remote STOP could not be confirmed: ' + str(error)
    return None


def sol_preflight(args):
    prep.require(prep.ref(args.registration)['sha256'] == args.registration_sha256, 'Exact registration SHA required')
    registration = prep.read(args.registration)
    prep.require(registration['scope'] == 'n5_r5_facts_only' and registration['status'] == 'PREPARED_NOT_RELEASED' and
                 registration['remote_control'] == prep.C, 'Wrong R5 registration')
    prep.check(registration['script'], Path(__file__).with_name('prepare_facts_runtime.py'))
    prep.check(registration['dispatch_script'], Path(__file__))
    exported = prep.read(prep.check(registration['export'])['path'])
    prep.check(args.capacity_ref)
    capacity = prep.read(args.capacity)
    prep.require(capacity['export_receipt_sha256'] == registration['export']['sha256'] and
                 capacity['SOL_contexts'] == 72 and capacity['all_72_text_proxy_checks_pass'] is True and
                 capacity['configured_context_window'] == 512000 and capacity['configured_auto_compact_limit'] == 450000,
                 'Actual R5 complete-carrier capacity gate absent/failed')
    p, _, _, audit = prep.modules(registration['source_root'], registration['source_manifest'])
    prep.check(registration['cli'])
    expected = {(r['packet_id'], r['judge_id']) for r in exported['records'] if r['judge_id'] != 'deepseek'}
    calls = registration['calls']
    prep.require(len(calls) == 72 and len(expected) == 72 and {(r['packet_id'], r['judge_id']) for r in calls} == expected, 'Fixed 72 roster differs')
    for row in calls:
        argv = row['argv']
        prep.require(Path(argv[2]) == p / 'run_sol_review.py' and argv[-1] == '--execute-one-call' and
                     '--allow-development-single' not in argv and '--own-facts-seal' not in argv and
                     option(argv, '--source-root') == Path(registration['source_root']), 'Only registered fresh facts argv allowed')
        prep.check(row['remote_prepare'], option(argv, '--prepared') / 'prepare.json')
        prep.check(row['remote_request'], option(argv, '--prepared') / 'request.http_body.json')
        prep.check(row['preview_stdin'])
        prep.check(row['preview_transport'])
        prep.require(not option(argv, '--output').exists() and not option(argv, '--runtime-home').exists(), 'Never reuse an attempted slot/home')
    return registration, calls, audit


def sol_finish(active, audit):
    process, row, slot, stdout, stderr = active
    code = process.wait()
    stdout.close(); stderr.close()
    result = dict(packet_id=row['packet_id'], judge_id=row['judge_id'], finished_utc=prep.now(),
                  returncode=code, status='FAILED_STOP_NEW_DISPATCH', stdout=prep.ref(slot / 'stdout.log'),
                  stderr=prep.ref(slot / 'stderr.log'), answer_rewritten=False, facts_sealed=False)
    try:
        output = option(row['argv'], '--output')
        result['original_driver_receipt'] = prep.ref(output / 'receipt.json')
        receipt = prep.read(output / 'receipt.json')
        result['raw_answer'] = prep.ref(output / 'raw_answer.json')
        prep.require(code == receipt['exit_code'] == 0 and receipt['status'] == 'COMPLETED_TRANSPORT_VERIFIED_NOT_RELEASED', 'Transport did not verify')
        prep.check(receipt['raw_answer'], output / 'raw_answer.json')
        prep.check(receipt['access_trace'])
        result['auth_removed'] = not (option(row['argv'], '--runtime-home') / 'auth.json').exists()
        prep.require(result['auth_removed'], 'Copied auth was not removed; stop new dispatch')
        prep.require(prep.ref(output / 'stdin.txt')['sha256'] == row['preview_stdin']['sha256'], 'Executed full carrier differs')
        structure = audit.audit_contract(option(row['argv'], '--prepared') / 'prepare.json', output / 'raw_answer.json')
        prep.save(slot / 'facts_contract_audit.json', structure)
        result['structure_audit'] = prep.ref(slot / 'facts_contract_audit.json')
        result['necessary_fact_count'] = structure['necessary_fact_count']
        prep.require(structure['structural_contract_complete'] and not structure['errors'], 'Facts structure/reference audit failed')
        result['status'] = 'COMPLETED_TRANSPORT_AND_STRUCTURE_NOT_SEALED'
    except Exception as error: result['error'] = {'type': type(error).__name__, 'message': str(error)}
    prep.save(slot / 'completion.json', result)
    return result


def run_sol(args):
    prep.require(not args.output.exists(), 'Exclusive new batch directory required; never resume')
    registration, calls, audit = sol_preflight(args)
    if not args.execute_facts: return {'status': 'PREFLIGHT_ONLY_NOT_SENT', 'slots': 72}
    args.output.mkdir(parents=True, exist_ok=False)
    local_stop = Path(__file__).parent / STOP_NAME
    prep.save(args.output / 'start.json', {'registration': prep.ref(args.registration), 'capacity': args.capacity_ref,
              'started_utc': prep.now(), 'max_parallel': 2, 'application_retries': 0, 'hidden_transport_retries': 'unobserved',
              'facts_sealed': 0, 'reports_prepared': 0})
    active, results, next_slot, stop = [], [], 0, None
    try:
        while active or (stop is None and next_slot < 72):
            for item in list(active):
                if item[0].poll() is not None:
                    result = sol_finish(item, audit); results.append(result); active.remove(item)
                    if result['status'] != 'COMPLETED_TRANSPORT_AND_STRUCTURE_NOT_SEALED':
                        stop = stop or {'reason': 'slot_failed', 'packet': result['packet_id'], 'judge': result['judge_id']}
            while stop is None and len(active) < 2 and next_slot < 72:
                if local_stop.exists() or remote_stop(): stop = {'reason': 'shared_stop_observed'}; break
                if any(item[0].poll() is not None for item in active): break
                row = calls[next_slot]
                prep.check(row['remote_prepare'], option(row['argv'], '--prepared') / 'prepare.json')
                prep.check(row['remote_request'], option(row['argv'], '--prepared') / 'request.http_body.json')
                prep.require(not option(row['argv'], '--output').exists() and not option(row['argv'], '--runtime-home').exists(), 'Slot no longer fresh')
                slot = args.output / row['packet_id'] / row['judge_id']
                slot.mkdir(parents=True, exist_ok=False)
                prep.save(slot / 'dispatch.json', {'argv': row['argv'], 'prepare': row['remote_prepare'], 'request': row['remote_request'], 'attempt': 1, 'utc': prep.now()})
                next_slot += 1  # A process-creation failure also spends this attempt.
                stdout, stderr = (slot / 'stdout.log').open('xb'), (slot / 'stderr.log').open('xb')
                try:
                    process = subprocess.Popen(row['argv'], stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                except BaseException: stdout.close(); stderr.close(); raise
                active.append((process, row, slot, stdout, stderr))
                prep.save(slot / 'process.json', {'pid': process.pid, 'utc': prep.now()})
            if stop is not None: break  # Publish STOP once below, then only drain in-flight calls.
            if active: time.sleep(1)
    except BaseException as error: stop = stop or {'reason': type(error).__name__, 'message': str(error)}
    if stop is not None: stop['cross_stop_error'] = stop_new(local_stop, 'sol', stop)
    # No cancel/terminate/resume/retry: every in-flight original attempt finishes.
    for item in active:
        result = sol_finish(item, audit); results.append(result)
        if result['status'] != 'COMPLETED_TRANSPORT_AND_STRUCTURE_NOT_SEALED' and stop is None:
            stop = {'reason': 'in_flight_failed'}
            stop['cross_stop_error'] = stop_new(local_stop, 'sol', stop)
    complete = len(results) == 72 and all(r['status'] == 'COMPLETED_TRANSPORT_AND_STRUCTURE_NOT_SEALED' for r in results)
    result = dict(status='COMPLETED_72_FACTS_NOT_SEALED' if complete else 'STOPPED_NO_NEW_DISPATCH', stop=stop,
                  returned=len(results), attempted=next_slot, results=results, unstarted=[(r['packet_id'], r['judge_id']) for r in calls[next_slot:]],
                  facts_sealed=0, reports_prepared=0, application_retries=0, semantic_acceptance=False, fees_usd=None)
    prep.save(args.output / 'final.json', result)
    return result


def run_ds(args):
    prep.require(not args.output.exists(), 'Exclusive new batch directory required; never resume')
    prep.require(prep.ref(args.export)['sha256'] == args.export_sha256, 'Exact export SHA required')
    exported = prep.read(args.export)
    inputs = prep.read(prep.check(exported['inputs'])['path'])
    card, _ = prep.gates(inputs)
    p, _, _, audit = prep.modules(inputs['source_root'], inputs['source_manifest'])
    prep.check(exported['dispatch_script'], Path(__file__))
    prep.check(args.capacity_ref)
    proposal = prep.read(args.capacity)
    prep.require(proposal['export_receipt_sha256'] == args.export_sha256 and
                 proposal['capture_card_sha256'] == exported['capture_card']['sha256'] and
                 tuple(r['packet_id'] for r in proposal['rows']) == prep.DS_SLOTS, 'Actual DS8 capacity proposal identity differs')
    records = {r['packet_id']: r for r in exported['records'] if r['judge_id'] == 'deepseek'}
    prep.require(len(records) == 8 and card['budget']['api_phase_limit_usd'] == 15, 'DS8/total inherited budget differs')
    for row in proposal['rows']:
        prep.require(row['judge_id'] == 'deepseek' and row['output_token_reservation'] == 65536 and
                     type(row['suggested_input_token_reservation']) is int and row['suggested_input_token_reservation'] > 0, 'DS high-thinking 65536/reservation required')
        prep.check(row['prepared']); prep.check(row['request'])
        prep.require(row['prepared']['sha256'] == records[row['packet_id']]['remote_prepare']['sha256'] and
                     row['request']['sha256'] == records[row['packet_id']]['remote_request']['sha256'], 'DS proposal/request differs')
    if not args.execute_facts: return {'status': 'PREFLIGHT_ONLY_NOT_SENT', 'slots': 8}
    args.output.mkdir(parents=True, exist_ok=False)  # One exclusive writer/attempt.
    stop_path, results = Path(prep.C) / STOP_NAME, []
    final = dict(status='FAILED_STOP_NEW_DISPATCH', started_utc=prep.now(), export=prep.ref(args.export), capacity=args.capacity_ref, results=results)
    try:
        budget = card['budget']
        ledger_path = Path(budget['working_ledger_remote_path'])
        prep.require(ledger_path == Path(prep.C) / 'budget_ledger.working.json', 'Inherited R5 ledger path differs')
        for row in proposal['rows']:
            if stop_path.exists(): final['status'] = 'STOPPED_NO_NEW_DISPATCH'; break
            directory = Path(row['prepared'].get('path', row['prepared'].get('file'))).parent
            prep.require(not any((directory / name).exists() for name in ('receipt.json', 'raw_response.json', 'raw_answer.txt', 'raw_reasoning.txt')), 'Never retry an attempted DS slot')
            prep.check(row['prepared']); prep.check(row['request'])
            release = copy.deepcopy(prep.read(directory / 'release.draft.json'))
            release.update(status='RELEASED_FOR_ONE_CALL', input_token_reservation=row['suggested_input_token_reservation'],
                           reserved_usd=budget['reserved_usd_per_review'], prices_usd_per_million=budget['prices_usd_per_million'], pricing_source=budget['pricing_source'])
            release['budget'] = dict(ledger_file=str(ledger_path), ledger_sha256=prep.ref(ledger_path)['sha256'], budget_id=budget['budget_id'], limit_usd=budget['api_phase_limit_usd'])
            slot = args.output / row['packet_id']; slot.mkdir()
            prep.save(slot / 'release.json', release)
            if stop_path.exists(): final['status'] = 'STOPPED_NO_NEW_DISPATCH'; break
            argv = [sys.executable, '-B', str(p / 'run_confirmation_review.py'), 'run', '--prepared', str(directory),
                    '--release', str(slot / 'release.json'), '--secret', '/data/yuyangliu/secrets/deepseek.env']
            prep.save(slot / 'dispatch.json', dict(argv=argv, utc=prep.now(), attempt=1, ledger_before=prep.ref(ledger_path), release=prep.ref(slot / 'release.json')))
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                               PYTHONPATH=os.pathsep.join(map(str, (Path(inputs['source_root']) / 'src', Path(inputs['source_root']), p))))
            with (slot / 'stdout.log').open('xb') as stdout, (slot / 'stderr.log').open('xb') as stderr:
                code = subprocess.run(argv, cwd=inputs['source_root'], env=environment, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr).returncode
            record = dict(packet_id=row['packet_id'], judge_id='deepseek', returncode=code, finished_utc=prep.now(), status='FAILED_STOP_NEW_DISPATCH')
            results.append(record)
            try:
                receipt = prep.read(directory / 'receipt.json')
                record['original_runner_receipt'] = prep.ref(directory / 'receipt.json')
                record['ledger_after'] = prep.ref(ledger_path)
                prep.require(code == 0 and receipt.get('complete') is True, 'Original DS runner incomplete')
                structure = audit.audit_contract(directory / 'prepare.json', directory / 'raw_answer.txt')
                prep.save(slot / 'facts_contract_audit.json', structure)
                record['structure_audit'] = prep.ref(slot / 'facts_contract_audit.json')
                prep.require(structure['structural_contract_complete'] and not structure['errors'], 'DS facts structure/reference audit failed')
                record.update(status='COMPLETED_TRANSPORT_AND_STRUCTURE_NOT_SEALED', raw_answer=prep.ref(directory / 'raw_answer.txt'), necessary_fact_count=structure['necessary_fact_count'])
            except Exception as error: record['error'] = str(error)
            prep.save(slot / 'return.json', record)
            prep.require(record['status'] == 'COMPLETED_TRANSPORT_AND_STRUCTURE_NOT_SEALED', 'Stop after incomplete DS slot; no retry')
        else: final['status'] = 'COMPLETED_8_FACTS_NOT_SEALED'
    except BaseException as error:
        final['error'] = str(error); stop_new(stop_path, 'deepseek', str(error))
    final.update(finished_utc=prep.now(), facts_sealed=0, reports_prepared=0, application_retries=0, semantic_acceptance=False)
    prep.save(args.output / 'execution_receipt.json', final)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('carrier', choices=('sol', 'deepseek'))
    parser.add_argument('--registration', type=Path)
    parser.add_argument('--registration-sha256')
    parser.add_argument('--export', type=Path)
    parser.add_argument('--export-sha256')
    parser.add_argument('--capacity', type=Path, required=True)
    parser.add_argument('--capacity-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute-facts', action='store_true')
    args = parser.parse_args()
    args.capacity_ref = {'path': str(args.capacity.resolve()), 'sha256': args.capacity_sha256}
    try:
        result = run_sol(args) if args.carrier == 'sol' else run_ds(args)
    except BaseException as error:
        if args.execute_facts:
            marker = Path(__file__).parent / STOP_NAME if args.carrier == 'sol' else Path(prep.C) / STOP_NAME
            cross_error = stop_new(marker, args.carrier, str(error))
            print(prep.json.dumps({'status': 'PRE_DISPATCH_FAILED_STOP_NEW', 'error': str(error), 'cross_stop_error': cross_error}))
        else: print(prep.json.dumps({'status': 'PREFLIGHT_FAILED_NOT_SENT', 'error': str(error)}))
        return 1
    print(prep.json.dumps({k: result.get(k) for k in ('status', 'slots', 'returned', 'attempted')}))
    return 0 if result['status'].startswith(('PREFLIGHT_', 'COMPLETED_')) else 1


if __name__ == '__main__': raise SystemExit(main())
