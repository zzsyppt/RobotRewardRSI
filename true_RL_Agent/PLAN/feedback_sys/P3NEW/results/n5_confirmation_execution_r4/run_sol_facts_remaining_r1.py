"""Prepare/run only 68 unstarted original SOL slots under runtime amendment R1."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import run_sol_facts_batch as old

E, F, prep = old.E, old.F, old.prep
RUN = E / 'sol_facts_remaining_r1'
STOP = E / 'STOP_FACTS_DISPATCH_AMENDMENT_R1'
OLD_SHA = 'a1193b2b776b6daf6f40941ea807bfae153828ba87ecae546846274cd5d850f6'
RETAINED = E / 'control_cpu/facts_runtime_supervision/terminal_supervision_receipt.json'
RETAINED_SHA = '4c577a19ddc609b4faaaf5cf1b167801015c7c65d1e21fc1d1333e6fa2e40072'
AUDITOR = E / 'facts_contract_audit.py'
AUDITOR_SHA = '018078561cdc869562a4b5379a63f647418d450e6990cc75beb9c58b715dd5fd'
EXCLUDED = {('episode-01', 'gpt6.1-sol-a'), ('episode-02', 'gpt6.1-sol-a'),
            ('episode-02', 'gpt6.1-sol-b'), ('episode-03', 'gpt6.1-sol-a')}


def pin(path, sha):
    return old.check({'sha256': sha}, path)


def prepare(args):
    amendment_ref = pin(args.amendment, args.amendment_sha256)
    amendment = prep.read(args.amendment)
    prep.require(amendment['status'] == 'REGISTERED_NOT_EXECUTED', 'Amendment not registered')
    prep.require(amendment['original_capture_card_sha256'] == prep.CARD_SHA and
                 amendment['original_source_manifest_sha256'] == prep.SOURCE_SHA,
                 'Amendment original source/card differs')
    prep.require(amendment['scope'] == 'r4_runtime_limit_amendment_same_cohort_engineering_semantic_acceptance' and
                 amendment['recovery_gate']['expected_slot'] ==
                 {'packet_id': 'episode-13', 'judge_id': 'deepseek', 'stage': 'facts'} and
                 amendment['recovery_gate']['required_output_max_tokens'] == 65536,
                 'Amendment scope/recovery gate differs')
    pin(Path(old.__file__), OLD_SHA)
    pin(AUDITOR, AUDITOR_SHA)
    for path, sha in old.PINS.items(): pin(path, sha)
    prep.verify_source(old.SOURCE, E / 'runtime_source_manifest.json')
    pin(E / 'capture_card.json', prep.CARD_SHA)
    prep.require((E / 'STOP_FACTS_DISPATCH').is_file(), 'Original STOP must remain')
    pin(RETAINED, RETAINED_SHA)
    retained = prep.read(RETAINED)['SOL_returns']
    prep.require({(r['packet_id'], r['judge_id']) for r in retained} == EXCLUDED and
                 len(retained) == 4, 'Actual retained four differ')
    final = prep.read(old.RUN / 'final.json')
    prep.require(final['status'] == 'STOPPED_NO_NEW_DISPATCH' and
                 final['attempted_slots'] == final['returned_calls'] == 4,
                 'Original batch not drained at exactly four')
    contexts = {(r['packet_id'], r['judge_id']): r for r in
                prep.read(F / 'local_preparation_receipt.json')['contexts']}
    records = {(r['packet_id'], r['judge_id']): r for r in
               prep.read(F / 'export_receipt.json')['records']}
    calls = prep.read(F / 'pending_sol_launch_commands.json')['calls']
    prep.require(len(calls) == 72 and len({(r['packet_id'], r['judge_id']) for r in calls}) == 72,
                 'Original 72 roster differs')
    retained_by_key = {(r['packet_id'], r['judge_id']): r for r in retained}
    registered = {(r['packet_id'], r['judge_id']): r for r in amendment['retained_sol_facts']}
    prep.require(len(registered) == 4 and set(registered) == EXCLUDED, 'Amendment retained four differ')
    for row in calls:
        key, argv = (row['packet_id'], row['judge_id']), row['argv']
        prepared, output, runtime = (old.option(argv, flag) for flag in
                                     ('--prepared', '--output', '--runtime-home'))
        prep.require(argv[-1] == '--execute-one-call' and Path(argv[2]) == old.P / 'run_sol_review.py',
                     'Original runner/argv differs')
        original, context = records[key], contexts[key]
        old.check(original['remote_prepare'], prepared / 'prepare.json')
        old.check(original['remote_request'], prepared / 'request.http_body.json')
        old.check(context['stdin'])
        transport = prep.read(Path(old.check(context['transport_receipt'])['path']))
        for item in transport['child_read_whitelist']: old.check(item)
        card = prep.read(prepared / 'prepare.json')
        prep.require(card['stage'] == 'facts' and card['judge_id'] == key[1] and
                     card['packet_id'] == key[0] and card['max_output_tokens'] == 32768 and
                     card['own_facts_seal'] is None and card['capacity_facts_fixture'] is None,
                     'Original facts prepared differs')
        if key not in EXCLUDED:
            prep.require(not output.exists() and not runtime.exists(), 'Unstarted slot not fresh: ' + str(key))
            continue
        saved = retained_by_key[key]
        prep.require(registered[key]['stage'] == 'facts' and
                     registered[key]['return']['sha256'] == saved['return']['sha256'] and
                     registered[key]['structure_audit']['sha256'] == saved['contract_audit']['sha256'],
                     'Amendment retained evidence differs')
        result = prep.read(Path(old.check(saved['return'])['path']))
        prep.require(result['status'] == 'COMPLETED_TRANSPORT_AND_MECHANICAL_ONLY_NOT_SEALED'
                     and result['returncode'] == 0, 'Retained call incomplete')
        old.check(result['raw_answer'])
        prep.require(registered[key]['raw_answer']['sha256'] == result['raw_answer']['sha256'],
                     'Amendment retained raw answer differs')
        driver = prep.read(Path(old.check(result['original_driver_receipt'])['path']))
        prep.require(driver['status'] == 'COMPLETED_TRANSPORT_VERIFIED_NOT_RELEASED'
                     and driver['exit_code'] == 0, 'Retained driver not verified')
        old.check(driver['raw_answer']); old.check(driver['access_trace'])
        executed = prep.read(output / 'transport_receipt.json')
        prep.require(executed['source_request']['sha256'] == card['request_sha256'] and
                     prep.ref(output / 'stdin.txt')['sha256'] == context['stdin']['sha256'],
                     'Retained original input binding differs')
        audit = prep.read(Path(old.check(saved['contract_audit'])['path']))
        prep.require(not audit['errors'] and audit['structural_contract_complete'] and
                     audit['inputs']['answer']['sha256'] == result['raw_answer']['sha256'] and
                     audit['inputs']['request']['sha256'] == card['request_sha256'] and
                     audit['inputs']['prepared']['sha256'] == original['remote_prepare']['sha256'],
                     'Retained structure/input SHA differs')
    remaining = [row for row in calls if (row['packet_id'], row['judge_id']) not in EXCLUDED]
    prep.require(len(remaining) == 68, 'Remaining roster is not 68')
    return remaining, contexts, records, amendment_ref


def recovery_gate(args):
    prep.require(all((args.ds_recovery_receipt, args.ds_recovery_receipt_sha256,
                      args.ds_recovery_structure_audit, args.ds_recovery_structure_audit_sha256,
                      args.ds_recovery_http_audit, args.ds_recovery_http_audit_sha256)),
                 'Actual DS recovery receipt, structure and HTTP audits required before execution')
    receipt_ref = pin(args.ds_recovery_receipt, args.ds_recovery_receipt_sha256)
    audit_ref = pin(args.ds_recovery_structure_audit, args.ds_recovery_structure_audit_sha256)
    http_ref = pin(args.ds_recovery_http_audit, args.ds_recovery_http_audit_sha256)
    receipt, audit = prep.read(args.ds_recovery_receipt), prep.read(args.ds_recovery_structure_audit)
    http = prep.read(args.ds_recovery_http_audit)
    prep.require(http['status'] == 'PASS_ONLY_MAX_TOKENS_CHANGED' and
                 http['old_max_tokens'] == 32768 and http['new_max_tokens'] == 65536 and
                 http['old_request_sha256'] == '67688182fe8d263092b4372c2550cc5f47f9966686307a52870158d26d68eadb' and
                 http['new_request_sha256'] == receipt['request_sha256'] and
                 http['request']['sha256'] == receipt['request_sha256'] and
                 http['prepared']['sha256'] == receipt['prepared']['sha256'],
                 'Recovery HTTP change/prepared binding differs')
    prep.require(receipt['packet_id'] == 'episode-13' and receipt['judge_id'] == 'deepseek' and
                 receipt['stage'] == 'facts' and receipt['complete'] is True and
                 receipt['request_attempted'] is True and not receipt['incomplete_reasons'],
                 'Actual DS episode-13 recovery incomplete')
    prep.require(audit['packet_id'] == 'episode-13' and audit['judge_id'] == 'deepseek' and
                 not audit['errors'] and audit['structural_contract_complete'] and
                 audit['frozen_reference_audit']['mechanical_valid'] and
                 audit['frozen_reference_audit']['review_complete'] and
                 audit['inputs']['answer']['sha256'] == receipt['content']['sha256'] and
                 audit['inputs']['request']['sha256'] == receipt['request_sha256'],
                 'Recovery structure/reference/raw-answer binding differs')
    return [receipt_ref, audit_ref, http_ref]


def stop_marker(reason):
    try:
        with STOP.open('x', encoding='utf-8') as stream:
            json.dump({'observed_at_utc': old.now(), 'reason': reason}, stream)
    except FileExistsError:
        pass


def finish(item, review):
    result = old.finish(item, review)  # Unchanged transport/envelope/frozen coverage implementation.
    if result['status'] == 'COMPLETED_TRANSPORT_AND_MECHANICAL_ONLY_NOT_SEALED':
        try:
            slot = item[2]
            code = subprocess.run([sys.executable, '-B', str(AUDITOR), '--prepared',
                                   str(old.option(item[1]['argv'], '--prepared') / 'prepare.json'),
                                   '--answer', result['raw_answer']['path'], '--output',
                                   str(slot / 'facts_contract_audit.json')],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
            audit = prep.read(slot / 'facts_contract_audit.json')
            result['structural_contract_audit'] = prep.ref(slot / 'facts_contract_audit.json')
            prep.require(code == 0 and not audit['errors'] and audit['structural_contract_complete'],
                         'Returned structure incomplete')
        except Exception as error:
            result['status'] = 'FAILED_STRUCTURE_STOP_NEW_DISPATCH'
            result['structure_error'] = {'type': type(error).__name__, 'message': str(error)}
    if result['status'] != 'COMPLETED_TRANSPORT_AND_MECHANICAL_ONLY_NOT_SEALED':
        stop_marker({'packet_id': result['packet_id'], 'judge_id': result['judge_id'],
                     'returned_status': result['status']})
    prep.save(item[2] / 'completion.json', result)  # Final slot status; old return.json stays intact.
    return result


def execute(args, calls, contexts, records, amendment_ref):
    gates = recovery_gate(args)
    prep.require(not STOP.exists(), 'New amendment STOP exists')
    RUN.mkdir(exist_ok=False)  # Exclusive attempt; never restart this controller.
    _, review, _ = prep.modules(old.SOURCE)
    source = [prep.ref(old.P / name) for name in ('run_sol_review.py', 'prepare_desktop_review.py',
              'run_confirmation_review.py', 'confirmation_review_coverage.py',
              'N5_SOL_CLI_POLICY_R4.json', 'N5_REVIEW_ASSIGNMENT_R4.json')]
    prep.save(RUN / 'start.json', {'scope': 'runtime_amendment_not_original_R4_acceptance',
        'started_at_utc': old.now(), 'script': prep.ref(Path(__file__)), 'amendment': amendment_ref,
        'recovery_gate': gates, 'retained': prep.ref(RETAINED), 'original_commands': prep.ref(F / 'pending_sol_launch_commands.json'),
        'slots': 68, 'max_parallel': 2, 'original_R4_disposition': 'INCONCLUSIVE',
        'application_retries': 0, 'facts_seals_created': 0, 'reports_created': 0})
    active, results, index, stop = [], [], 0, None
    try:
        while active or (stop is None and index < len(calls)):
            for item in list(active):
                if item[0].poll() is not None:
                    result = finish(item, review); results.append(result); active.remove(item)
                    if result['status'] != 'COMPLETED_TRANSPORT_AND_MECHANICAL_ONLY_NOT_SEALED':
                        stop = 'returned_call_failed'
            while stop is None and len(active) < 2 and index < len(calls):
                if STOP.exists(): stop = 'amendment_stop_marker'; break
                if any(item[0].poll() is not None for item in active): break
                row = calls[index]; key = row['packet_id'], row['judge_id']
                for bound in source: old.check(bound)
                pin(AUDITOR, AUDITOR_SHA)
                for name, file in (('remote_request', 'request.http_body.json'), ('remote_prepare', 'prepare.json')):
                    old.check(records[key][name], old.option(row['argv'], '--prepared') / file)
                prep.require(not old.option(row['argv'], '--output').exists() and
                             not old.option(row['argv'], '--runtime-home').exists(), 'Slot no longer fresh')
                slot = RUN / key[0] / key[1]; slot.mkdir(parents=True, exist_ok=False)
                prep.save(slot / 'dispatch.json', {'scope': 'facts_only', 'dispatched_at_utc': old.now(),
                    'packet_id': key[0], 'judge_id': key[1], 'argv': row['argv'], 'attempt': 1,
                    'original_prepare': records[key]['remote_prepare'], 'original_request': records[key]['remote_request'],
                    'prepared_transport': contexts[key]['transport_receipt']})
                index += 1
                stdout, stderr = (slot / 'stdout.log').open('xb'), (slot / 'stderr.log').open('xb')
                try:
                    process = subprocess.Popen(row['argv'], stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                except BaseException:
                    stdout.close(); stderr.close(); raise
                active.append((process, row, slot, stdout, stderr))
                prep.save(slot / 'process.json', {'pid': process.pid, 'registered_at_utc': old.now()})
            if active: time.sleep(1)
    except BaseException as error:
        stop = {'reason': 'controller_exception', 'type': type(error).__name__, 'message': str(error)}
        stop_marker(stop)
    for item in active: results.append(finish(item, review))  # Drain, never cancel/retry.
    complete = len(results) == 68 and all(r['status'] == 'COMPLETED_TRANSPORT_AND_MECHANICAL_ONLY_NOT_SEALED' for r in results)
    prep.save(RUN / 'final.json', {'status': 'COMPLETED_68_FACTS_NOT_SEALED' if complete else 'STOPPED_NO_NEW_DISPATCH',
        'finished_at_utc': old.now(), 'stop_reason': stop, 'attempted_slots': index, 'results': results,
        'unstarted_slots': [{'packet_id': r['packet_id'], 'judge_id': r['judge_id']} for r in calls[index:]],
        'original_R4_disposition': 'INCONCLUSIVE', 'facts_seals_created': 0, 'reports_created': 0,
        'semantic_admission_claimed': False, 'application_retries': 0, 'cancellations': 0})
    return 0 if complete else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--amendment', type=Path, required=True)
    parser.add_argument('--amendment-sha256', required=True)
    parser.add_argument('--ds-recovery-receipt', type=Path)
    parser.add_argument('--ds-recovery-receipt-sha256')
    parser.add_argument('--ds-recovery-structure-audit', type=Path)
    parser.add_argument('--ds-recovery-structure-audit-sha256')
    parser.add_argument('--ds-recovery-http-audit', type=Path)
    parser.add_argument('--ds-recovery-http-audit-sha256')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    calls, contexts, records, amendment_ref = prepare(args)
    if not args.execute:
        print(json.dumps({'status': 'PREPARED_68_NOT_EXECUTED', 'slots': len(calls), 'amendment': amendment_ref,
                          'max_parallel': 2, 'old_STOP_preserved': True, 'new_STOP': str(STOP)}))
        return 0
    return execute(args, calls, contexts, records, amendment_ref)


if __name__ == '__main__':
    raise SystemExit(main())
