"""Prepare the R5 candidate only; no resource claim, ledger, deployment or launch."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'n5_confirmation_execution_r4'
SOURCE = HERE.parent / 'n5_confirmation_source_r5'
PHASE = 'PLAN/feedback_sys/P3NEW/'


def read(p): return json.loads(Path(p).read_bytes())
def ref(p):
    p = Path(p).resolve()
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'size_bytes': p.stat().st_size}
def save(name, value):
    with (HERE / name).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2); f.write('\n')


def main():
    old = read(PRIOR / 'capture_card.json')
    freeze = read(SOURCE / 'source_freeze_receipt.json')
    assert freeze['status'] == 'SOURCE_FROZEN_CPU_VERIFIED_NOT_CAPTURE_RELEASE'
    assert ref(SOURCE / 'source_freeze_receipt.json')['sha256'] == 'b009ffdaaaa7f23ecd075c6784afa04624afd8f0cf95e83a0cb6938f0b774c53'
    for key in ('source_manifest', 'source_bundle', 'verification', 'commands'):
        assert ref(freeze[key]['path']) == freeze[key]
    manifest, verification = read(freeze['source_manifest']['path']), read(freeze['verification']['path'])
    commands = read(freeze['commands']['path'])
    assert len(manifest) == freeze['source_files'] == verification['source_files']
    assert freeze['source_unchanged_after_execution'] and all(r['exit_code'] == 0 for r in verification['records'])

    def frozen(name):
        p = SOURCE / 'source' / name
        assert {k: ref(p)[k] for k in ('sha256', 'size_bytes')} == manifest[name]
        return {**ref(p), 'remote_path': commands['source_root'] + '/' + name}

    card = deepcopy(old)
    card.pop('frozen_at_utc')
    card.update(status='CANDIDATE_NOT_RELEASED', prepared_at_utc=datetime.now(timezone.utc).isoformat(),
        purpose='R5 unused-cohort confirmation candidate after task-report, reference-ID and neutral focus-index development',
        main_plan_version=re.search(r'版本：v([\d.]+)', (SOURCE / 'source/PLAN/feedback_sys/FEEDBACK_SYSTEM_UPGRADE_PLAN_ZH.md').read_text(encoding='utf-8')).group(1),
        execution_status='AWAITING_ACTUAL_RESOURCE_LEDGER_AND_ROOT_RELEASE', review_status='NOT_STARTED_UNQUALIFIED',
        output_root=commands['capture_root'], execution_control_root=commands['execution_control_root'],
        original_card=ref(PRIOR / 'capture_card.json'), static_preflight=freeze['verification'], cohort_verification=freeze['verification'],
        experiment_files=freeze['source_manifest'], run_contract=frozen(PHASE + 'N5_RUN_CARD_R5_ZH.md'),
        review_protocol=frozen(PHASE + 'N5_FACT_REVIEW_PROTOCOL_ZH.md'), dual_review_protocol=frozen(PHASE + 'REVIEW_PROTOCOL_ZH.md'),
        main_plan=frozen('PLAN/feedback_sys/FEEDBACK_SYSTEM_UPGRADE_PLAN_ZH.md'),
        review_assignment=frozen(PHASE + 'N5_REVIEW_ASSIGNMENT_R5.json'), cli_policy=frozen(PHASE + 'N5_SOL_CLI_POLICY_R5.json'),
        latest_resource_prestart=None, module_review_status='PENDING_NEW_CAPTURE', prior_cohort_disposition=None,
        prior_cohort_status='Original R4 and output-cap amendment remain INCONCLUSIVE; no old answer is a new R5 confirmation',
        development_regression_evidence=freeze['prior_same_source_evidence'])
    card['source'] = {'remote_root': commands['source_root'], 'manifest': freeze['source_manifest'],
        'manifest_layout': old['source']['manifest_layout'], 'file_count': len(manifest), 'bundle': freeze['source_bundle'],
        'freeze_receipt': ref(SOURCE / 'source_freeze_receipt.json'), 'local_snapshot_verified': True,
        'remote_verified': True, 'is_final_n5_experiment_freeze': True, 'freeze_scope': 'SOURCE_AND_CPU_ONLY'}
    for task, spec in card['tasks'].items():
        result_ref = freeze['tasks'][task]['local_preflight_result']
        assert ref(result_ref['path']) == result_ref
        result, row = read(result_ref['path']), commands['tasks'][task]
        assert result['checkpoint_sha256'] == 'sha256:' + spec['checkpoint']['sha256']
        spec.update(current_evaluation_binding=result['evaluation_binding'],
            preserved_training_binding_hash=result['preserved_training_binding_hash'],
            source_training_manifest_hash=result['source_training_manifest_hash'], static_preflight_result=result_ref,
            runner_seeds=freeze['tasks'][task]['runner_seeds'],
            cohort=({'cli': 'diagnostic', 'diagnostic_cohort': commands['diagnostic_cohort'],
                     'namespace': result['seed_namespace'], 'fixed_terrain_new_reset_trajectories': True}
                    if task == 'stair' else {'eval_base_seed': commands['push_eval_base_seed']}))
        before, after = row['preflight_argv'], row['capture_argv_not_executed']
        diffs = [{'index': i, 'before': a, 'after': b} for i, (a, b) in enumerate(zip(before, after)) if a != b]
        assert len(before) == len(after) and {before[d['index'] - 1] for d in diffs} == {'--mode', '--output', '--video-dir'}
        card['commands'][task] = {'cwd': row['cwd'], 'argv': after, 'actual_preflight_argv': before, 'only_changes': diffs,
            'environment_set': row['capture_environment'], 'environment_unset_process_local': row['capture_environment_unset'], 'executed': False}
    card['model_roles']['deepseek_judge']['max_tokens'] = 65536
    for role in ('gpt61_sol_a_judge', 'gpt61_sol_b_judge'):
        card['model_roles'][role]['policy'] = card['cli_policy']
    materials = card['materials']
    materials.update(review_assignment_sha256=card['review_assignment']['sha256'],
        actual_packet_task_routing='PENDING: verify actual R5 first per-task episode/focus IDs after packing',
        output_reserved_by_judge={'deepseek': 65536, 'gpt6.1-sol-a': 32768, 'gpt6.1-sol-b': 32768},
        explicit_numeric_preparation_options=['--profile', 'related_v2', '--max-output-tokens', '32768',
            '--capture-card', commands['execution_control_root'] + '/capture_card.json', '--protocol',
            card['review_protocol']['remote_path'], '--review-assignment', card['review_assignment']['remote_path']],
        explicit_media_preparation_options=['--review-assignment', card['review_assignment']['remote_path']],
        packed_requests_are_not_dispatch_inputs=True,
        actual_dispatch_preparation={'entrypoint': frozen(PHASE + 'prepare_facts_offline.py'),
            'command': ['python', '-B', commands['source_root'] + '/' + PHASE + 'prepare_facts_offline.py',
                        '--index', '<actual-40-unit-index.json>', '--output', '<new-facts-directory>'],
            'scope': 'Per-judge actual requests only: DS65536/SOL32768. Recompute capacity; do not dispatch packer default prepared requests.'})
    # Existing shared proxy totals describe SOL only, not the DS hard output cap.
    materials['proxy_reservation_scope'] = 'Existing 32768/433728 fields apply only to SOL; DeepSeek actual request budget uses output_reserved_by_judge.'
    card['resources'].update(receipt=None, preparation_disk_free_bytes=None,
        preparation_disk_gate_passed={'stair': None, 'push': None}, candidate_gate_status='PENDING_ACTUAL_RESOURCE_SNAPSHOT')
    card['budget'].update(history_peak_usage_ceiling_usd=None, history_plus_planned_reserve_usd=None,
        initial_ledger=None, ledger_inheritance=None, reserved_usd_per_review=0.20, new_api_reservation_total_usd=3.2,
        working_ledger_remote_path=commands['execution_control_root'] + '/budget_ledger.working.json',
        reservation_status='CANDIDATE: actual ledger and per-request formula required before any API release',
        author_slots_used=8, new_author_slots=0)
    card['known_limits'] = [v for v in card['known_limits'] if not v.startswith('No R4 module review')]
    card['known_limits'].append('No R5 capture, observation, facts seal or report exists. Resource/ledger/card SHA remain pending; estimated cleanup space is not PASS.')
    card['stop_policy'] = 'Preserve all historical STOP markers; no inherited STOP is cleared by this candidate. Any new dispatch requires explicit root release.'
    assert card['acceptance'] == old['acceptance'] and card['capture_profile'] == old['capture_profile'] and card['selection'] == old['selection']
    assert card['resources']['minimum_disk_before_task_bytes'] == old['resources']['minimum_disk_before_task_bytes']
    save('capture_card.candidate.json', card)
    original = (PRIOR / 'capture_driver.py').read_text(encoding='utf-8')
    old_sha = ref(PRIOR / 'capture_card.json')['sha256']
    assert original.count(old_sha) == 1
    template = original.replace(old_sha, 'PENDING_ROOT_FINAL_CAPTURE_CARD_SHA')
    compile(template, str(HERE / 'capture_driver.candidate.py'), 'exec')
    with (HERE / 'capture_driver.candidate.py').open('x', encoding='utf-8', newline='') as f: f.write(template)
    save('candidate_preparation.json', {'status': 'CANDIDATE_NOT_RELEASED_NOT_DEPLOYED',
        'card': ref(HERE / 'capture_card.candidate.json'), 'driver_template': ref(HERE / 'capture_driver.candidate.py'),
        'parent_driver': ref(PRIOR / 'capture_driver.py'), 'driver_only_change': 'Replace exact R4 card commitment with an intentionally non-valid pending placeholder; bind final SHA after root release',
        'source_freeze': ref(SOURCE / 'source_freeze_receipt.json'), 'source_unchanged': True,
        'remaining': ['actual resource snapshot', 'latest original inherited ledger bytes/SHA and cumulative reservation',
                      'root capture release', 'final new card SHA and driver commitment', 'postcapture bindings reviewed before deployment'],
        'ledger_created': False, 'driver_imported_or_executed': False, 'capture_or_models': 0})


if __name__ == '__main__': main()
