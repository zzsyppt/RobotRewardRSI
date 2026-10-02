"""Bind the returned static r4 evidence; never authorize capture."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[4]
SOURCE = HERE / 'source'
sys.path.insert(0, str(SOURCE / 'src'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ref(p):
    p = Path(p)
    return dict(path=str(p), sha256=sha(p), size_bytes=p.stat().st_size)


def load(p):
    return json.loads(Path(p).read_bytes())


def save(name, value):
    with (HERE / name).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    from true_rl_agent.tasks.stair_climbing.unified_protocol import StairUnifiedProtocol

    card = load(HERE / 'preflight_card.json')
    remote = HERE / 'remote_preflight'
    result = load(remote / 'task_preflight_results.json')
    source_manifest = load(HERE / 'source_manifest.json')
    receipt = load(remote / 'remote_static_receipt.json')
    prior = HERE.parent / 'n5_source_preflight_r3'
    inherited = load(HERE / 'inherited_r3_evidence.json')
    assert all(sha(row['path']) == row['sha256'] for row in inherited['source'].values())
    raw_asset = load(prior / 'remote_preflight/asset_cache_check.json')
    extra_asset = load(prior / 'asset_cache_supplement.json')
    history = load(prior / 'history_disambiguation.json')
    expected_remote_prefix = '/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_preflight_r4/'
    checked = []
    for entry in receipt['files']:
        assert entry['path'].startswith(expected_remote_prefix)
        relative = entry['path'][len(expected_remote_prefix):]
        p = Path('\\\\?\\' + str(remote)) / relative
        assert sha(p) == entry['sha256'], relative
        checked.append(relative)
    assert all(sha(SOURCE / p) == h for p, h in source_manifest['files'].items())
    current_changes = [p for p, h in source_manifest['files'].items() if sha(PROJECT / p) != h]
    assert not current_changes, current_changes
    old = HERE.parent / 'n5_source_preflight_r3'
    old_manifest = load(old / 'source_manifest.json')
    assert all(sha(old / 'source' / p) == h for p, h in old_manifest['files'].items())
    freeze = load(HERE / 'local_freeze_receipt.json')
    assert all(sha(p) == h for p, h in freeze['old_card_and_manifest_unchanged'].items())

    protocol_path = SOURCE / 'configs/tasks/stair_go2_12dof_unified_protocol_v3.json'
    suite = StairUnifiedProtocol.load(protocol_path).evaluation_suite
    assignments = suite.private_cell_assignment('public_dev')
    mapping = {'stair': [], 'push': []}
    assert len(assignments) == card['tasks']['stair']['numeric_eval_envs']
    for env in card['tasks']['stair']['selected_envs']:
        cell = assignments[env]
        mapping['stair'].append(dict(selected_env=env, terrain_type_index=cell.terrain_type_index,
            episode_index_within_cell=env % suite.episodes_per_seed_per_cell, cell=cell.spec.identity_payload()))
    push_manifest = load(remote / 'push/composition/resolved_cell_manifest.json')
    params = push_manifest['robot_task_binding']['task_projection']['push_resolved_protocol']
    names = ('low', 'medium', 'high', 'repeated')
    counts = [params['evaluation_cohort_counts'][n] for n in names]
    assert sum(counts) == card['tasks']['push']['numeric_eval_envs']
    for env in card['tasks']['push']['selected_envs']:
        offset = 0
        for i, (name, count) in enumerate(zip(names, counts)):
            if env < offset + count:
                mapping['push'].append(dict(selected_env=env, stratum=name, stratum_index_0based=i,
                    direction_index_0based=(env-offset) % params['evaluation_direction_policy']['count'],
                    planned_push_count=2 if name == 'repeated' else 1))
                break
            offset += count
    prior_mapping = load(HERE.parent / 'N5_preflight_development_draft/remote_static_materials/static_preflight_summary.json')['static_env_mapping']
    for task in mapping:
        for now, previous in zip(mapping[task], prior_mapping[task]):
            assert all(previous[k] == v for k, v in now.items()), (task, now)
    mapping['evidence'] = dict(stair_protocol=ref(protocol_path),
        stair_mapping_source=ref(SOURCE / 'src/true_rl_agent/tasks/stair_climbing/ordered_evaluation.py'),
        push_mapping_source=ref(SOURCE / 'src/true_rl_agent/tasks/push_recovery/runtime.py'),
        push_manifest=ref(remote / 'push/composition/resolved_cell_manifest.json'),
        boundary='CPU static mapping from current frozen task-owned suite and recorded Push bucket/count rule; no environment execution.')
    save('static_env_mapping.json', mapping)
    # Independent CPU reconstruction from this snapshot and returned manifests.
    from true_rl_agent.composition.cell_manifest import load_resolved_cell_manifest
    from true_rl_agent.evaluation.binding import EvaluationBindingV1, canonical_json_hash
    from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
    from true_rl_agent.tasks.stair_climbing.diagnostic_evaluation import _binding
    from true_rl_agent.tasks.stair_climbing.seed_policy import STAIR_PUBLIC_DEV_DIAGNOSTIC_EVAL_NAMESPACE
    from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
    bindings = {}
    for task in ('stair', 'push'):
        returned = result[task]['result']
        manifest_file = remote / task / 'composition/resolved_cell_manifest.json'
        manifest = load_resolved_cell_manifest(manifest_file)
        command = result[task]['argv']
        expected_root = card['source']['remote_root']
        assert command[1].startswith(expected_root + '/scripts/')
        assert result[task]['environment']['PYTHONPATH'] == expected_root + '/src'
        assert receipt['source_root'] == expected_root
        assert receipt['source_manifest']['sha256'] == sha(HERE / 'source_manifest.json')
        assert receipt['source_bundle']['sha256'] == sha(HERE / 'source.tar.gz')
        assert command[command.index('--output') + 1] == expected_remote_prefix + task
        if task == 'stair':
            config = StairClimbingTaskConfig.defaults(protocol_path,
                resolved_cell_task_protocol_hash=manifest.resolved_cell_task_protocol_hash)
            expected = _binding(config, STAIR_PUBLIC_DEV_DIAGNOSTIC_EVAL_NAMESPACE).to_public_dict()
            assert command[command.index('--protocol') + 1] == expected_root + '/' + protocol_path.relative_to(SOURCE).as_posix()
        else:
            config = PushRecoveryTaskConfig.from_manifest(manifest,
                eval_base_seed=int(command[command.index('--eval-base-seed') + 1]))
            expected = EvaluationBindingV1(
                eval_protocol_hash=config.evaluation_protocol().protocol_hash,
                acceptance_protocol_hash=config.acceptance_protocol().protocol_hash,
                task_protocol_hash=manifest.resolved_cell_task_protocol_hash,
                seed_cohort_hash=config.seed_cohort_hash()).to_public_dict()
        assert expected == returned['evaluation_binding'], task
        job_file = remote / task / 'public_dev_job.json'
        job = load(job_file)
        spec = job['spec'] if task == 'stair' else job
        plan = spec['evaluation_plan']
        assert plan['evaluation_binding'] == expected
        assert canonical_json_hash(plan) == spec['evaluation_plan_hash'] == returned['evaluation_plan_hash']
        expected_job_hash = returned['public_dev_job_spec_hash'] if task == 'stair' else returned['job_spec_hash']
        assert canonical_json_hash(spec) == expected_job_hash
        assert plan['training_binding_hash'] == returned['preserved_training_binding_hash'] == card['tasks'][task]['inherited_training_binding_hash']
        assert plan['checkpoint']['checksum'] == returned['checkpoint_sha256']
        assert returned['source_training_manifest_hash'] == card['tasks'][task]['inherited_source_training_manifest_hash']
        assert manifest.manifest_hash == returned['current_manifest_hash' if task == 'stair' else 'evaluation_manifest_hash']
        bindings[task] = dict(status='INDEPENDENT_CURRENT_SNAPSHOT_BINDING_MATCH',
            reconstructed_evaluation_binding=expected, manifest=ref(manifest_file), job=ref(job_file),
            actual_result=ref(remote / task / 'result.json'),
            training_binding_hash=plan['training_binding_hash'], source_training_manifest_hash=returned['source_training_manifest_hash'],
            remote_source_root=expected_root, source_manifest_sha256=sha(HERE / 'source_manifest.json'),
            job_and_plan_hashes_recomputed=True, same_source_script_protocol_pythonpath_and_receipt=True)
    imported = {name: str(Path(module.__file__).resolve()) for name, module in sys.modules.items()
                if name.startswith('true_rl_agent') and getattr(module, '__file__', None)}
    assert all(Path(path).is_relative_to(SOURCE / 'src') for path in imported.values())
    save('independent_binding_verification.json', dict(status='PASS', tasks=bindings,
        imported_module_source_roots_checked=len(imported), source_root=str(SOURCE),
        source_manifest=ref(HERE / 'source_manifest.json'), provider_requests=0, gpu_jobs=0,
        boundary='CPU configuration reconstruction and returned job/manifest hashes only; no runner or simulation invoked.'))
    asset_matches = {e['sha256']: list(e['local_matches']) for e in raw_asset['expected']}
    for e in extra_asset['matching_assets']:
        asset_matches[e['sha256']].append(e['path'])
    assert all(asset_matches.values())
    assert history['exact_scalar_matches'] == []

    completed = deepcopy(card)
    completed.update(status='STATIC_PREFLIGHT_PASS_CAPTURE_NOT_RELEASED',
        purpose='Actual current r4 source static preflight verified; no formal N5 evaluation or capture release',
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        release_gate='Root owns final N5 freeze and capture decision. GPU capacity is currently occupied by existing VLLM workers; do not disturb them.',
        remote_commands_executed='Two task preflight commands plus recorded read-only verification/transfer commands',
        prior_gpu_disk_process_snapshot_status='REFRESHED_EXISTING_VLLM_OCCUPANCY',
        prior_seed_freshness_status='INHERITED_R3_NO_TARGET_COHORT_USE_IN_588_EXPLICIT_FILES_NOT_RESCANNED',
        prior_qualification_status='COMPLETED_TRAINING_RECEIPTS_AND_CHECKPOINT_BYTES_VERIFIED_NO_NEW_QUALIFICATION')
    completed['source']['remote_verified'] = True
    completed['source']['workspace_changed_since_snapshot'] = current_changes
    completed['source']['remote_static_receipt'] = ref(remote / 'remote_static_receipt.json')
    completed['source']['initial_card_immutable_reference'] = ref(HERE / 'preflight_card.json')
    for task in completed['tasks']:
        r = result[task]
        assert r['returncode'] == 0 and r['result']['mode'] == 'preflight'
        assert r['training_binding_matches_prior'] and r['training_manifest_matches_prior'] and r['evaluation_binding_matches_prior']
        assert r['result']['provider_requests'] == r['result']['train_jobs'] == 0
        completed['commands'][task] = dict(executed_preflight_argv=r['argv'], environment=r['environment'], returncode=r['returncode'])
        completed['tasks'][task].update(server_verification_status=r['result']['status'],
            current_evaluation_binding=r['result']['evaluation_binding'], result_ref=ref(remote / task / 'result.json'),
            checkpoint_bytes_independently_verified=True, env_mapping_status='VERIFIED_CURRENT_SOURCE_SAME_PRESELECTED_ENV_MAPPING')
        if 'current_local_protocol_reference' in completed['tasks'][task]:
            completed['tasks'][task]['current_local_protocol_reference']['verification'] = 'Snapshot and remote file SHA verified'
    completed['verification'] = dict(
        source_input_verification=ref(remote / 'source_input_verification.json'),
        independent_binding_verification=ref(HERE / 'independent_binding_verification.json'),
        task_results=ref(remote / 'task_preflight_results.json'), static_env_mapping=ref(HERE / 'static_env_mapping.json'),
        asset_cache_check=ref(prior / 'remote_preflight/asset_cache_check.json'), asset_cache_supplement=ref(prior / 'asset_cache_supplement.json'),
        all_expected_asset_bytes_found=True, runtime_asset_resolver_load_verified=False,
        history_check=ref(prior / 'remote_preflight/cohort_history_check.json'), history_disambiguation=ref(prior / 'history_disambiguation.json'),
        history_files_checked=history['files_checked'], seed_nonuse_global_claim=False,
        resources_before=ref(remote / 'resources_before.json'), resources_after=ref(remote / 'resources_after.json'),
        downloaded_remote_file_hashes_checked=len(checked), old_r3_snapshot_and_inherited_card_unchanged=True,
        inherited_r3_evidence=ref(HERE / 'inherited_r3_evidence.json'), current_history_files_rescanned=0, current_asset_files_rehashed=0, extended_path_extraction_used=True)
    completed['remaining_runtime_checks'] = [
        'GPU/lease availability must be rechecked at authorized run time; current VLLM process occupancy is not ours to stop.',
        'No Kit/simulator/runtime asset resolver was started. Existing runtime asset-closure/environment and checkpoint-admission gates remain required.',
        'No fresh cohort was consumed by this work. History nonuse is limited to the recorded roots/files, not a global-server proof.',
        'Final review protocols, arithmetic responsibility, media/report/facts material roles/capacity/call budgets require separate root freeze.']
    save('completed_preflight_card.json', completed)
    print(json.dumps(dict(status=completed['status'], completed_card=ref(HERE / 'completed_preflight_card.json'),
        current_source_changes=current_changes, remote_files_verified=len(checked), all_assets_found=True,
        history_files_checked=history['files_checked'], mappings={k:v for k,v in mapping.items() if k!='evidence'}), ensure_ascii=False))


if __name__ == '__main__':
    main()
