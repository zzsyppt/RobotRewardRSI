"""Bind the returned static r3 evidence; never authorize capture."""
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
    raw_asset = load(remote / 'asset_cache_check.json')
    extra_asset = load(HERE / 'asset_cache_supplement.json')
    history = load(HERE / 'history_disambiguation.json')
    expected_remote_prefix = '/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_preflight_r3/'
    checked = []
    for entry in receipt['files']:
        assert entry['path'].startswith(expected_remote_prefix)
        relative = entry['path'][len(expected_remote_prefix):]
        p = Path('\\\\?\\' + str(remote)) / relative
        assert sha(p) == entry['sha256'], relative
        checked.append(relative)
    assert all(sha(SOURCE / p) == h for p, h in source_manifest['files'].items())
    current_changes = [p for p, h in source_manifest['files'].items() if sha(PROJECT / p) != h]
    old = HERE.parent / 'n5_source_preflight_r2'
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
    asset_matches = {e['sha256']: list(e['local_matches']) for e in raw_asset['expected']}
    for e in extra_asset['matching_assets']:
        asset_matches[e['sha256']].append(e['path'])
    assert all(asset_matches.values())
    assert history['exact_scalar_matches'] == []

    completed = deepcopy(card)
    completed.update(status='STATIC_PREFLIGHT_PASS_CAPTURE_NOT_RELEASED',
        purpose='Actual current r3 source static preflight verified; no formal N5 evaluation or capture release',
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        release_gate='Root owns final N5 freeze and capture decision. GPU capacity is currently occupied by existing VLLM workers; do not disturb them.',
        remote_commands_executed='Two task preflight commands plus recorded read-only verification/transfer commands',
        prior_gpu_disk_process_snapshot_status='REFRESHED_EXISTING_VLLM_OCCUPANCY',
        prior_seed_freshness_status='NO_TARGET_COHORT_USE_FOUND_IN_EXPLICIT_HISTORY_SCOPE',
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
        task_results=ref(remote / 'task_preflight_results.json'), static_env_mapping=ref(HERE / 'static_env_mapping.json'),
        asset_cache_check=ref(remote / 'asset_cache_check.json'), asset_cache_supplement=ref(HERE / 'asset_cache_supplement.json'),
        all_expected_asset_bytes_found=True, runtime_asset_resolver_load_verified=False,
        history_check=ref(remote / 'cohort_history_check.json'), history_disambiguation=ref(HERE / 'history_disambiguation.json'),
        history_files_checked=history['files_checked'], seed_nonuse_global_claim=False,
        resources_before=ref(remote / 'resources_before.json'), resources_after=ref(remote / 'resources_after.json'),
        downloaded_remote_file_hashes_checked=len(checked), old_r2_snapshot_and_inherited_card_unchanged=True,
        original_local_extraction_limit_recovery=ref(HERE / 'local_extraction_completion.json'))
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
