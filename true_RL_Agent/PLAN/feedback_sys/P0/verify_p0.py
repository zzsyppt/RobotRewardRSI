"""Recompute P0 source, measurement and lineage checks offline."""
from __future__ import annotations

import base64
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PROBE_NAMES = {'initial', 'long_history', 'source_followup', 'repair', 'minimum_core',
               'long_history_repeat', 'media_text_control', 'media_native'}


def load(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_hash_rows(rows, base, hash_key='sha256', path_key='path'):
    mismatches = []
    for row in rows:
        path = base / row[path_key]
        if not path.is_file() or sha(path) != row[hash_key]:
            mismatches.append(str(path))
    return mismatches


def receipt_totals(receipts):
    if set(receipts) != PROBE_NAMES:
        raise ValueError('expected exactly eight named probe receipts')
    totals = {'calls': len(receipts), 'prompt_tokens': 0, 'completion_tokens': 0,
              'cache_hit_tokens': 0, 'cache_miss_tokens': 0, 'peak_price_estimate_usd': 0, 'elapsed_s': 0}
    for name, row in receipts.items():
        request_path = OUT / 'provider_requests' / (name + '.json')
        request = json.loads(request_path.read_text())
        if row['status'] != 'received' or row['request_sha256'] != sha(request_path):
            raise ValueError('receipt does not bind a completed matching request')
        if row['requested_model'] != request['model'] or row['requested_output_tokens'] != request['max_tokens']:
            raise ValueError('request configuration mismatch')
        usage = row['usage']
        keys = ('prompt_tokens', 'completion_tokens', 'total_tokens', 'prompt_cache_hit_tokens', 'prompt_cache_miss_tokens')
        if not all(type(usage[k]) is int and usage[k] >= 0 for k in keys):
            raise ValueError('usage is not a nonnegative integer count')
        if usage['prompt_tokens'] != usage['prompt_cache_hit_tokens'] + usage['prompt_cache_miss_tokens']:
            raise ValueError('cache count does not close')
        if usage['total_tokens'] != usage['prompt_tokens'] + usage['completion_tokens']:
            raise ValueError('total count does not close')
        if usage['completion_tokens'] > request['max_tokens']:
            raise ValueError('response exceeds requested output limit')
        for key in ('prompt_tokens', 'completion_tokens'):
            totals[key] += usage[key]
        totals['cache_hit_tokens'] += usage['prompt_cache_hit_tokens']
        totals['cache_miss_tokens'] += usage['prompt_cache_miss_tokens']
        totals['peak_price_estimate_usd'] += (usage['prompt_cache_hit_tokens'] * .006 +
            usage['prompt_cache_miss_tokens'] * .30 + usage['completion_tokens'] * 1.20) / 1e6
        if not math.isfinite(row['elapsed_s']) or row['elapsed_s'] < 0:
            raise ValueError('invalid elapsed time')
        totals['elapsed_s'] += row['elapsed_s']
    return totals


def main():
    checks = []

    def check(name, passed, detail=None):
        checks.append({'name': name, 'passed': bool(passed), 'detail': detail})

    baseline = load('source_baseline.json')
    changed = verify_hash_rows(baseline['files'], ROOT)
    manifest = json.dumps(baseline['files'], ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    check('historical_source_manifest_identity', hashlib.sha256(manifest).hexdigest() == baseline['manifest_sha256']
          and bool(baseline['captured_at']) and len(baseline['head']) == 40 and isinstance(baseline['status'], str),
          {'files': len(baseline['files']), 'captured_at': baseline['captured_at'], 'head': baseline['head'],
           'dirty_at_capture': bool(baseline['status']), 'manifest_sha256': baseline['manifest_sha256'],
           'scope': 'historical calibration snapshot; not a claim that current production files or tests are unchanged'})
    remote = load('remote_baseline.json')
    mismatches = verify_hash_rows(remote['files'], OUT / 'inventory_remote_raw')
    check('remote_raw_byte_hashes', not mismatches, {'files': len(remote['files']), 'mismatches': mismatches})
    check('historical_checkpoint_availability', len(remote['checkpoints']) == 71 and all(r['exists'] and r['matches_historical'] for r in remote['checkpoints']))
    reference = load('reference_remote_verification.json')
    check('current_v3_reference_bytes_available', len(reference['reference']) == 2 and all(r['exists'] and r['matches_expected'] for r in reference['reference']))
    parent = load('inventory_parent_remote_verification.json')
    check('formal_parent_checkpoint_and_raw_sources', len(parent['checkpoints']) == 1
          and all(r['exists'] and r['sha256'] == r['expected_sha256'] for r in parent['checkpoints'])
          and not verify_hash_rows(parent['files'], ROOT))

    inventory = load('inventory_manifest.json')
    mismatches = verify_hash_rows(inventory['sources'], ROOT)
    check('inventory_sources_still_match', not mismatches, {'sources': len(inventory['sources']), 'mismatches': mismatches})
    check('policy_and_parent_catalog_checks', bool(inventory['checks']) and all(r['passed'] for r in inventory['checks']), inventory['checks'])
    check('inventory_extractor_bound', inventory['extractor_sha256'] == sha(OUT / 'inventory_extract.py'))
    check('inventory_paths_are_repo_relative', all(not Path(s['path']).is_absolute() and
          '..' not in Path(s['path']).parts and '\\' not in s['path'] for s in inventory['sources']))
    sources = {s['source_id']: ROOT / s['path'] for s in inventory['sources']}
    policy_ids = {p['policy_id'] for p in inventory['policies']}
    relation_errors = []
    for pair in inventory['parent_candidate_pairs']:
        ledger = [json.loads(line) for line in sources[pair['ledger_source_id']].read_text(encoding='utf-8').splitlines() if line.strip()]
        generation_index, _, lane_index = pair['ledger_pointer'].strip('/').split('/')
        generation = ledger[int(generation_index)]
        trial = generation['trials'][int(lane_index)]
        train = json.loads(sources[pair['train_source_id']].read_text(encoding='utf-8'))
        valid = (pair['parent_policy_id'] == generation['parent_current_best']['checkpoint_hash']
                 and pair['candidate_policy_id'] == trial['train']['checkpoint_hash'] == train['checkpoint']['checksum']
                 and pair['parent_policy_id'] in policy_ids and pair['candidate_policy_id'] in policy_ids
                 and len(pair['selected_prompt_source_ids']) == 1
                 and 'sha256:' + sha(sources[pair['selected_prompt_source_ids'][0]]) == trial['authoring']['prompt_payload_checksum'])
        if not valid:
            relation_errors.append(pair['trial_id'])
    check('parent_edges_bound_to_ledger_train_and_author_input', not relation_errors and len(inventory['parent_candidate_pairs']) == 33,
          {'pairs': len(inventory['parent_candidate_pairs']), 'errors': relation_errors})

    behavior = load('stair_behavior_baseline.json')
    behavior_errors = []
    for label, row in behavior['policies'].items():
        origin = ROOT / row['source']['path']
        raw = json.loads(origin.read_text(encoding='utf-8'))
        ordered = raw['task_evidence']['summary']['ordered_summary']
        facts = [fact for seed in ordered['seed_summaries'] for values in seed['episode_facts_by_cell'].values() for fact in values]
        counts = dict(Counter(f['primary_reason'] for f in facts))
        valid = ('sha256:' + sha(origin) == row['source']['sha256'] and len(facts) == row['aggregate']['episodes'] == 864
                 and counts == row['aggregate']['primary_counts'] and row['identity']['checkpoint_hash'] == raw['checkpoint_safety_report']['checkpoint_hash'])
        for reason, group in row['aggregate']['by_primary'].items():
            subset = [f for f in facts if f['primary_reason'] == reason]
            valid = valid and group['n'] == len(subset) and all(
                n == sum(flag in f['diagnostic_flags'] for f in subset) for flag, n in group['diagnostic_flag_counts'].items())
        if not valid:
            behavior_errors.append(label)
    check('two_stair_policies_recomputed_from_original_facts', not behavior_errors and len(behavior['policies']) == 2, behavior_errors)

    author = load('author_evidence_baseline.json')
    check('human_annotation_fixture_bound', sha(ROOT / author['human_annotations']['path']) == author['human_annotations']['sha256'])
    author_errors = []
    for row in author['push'] + author['formal']:
        for key in ('source', 'payload', 'receipt'):
            if sha(ROOT / row[key]['path']) != row[key]['sha256']:
                author_errors.append(row['candidate_id'] + ':' + key)
        receipt = json.loads((ROOT / row['receipt']['path']).read_text(encoding='utf-8'))
        if receipt['prompt_payload_checksum'] != 'sha256:' + row['payload']['sha256']:
            author_errors.append(row['candidate_id'] + ':prompt_binding')
    check('actual_author_baseline_sources_and_inputs', not author_errors and len(author['push']) == 10 and len(author['formal']) == 36,
          {'push': len(author['push']), 'formal': len(author['formal']), 'errors': author_errors})

    historical = load('historical_prompt_usage.json')
    historical_errors = []
    calibration_path = ROOT / 'PLAN/2new_world/P5/audit_capacity/result.json'
    calibration = json.loads(calibration_path.read_text())
    for row in historical['rows']:
        if row['kind'] == 'actual_historical_authoring_attempt':
            original = json.loads((ROOT / row['receipt_path']).read_text())
            ok = (sha(ROOT / row['receipt_path']) == row['receipt_sha256'] and
                  sha(ROOT / row['prompt_path']) == row['prompt_sha256'] and
                  original['prompt_payload_checksum'] == row['receipt_prompt_checksum'] == 'sha256:' + row['prompt_sha256'] and
                  original['provider_attempt'].get('provider_usage') == row['usage'])
        else:
            original = calibration['frames'][row['name']]
            ok = (sha(calibration_path.parent / (row['name'] + '.json')) == original['source_sha256'] == row['source_sha256']
                  and row['usage'] == original['usage'])
        if not ok:
            historical_errors.append(row.get('name', row.get('prompt_path')))
    check('all_historical_prompt_hashes_match', not historical_errors and len(historical['rows']) ==
          historical['hash_match_count'] == historical['row_count'] == 60, historical_errors)
    probe = load('prompt_measurements.json')
    frames = probe['frames']
    check('five_required_frame_families', len(frames) == 5 and {r['name'] for r in frames} == {'initial', 'long_history', 'source_followup', 'repair', 'minimum_core'})
    receipts = {p.stem: json.loads(p.read_text()) for p in (OUT / 'provider_receipts').glob('*.json')}
    try:
        actual_totals = receipt_totals(receipts)
        totals_match = set(actual_totals) == set(probe['totals']) and all(
            math.isclose(actual_totals[k], probe['totals'][k], rel_tol=1e-12, abs_tol=1e-12) for k in actual_totals)
        check('all_eight_receipts_and_recomputed_totals', totals_match, actual_totals)
    except (ValueError, KeyError) as exc:
        actual_totals = None
        check('all_eight_receipts_and_recomputed_totals', False, str(exc))
    frame_errors = []
    for row in frames:
        original = ROOT / row['source_path']
        payload = OUT / 'prompt_inputs' / (row['name'] + '.json')
        request_path = OUT / 'provider_requests' / (row['name'] + '.json')
        request = json.loads(request_path.read_text())
        receipt = load('provider_receipts/' + row['name'] + '.json')
        ok = (sha(original) == row['source_sha256'] and sha(payload) == row['payload_sha256'] and
              sha(request_path) == row['request_sha256'] == receipt['request_sha256'] and
              request['messages'][1]['content'].encode() == payload.read_bytes() and
              len(payload.read_bytes()) == row['payload_bytes'] and
              row['usage'] == receipt['usage'])
        if not ok:
            frame_errors.append(row['name'])
    check('source_to_payload_to_request_to_usage_binding', not frame_errors, frame_errors)
    # Fit on the original four measurements; score the five other request identities.
    points = [(r['prompt_bytes'], r['usage']['prompt_tokens']) for r in calibration['frames'].values()]
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    slope = sum((x-mean_x)*(y-mean_y) for x, y in points) / sum((x-mean_x)**2 for x, _ in points)
    intercept = mean_y - slope * mean_x
    estimates = probe['text_estimator']
    expected_errors = [max(0, math.ceil(slope*r['payload_bytes']+intercept)) / r['usage']['prompt_tokens'] - 1 for r in frames]
    check('text_estimate_calibration_and_validation', estimates['calibration_samples'] == 4 and estimates['validation_samples'] == 5
          and estimates['calibration_source']['sha256'] == sha(ROOT / estimates['calibration_source']['path'])
          and estimates['calibration_validation_hashes_disjoint'] is True
          and math.isclose(estimates['tokens_per_user_utf8_byte'], slope)
          and math.isclose(estimates['intercept_tokens'], intercept)
          and math.isclose(estimates['max_validation_absolute_relative_error'], max(map(abs, expected_errors))),
          {'max_observed_validation_error': max(map(abs, expected_errors)), 'future_error_bound': False})
    check('cache_repeat_is_exact_request', sha(OUT / 'provider_requests/long_history.json') == sha(OUT / 'provider_requests/long_history_repeat.json')
          and receipts['long_history']['usage']['prompt_tokens'] == receipts['long_history_repeat']['usage']['prompt_tokens'])
    media = load('media_baseline.json')
    check('native_media_source_hashes', sha(ROOT / media['source_video']) == media['source_video_sha256'] and
          sha(ROOT / media['source_recordings']) == media['source_recordings_sha256'] and
          sha(OUT / media['frame_path']) == media['frame_sha256'])
    check('media_transport_does_not_pass_vision_diagnosis', probe['media_probe']['transport_status'] == 'PASS'
          and probe['media_probe']['vision_diagnosis_profile'] == 'NOT_STARTED' and probe['media_probe']['images_sent'] == 1)
    media_request = load('provider_requests/media_native.json')
    control_request = load('provider_requests/media_text_control.json')
    blocks = media_request['messages'][0]['content']
    images = [b for b in blocks if b['type'] == 'image_url']
    no_images = [b for b in control_request['messages'][0]['content'] if b['type'] == 'image_url']
    image_matches = len(images) == 1 and not no_images and hashlib.sha256(base64.b64decode(
        images[0]['image_url']['url'].split(',', 1)[1], validate=True)).hexdigest() == media['frame_sha256']
    check('sent_image_and_media_usage_delta_bound', image_matches and
          probe['media_probe']['input_token_delta'] == receipts['media_native']['usage']['prompt_tokens'] - receipts['media_text_control']['usage']['prompt_tokens']
          and probe['media_probe']['native_response'] == receipts['media_native']['response_text']
          and probe['media_probe']['no_image_response'] == receipts['media_text_control']['response_text'])

    push = load('push_cpu_regression.json')
    shared = load('shared_regressions.json')
    stair = load('stair_cpu_regression_receipt.json')
    stair_log = (OUT / stair['log']).read_text(encoding='utf-8')
    check('historical_push_repairs_regression_receipt', push['passed_without_skip'] and push['tests_run'] == 70)
    check('historical_shared_repairs_regression_receipt', shared['status'] == 'PASS' and shared['tests'] == 161 and not shared['skipped'])
    check('historical_stair_repairs_regression_receipt', sha(OUT / stair['log']) == stair['log_sha256'] and
          stair['passed_test_lines'] == 78 and 'Ran 78 tests' in stair_log and stair_log.rstrip().endswith('OK'))
    stair_data = load('stair_real_public_dev_audit.json')
    check('real_v3_fact_closure', len(stair_data) == 2 and all(r['fact_count_closure'] == 'pass' and
          r['current_source_summary_validation']['status'] == 'pass' and r['total_episode_count'] == 864 for r in stair_data))

    limits = load('resource_limits.json')['stages']['p0_baseline']
    totals = actual_totals or probe['totals']
    local_bytes = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    check('provider_cost_within_stage_budget', totals['peak_price_estimate_usd'] <= limits['provider_usd'],
          {'local_bytes': local_bytes, 'provider': totals})
    check('scope_and_next_phase_cards_present', all((OUT / p).is_file() for p in (
        'IMPLEMENTATION_PLAN_ZH.md', 'RESOURCE_SCOPE_AND_MIN_LOOP_CARD_ZH.md', 'STAIR_DATA_MAP_ZH.md',
        'PUSH_DATA_MAP_ZH.md', 'EVIDENCE_INVENTORY_ZH.md', 'PROMPT_PROVIDER_MEDIA_BASELINE_ZH.md', 'ACCEPTANCE_ZH.md', 'DEVLOG_ZH.md',
        'AUTHOR_EVIDENCE_BASELINE_ZH.md', 'AUDIT_RESPONSE_ZH.md')))
    result = {'checked_at': datetime.now(timezone.utc).isoformat(), 'profile': 'P0_BASELINE',
              'status': 'PASS' if all(r['passed'] for r in checks) else 'FAIL', 'checks': checks,
              'current_tree_differences_from_historical_snapshot': {'informational_only': True,
                  'paths': [Path(p).relative_to(ROOT).as_posix() for p in changed],
                  'scope': 'P0 historical regression receipts do not validate later production changes; each later phase must run its own checks'},
              'capabilities': {'single_image_transport': 'PASS',
                               'CORE': 'NOT_STARTED', 'VISION_diagnosis': 'NOT_STARTED', 'search_gain': 'NOT_STARTED'}}
    (OUT / 'acceptance_checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'checks': len(checks), 'failures': [r for r in checks if not r['passed']]}, ensure_ascii=False))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
