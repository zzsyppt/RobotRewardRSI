"""One-off R5 source review/snapshot. No upload, preflight, capture or model call.

Run only after root explicitly releases the final source. Review is a live
inventory/diff, not a snapshot; snapshot requires that exact reviewed file SHA.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PRIOR = HERE.parent / 'n5_confirmation_source_r4'
PHASE = 'PLAN/feedback_sys/P3NEW/'
BASE = '/data/yuyangliu/AGENT/feedback_p3new_20260929'
SOURCE = BASE + '/n5_confirmation_source_r5_final'
CAPTURE = BASE + '/n5_confirmation_20261001_r5'
EXECUTION = BASE + '/n5_execution_20261001_r5'
PREFLIGHT = EXECUTION + '/cpu_preflight'
COHORT, PUSH_SEED = 'n5_r5_20261001', 1900000000
ADDED = [PHASE + name for name in (
    'N5_RUN_CARD_R5_ZH.md', 'N5_REVIEW_ASSIGNMENT_R5.json', 'N5_SOL_CLI_POLICY_R5.json',
    'prepare_facts_offline.py', 'prepare_confirmation_reports.py', 'audit_confirmation_facts.py',
    'test_report_preparation.py', 'REPORT_PREPARATION.md')]
# r3's index lives in these existing baseline files, not a new parallel packer.
INDEX_FILES = [PHASE + 'prepare_confirmation_materials.py', PHASE + 'test_related_semantic_input.py']
ROLE_SHA = '7ce626e648a1b85b22099444ba27b49ce120f4663b08e96afd1e981f999b2c7c'
POLICY_SHA = '07993fda0d35755d8186f51a1c8ae319c9e649544497f8a6e8720d8b05fe1bc0'


def read(path):
    return json.loads(Path(path).read_bytes())


def record(path):
    raw = Path(path).read_bytes()
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}


def ref(path):
    return {'path': str(Path(path).resolve()), **record(path)}


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def inventory():
    receipt = read(PRIOR / 'source_freeze_receipt.json')
    baseline_path = Path(receipt['source_manifest']['path'])
    assert record(baseline_path) == {k: receipt['source_manifest'][k] for k in ('sha256', 'size_bytes')}
    baseline = read(baseline_path)
    assert len(baseline) == receipt['source_files']
    assert all(record(Path(receipt['local_source']) / n) == item for n, item in baseline.items())
    paths = sorted(set(baseline) | set(ADDED) | set(INDEX_FILES))
    current = {name: record(ROOT / name) for name in paths}
    assert current[PHASE + 'N5_REVIEW_ASSIGNMENT_R5.json']['sha256'] == ROLE_SHA
    assert current[PHASE + 'N5_SOL_CLI_POLICY_R5.json']['sha256'] == POLICY_SHA
    # These are exact references to prior engineering evidence, not new tests.
    evidence = []
    for relative, source_key in (
        ('engineering_report_revision_r1/stair/receipt.json', 'production_source'),
        ('engineering_report_revision_r2/push/receipt.json', 'renderer')):
        path = HERE.parent / 'n5_confirmation_execution_r4' / relative
        proof = read(path)
        source = proof[source_key]
        actual = Path(source.get('file', source.get('path')))
        assert record(actual)['sha256'] == source['sha256'], 'Prior report evidence source changed'
        evidence.append({'receipt': ref(path), 'source': ref(actual), 'new_test_execution': False})
    focus_path = HERE.parent / 'n5_confirmation_execution_r4/focus_arithmetic_navigation_r1/validation_receipt.json'
    focus = read(focus_path)
    assert focus['tests']['returncode'] == 0 and focus['tests']['passed'] == 4
    assert all(current[PHASE + name]['sha256'] == digest for name, digest in focus['source_files'].items())
    evidence.append({'kind': 'stair_focus_index', 'receipt': ref(focus_path), 'source_files': focus['source_files'],
                     'tests': focus['tests'], 'new_test_execution': False})
    return {'status': 'LIVE_SOURCE_REVIEW_ONLY_NOT_FROZEN', 'observed_utc': datetime.now(timezone.utc).isoformat(),
        'baseline_manifest': ref(baseline_path), 'baseline_receipt': ref(PRIOR / 'source_freeze_receipt.json'),
        'files': current, 'source_count': len(current),
        'delta': [{'path': n, 'kind': 'added' if n not in baseline else 'changed',
                   'before': baseline.get(n), 'after': current[n]} for n in paths if current[n] != baseline.get(n)],
        'prior_same_source_engineering_evidence': evidence,
        'index_change_files': INDEX_FILES, 'snapshot_created': False, 'capture_or_models': 0}


def commands():
    value = deepcopy(read(PRIOR / 'commands.candidate.json'))
    replacements = {value['source_root']: SOURCE, value['preflight_root']: PREFLIGHT,
                    value['capture_root']: CAPTURE, value['execution_control_root']: EXECUTION}
    def rewrite(item):
        if isinstance(item, str):
            for old, new in replacements.items(): item = item.replace(old, new)
            return item
        if isinstance(item, list): return [rewrite(i) for i in item]
        if isinstance(item, dict): return {k: rewrite(v) for k, v in item.items()}
        return item
    value = rewrite(value)
    for task, row in value['tasks'].items():
        option, new = ('--diagnostic-cohort', COHORT) if task == 'stair' else ('--eval-base-seed', str(PUSH_SEED))
        for kind in ('preflight_argv', 'capture_argv_not_executed'):
            row[kind][row[kind].index(option) + 1] = new
        assert row['preflight_argv'][row['preflight_argv'].index('--mode') + 1] == 'preflight'
        before, after = row['preflight_argv'], row['capture_argv_not_executed']
        assert len(before) == len(after)
        assert {before[i - 1] for i, (a, b) in enumerate(zip(before, after)) if a != b} == {'--mode', '--output', '--video-dir'}
        row['preflight_environment']['CUDA_VISIBLE_DEVICES'] = ''
    value.update(mode='CPU_PREPARATION_ONLY', capture_authorized_or_executed=False,
                 diagnostic_cohort=COHORT, push_eval_base_seed=PUSH_SEED)
    return value


def snapshot(review_path, review_sha):
    assert record(review_path)['sha256'] == review_sha, 'Reviewed inventory SHA differs'
    reviewed, now = read(review_path), inventory()
    assert reviewed['files'] == now['files'] and reviewed['baseline_manifest'] == now['baseline_manifest'], 'Source changed since review'
    destination = HERE / 'source'
    destination.mkdir(exist_ok=False)
    for name, expected in reviewed['files'].items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        assert record(target) == expected
    assert all(record(ROOT / n) == e for n, e in reviewed['files'].items()), 'Workspace changed during copy'
    save(HERE / 'source_manifest.json', reviewed['files'])
    save(HERE / 'source_delta.json', reviewed['delta'])
    with tarfile.open(HERE / 'source.tar.gz', 'x:gz') as archive:
        for name in sorted(reviewed['files']): archive.add(destination / name, arcname=name, recursive=False)
    save(HERE / 'commands.candidate.json', commands())
    save(HERE / 'source_preparation.json', {'status': 'SOURCE_SNAPSHOT_CPU_PREFLIGHT_PENDING_NOT_RELEASED',
        'reviewed_inventory': ref(review_path), 'source_manifest': ref(HERE / 'source_manifest.json'),
        'source_bundle': ref(HERE / 'source.tar.gz'), 'source_files': len(reviewed['files']),
        'local_source': str(destination), 'remote_source': SOURCE, 'remote_control': EXECUTION + '/source_control',
        'commands': ref(HERE / 'commands.candidate.json'), 'prior_evidence': reviewed['prior_same_source_engineering_evidence'],
        'run_contract_status': 'DRAFT_NOT_RELEASED', 'capture_card_created': False,
        'model_calls': 0, 'gpu_runs': 0, 'capture_runs': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('review', 'snapshot'))
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--review-sha')
    args = parser.parse_args()
    if args.action == 'review': save(args.review, inventory())
    else:
        assert args.review_sha, 'Explicit reviewed inventory SHA required'
        snapshot(args.review, args.review_sha)
