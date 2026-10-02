"""R5 fixed CPU checks/preflights only. No history scan, upload or capture dispatch.

Requires new deployed source, original controls, R4 capture card and a separately
reviewed fresh history file. This file is only being prepared, not run yet.
"""
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

BASE = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
SOURCE = BASE / 'n5_confirmation_source_r5_final'
CONTROL = BASE / 'n5_execution_20261001_r5/source_control'
PREFLIGHT = BASE / 'n5_execution_20261001_r5/cpu_preflight'
PYTHON = '/data/yuyangliu/envs/isaaclab6_validation/bin/python'
PHASE = SOURCE / 'PLAN/feedback_sys/P3NEW'


def read(path): return json.loads(Path(path).read_bytes())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path): return {'path': str(path), 'sha256': sha(path), 'size_bytes': Path(path).stat().st_size}


def save(name, value):
    with (CONTROL / name).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')


def source_check():
    actual = {p.relative_to(SOURCE).as_posix(): {'sha256': sha(p), 'size_bytes': p.stat().st_size}
              for p in SOURCE.rglob('*') if p.is_file()}
    assert actual == read(CONTROL / 'source_manifest.json'), 'Frozen source drift'
    return len(actual)


def main():
    commands = read(CONTROL / 'commands.candidate.json')
    preparation = read(CONTROL / 'source_preparation.json')
    assert commands['source_root'] == str(SOURCE) and commands['preflight_root'] == str(PREFLIGHT)
    assert commands['capture_authorized_or_executed'] is False and not PREFLIGHT.exists()
    assert sha(CONTROL / 'source_manifest.json') == preparation['source_manifest']['sha256']
    assert sha(CONTROL / 'commands.candidate.json') == preparation['commands']['sha256']
    count = source_check()
    environment = {**os.environ, 'PYTHONPATH': ':'.join(map(str, (SOURCE / 'src', SOURCE, PHASE))),
                   'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': ''}
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    # Small changed-surface checks only. Old task-visual 26 and unrelated suites
    # are not rerun; prior same-source report regressions are separately bound.
    focus, = [p for p in preparation['prior_evidence'] if p.get('kind') == 'stair_focus_index']
    assert all(sha(PHASE / name) == digest for name, digest in focus['source_files'].items())
    jobs = [('reference_contract', [PYTHON, '-B', '-m', 'unittest', 'test_reference_runner', 'test_reference_catalog', 'test_reference_blocks']),
            ('facts_report_control', [PYTHON, '-B', '-m', 'unittest', 'test_report_preparation'])]
    for task in ('stair', 'push'):
        argv = commands['tasks'][task]['preflight_argv']
        assert argv[argv.index('--mode') + 1] == 'preflight'
        jobs.append((task + '_preflight', argv))
    records = []
    for name, argv in jobs:
        started = time.monotonic()
        with (CONTROL / (name + '.stdout.txt')).open('xb') as stdout, (CONTROL / (name + '.stderr.txt')).open('xb') as stderr:
            result = subprocess.run(argv, cwd=SOURCE, env=environment, stdout=stdout, stderr=stderr, timeout=900)
        text = (CONTROL / (name + '.stderr.txt')).read_text()
        tests = re.findall(r'Ran (\d+) tests?', text)
        entry = {'name': name, 'argv': argv, 'exit_code': result.returncode, 'seconds': time.monotonic() - started,
                 'test_count': int(tests[-1]) if tests else None,
                 'environment_set': {k: environment[k] for k in ('CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE')},
                 'stdout': ref(CONTROL / (name + '.stdout.txt')), 'stderr': ref(CONTROL / (name + '.stderr.txt'))}
        save(name + '.execution.json', entry); records.append(entry)
        print(json.dumps({'step': name, 'returncode': result.returncode}), flush=True)
        if result.returncode: raise SystemExit(result.returncode)
    sys.path[:0] = [str(SOURCE / 'src'), str(SOURCE), str(PHASE)]
    from true_rl_agent.composition.cell_manifest import load_resolved_cell_manifest
    from true_rl_agent.tasks.stair_climbing.entrypoint_contract import _resolved_stair_config
    from true_rl_agent.tasks.stair_climbing.diagnostic_evaluation import _binding
    from true_rl_agent.tasks.stair_climbing.seed_policy import diagnostic_seed_namespace
    from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
    from true_rl_agent.tasks.push_recovery.seed_policy import derive_runner_seed
    history_path, previous_path = CONTROL / 'history_cohorts.before.json', CONTROL / 'r4_capture_card.readonly.json'
    history, prior_card = read(history_path), read(previous_path)
    assert not history['errors']
    rows = {}
    for task in ('stair', 'push'):
        result_path = PREFLIGHT / task / 'result.json'
        result = read(result_path); binding = result['evaluation_binding']
        assert result['mode'] == 'preflight' and result['status'] == {'stair': 'PREFLIGHT_PASS', 'push': 'PREPARED'}[task]
        assert result['checkpoint_sha256'] == 'sha256:' + prior_card['tasks'][task]['checkpoint']['sha256']
        recorded = [r for r in history['files'] if r['task_id'] == 'unitree_' + ('stair_climbing' if task == 'stair' else 'push_recovery')]
        assert binding['seed_cohort_hash'] not in {r['binding']['seed_cohort_hash'] for r in recorded}
        manifest = load_resolved_cell_manifest(PREFLIGHT / task / 'composition/resolved_cell_manifest.json')
        if task == 'stair':
            config = _resolved_stair_config(SOURCE / 'configs/tasks/stair_go2_12dof_unified_protocol_v3.json', resolved_cell_manifest=manifest)
            resolver = config.seed_resolver(); namespace = diagnostic_seed_namespace(commands['diagnostic_cohort'])
            assert _binding(config, namespace).to_public_dict() == binding and result['seed_namespace'] == namespace
            derive = lambda cohort: [resolver.derive_seed(split='public_dev', run_index=i, namespace=diagnostic_seed_namespace(cohort)) for i in range(1, 4)]
            seeds = derive(commands['diagnostic_cohort'])
            previous = {c: derive(c) for c in ('v1', 'n5_r2_20260930', 'n5_r3_20260930', 'n5_r4_20261001')}
            assert previous['n5_r4_20261001'] == prior_card['tasks'][task]['runner_seeds']
        else:
            config = PushRecoveryTaskConfig.from_manifest(manifest, eval_base_seed=commands['push_eval_base_seed'])
            assert config.seed_cohort_hash() == binding['seed_cohort_hash']
            derive = lambda seed: [derive_runner_seed(base_seed=seed, split_name='public_dev', run_index=i) for i in range(1, 4)]
            seeds = derive(commands['push_eval_base_seed'])
            previous = {str(b): derive(b) for b in (1500000000, 1600000000, 1700000000, 1800000000)}
            assert previous['1800000000'] == prior_card['tasks'][task]['runner_seeds']
        assert len(set(seeds)) == 3 and all(not set(seeds) & set(old) for old in previous.values())
        rows[task] = {'preflight_result': ref(result_path), 'binding': binding, 'runner_seeds': seeds,
                      'previous_runner_seeds': previous, 'history_rows_compared': len(recorded)}
    origins = {}
    for name in ('prepare_confirmation_materials', 'confirmation_review_coverage', 'run_confirmation_review',
                 'audit_confirmation_facts', 'prepare_facts_offline', 'prepare_confirmation_reports'):
        path = Path(importlib.import_module(name).__file__).resolve()
        assert path.is_relative_to(SOURCE); origins[name] = ref(path)
    assert source_check() == count
    save('verification.json', {'status': 'CPU_STATIC_PREFLIGHT_PASS_NOT_CAPTURE_RELEASE',
        'finished_utc': datetime.now(timezone.utc).isoformat(), 'source_files': count,
        'source_manifest': ref(CONTROL / 'source_manifest.json'), 'source_unchanged_after_execution': True,
        'commands': ref(CONTROL / 'commands.candidate.json'), 'records': records, 'tasks': rows, 'module_origins': origins,
        'history': ref(history_path), 'prior_capture_card': ref(previous_path),
        'prior_same_source_evidence': preparation['prior_evidence'], 'capture_runs': 0, 'models': 0, 'gpu_runs': 0,
        'limits': ['No capture/model authorization; no semantic qualification.', 'Historical scope is the supplied explicit scan, not all server experiments.',
                   'Real capacity, fresh resources, inherited budget/STOP state and capture-card release remain separate.']})


if __name__ == '__main__': main()
