"""Preserve exact CPU commands, stdout/stderr, time and actual process peak memory."""
import argparse
from contextlib import redirect_stdout
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location('commands', Path(__file__).parents[1]/'P2b1/run_final.py')
commands = importlib.util.module_from_spec(spec); spec.loader.exec_module(commands)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['tests', 'terrain', 'real', 'final_audit', 'raw_audit'], required=True); a = p.parse_args()
    required = [ROOT/'outputs/p2c3/historical_02/stdout.json', *[ROOT/s['manifest_relative_path']
        for s in json.loads((Path(__file__).parents[1]/'P2a/source_lock.json').read_text())['sources']]]
    if any(not p.is_file() for p in required): raise FileNotFoundError('P2c3 acceptance requires historical artifacts; no skips')
    commands.OUT = a.output.resolve(); commands.OUT.mkdir(parents=True, exist_ok=False)
    if a.mode in ('tests', 'terrain'):
        jobs = [('terrain', ['-m', 'unittest', 'tests.unit.test_terrain_reference', 'tests.unit.test_terrain_offline', '-v']),
                ('compatibility', ['-m', 'unittest', 'tests.unit.test_joint_limits', 'tests.unit.test_trajectory_cycles',
                    'tests.unit.test_stair_cycle_supplement', 'tests.unit.test_stair_trajectory_report',
                    'tests.unit.test_stair_trajectory_pipeline', 'tests.unit.test_trajectory_card',
                    'tests.unit.test_trajectory_measurement', 'tests.unit.test_trajectory_cohort',
                    'tests.unit.test_push_trajectory_pipeline', 'tests.unit.test_push_trajectory', '-v']),
                ('frozen_artifacts', ['PLAN/feedback_sys/NEW_SYS/P2c1/verify_compatibility.py'])]
        if a.mode == 'terrain': jobs = jobs[:1]
    elif a.mode in ('final_audit', 'raw_audit'):
        jobs = [('terrain', ['-m', 'unittest', 'tests.unit.test_terrain_reference', 'tests.unit.test_terrain_offline', '-v']),
                ('independent_spotcheck', ['PLAN/feedback_sys/NEW_SYS/P2c3/independent_spotcheck.py'])]
        if a.mode == 'raw_audit': jobs = []
        for sid in ('k1', 'k2', 'k3'):
            jobs.append(('oracle_'+sid, ['PLAN/feedback_sys/NEW_SYS/P2c3/audit_raw.py', '--source', sid,
                '--directory', str(ROOT/'outputs/p2c3/real_03'/sid), '--audit-output', str(commands.OUT/(sid+'_audit.json'))]))
    else:
        jobs = []
        for s in json.loads((Path(__file__).parents[1]/'P2a/source_lock.json').read_text())['sources']:
            sid = s['source_id']; out = commands.OUT/sid
            jobs += [('build_'+sid, ['scripts/build_stair_terrain_supplement.py', '--manifest', str(ROOT/s['manifest_relative_path']),
                '--manifest-sha256', s['manifest_sha256'], '--source-id', sid, '--output', str(out),
                '--qualification', 'outputs/p2c3/historical_02/stdout.json', '--qualification-sha256',
                '84aecf9e412be7578ee567ca65cc1c6ff48351f8f45c1fb789a1a61913e8a48f',
                '--base-public', str(ROOT/'outputs/p2a'/sid/'public.json')]),
                ('oracle_'+sid, ['PLAN/feedback_sys/NEW_SYS/P2c3/audit_raw.py', '--source', sid, '--directory', str(out)])]
    receipts = []
    for name, argv in jobs:
        with (commands.OUT/(name+'.wrapper.txt')).open('x', encoding='utf-8') as f, redirect_stdout(f):
            result = commands.run(name, argv)
        receipts.append(result)
        (commands.OUT/'command_receipts.json').write_text(json.dumps(receipts, indent=2)+'\n', encoding='utf-8')
        print(name, 'exit', result['exit_code'], 'seconds', round(result['wall_seconds'], 3), flush=True)
        if result['exit_code']:
            print((commands.OUT/(name+'.stderr.txt')).read_text(encoding='utf-8'))
            sys.exit(result['exit_code'])
