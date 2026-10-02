"""Preserve the failed 491-file bundle; add one existing fixture, resume only pending CPU checks."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from deploy_cpu import HERE, SOURCE, SSH, run

ROOT = HERE.parents[4]
REMOTE = '/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_confirmation_source_r4_final_control/final'
FIXTURE = 'tests/unit/test_body_projection.py'


def read(p): return json.loads(Path(p).read_bytes())
def record(p):
    raw = Path(p).read_bytes()
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}
def save(name, value):
    with (HERE / name).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2); f.write('\n')


def main():
    initial = read(HERE / 'source_manifest.json')
    assert len(initial) == 491 and FIXTURE not in initial
    assert all(record(HERE / 'source' / n) == e for n, e in initial.items())
    fixture = record(ROOT / FIXTURE)
    target = HERE / 'source' / FIXTURE
    assert not target.exists()
    shutil.copyfile(ROOT / FIXTURE, target)
    assert record(target) == fixture
    final = {**initial, FIXTURE: fixture}
    save('source_manifest.final.json', dict(sorted(final.items())))
    with tarfile.open(HERE / 'source.final.tar.gz', 'x:gz') as archive:
        for n in sorted(final): archive.add(HERE / 'source' / n, arcname=n, recursive=False)
    save('source_delta.final.json', read(HERE / 'source_delta.json') + [
        {'path': FIXTURE, 'kind': 'added_test_fixture_dependency', 'previous': None, 'current': fixture}])
    prep = deepcopy(read(HERE / 'source_preparation.json'))
    prep.update(source_files=492, source_manifest_sha256=record(HERE / 'source_manifest.final.json')['sha256'],
                bundle=record(HERE / 'source.final.tar.gz'), added=[*prep['added'], FIXTURE],
                cpu_test_modules=['tests.unit.test_stair_body_projection'],
                initial_failure='Missing existing bare-import fixture in snapshot; no production failure. 26 task visual tests already passed and are not rerun.',
                remaining_unit_command='unittest discover -s tests/unit -p test_stair_body_projection.py')
    save('source_preparation.final.json', prep)
    save('source_completion.json', {'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'initial_manifest': record(HERE / 'source_manifest.json'), 'initial_bundle': record(HERE / 'source.tar.gz'),
        'final_manifest': record(HERE / 'source_manifest.final.json'), 'final_bundle': record(HERE / 'source.final.tar.gz'),
        'added_only': {FIXTURE: fixture}, 'existing_491_source_bytes_unchanged': True,
        'initial_source_directory_gains_one_file': True, 'production_changes': 0,
        'preserved_failure_stdout': record(HERE / 'cpu_validation.stdout.txt'),
        'already_passed_task_visual_tests_not_repeated': 26,
        'remaining': '2 Stair tests via unittest discover (bare fixture import), planned phase checks, both task preflights'})
    text = (HERE / 'remote_cpu.py').read_text()
    text = text.replace('CONTROL = BASE / "n5_confirmation_source_r4_final_control"',
                        'CONTROL = BASE / "n5_confirmation_source_r4_final_control/final"')
    old = 'jobs = [("unit_tests", [PYTHON, "-B", "-m", "unittest", *preparation["cpu_test_modules"]])]'
    new = 'jobs = [("remaining_stair_tests", [PYTHON, "-B", "-m", "unittest", "discover", "-s", str(SOURCE / "tests/unit"), "-p", "test_stair_body_projection.py"])]'
    assert text.count(old) == 1
    text = text.replace(old, new)
    compile(text, str(HERE / 'remote_cpu.final.py'), 'exec')
    (HERE / 'remote_cpu.final.py').write_text(text, encoding='utf-8', newline='\n')
    run('final_remote_control', SSH + ['/usr/bin/python3', '-'],
        f"from pathlib import Path\np=Path('{REMOTE}');p.mkdir(exist_ok=False)\n".encode())
    uploads = {'source_manifest.final.json': 'source_manifest.json', 'source_preparation.final.json': 'source_preparation.json',
               'source_delta.final.json': 'source_delta.json', 'source_completion.json': 'source_completion.json',
               'commands.candidate.json': 'commands.candidate.json', 'history_cohorts.before.json': 'history_cohorts.before.json',
               'r3_capture_card.readonly.json': 'r3_capture_card.readonly.json', 'remote_cpu.final.py': 'remote_cpu.py',
               'source.final.tar.gz': 'source.tar.gz'}
    bundle_path = HERE / 'completion_control.tar'
    with tarfile.open(bundle_path, 'x') as archive:
        for local, remote in uploads.items(): archive.add(HERE / local, arcname=remote, recursive=False)
        archive.add(target, arcname='test_body_projection.py', recursive=False)
    run('final_upload', ['scp', str(bundle_path), 'humanoid-server:' + REMOTE + '/completion_control.tar'], timeout=180)
    payload = f'''import json,hashlib,tarfile
from pathlib import Path
c=Path('{REMOTE}');s=Path('{SOURCE}')
with tarfile.open(c/'completion_control.tar') as t:
 for m in t.getmembers():assert m.isfile() and '/' not in m.name and '\\\\' not in m.name
 t.extractall(c)
initial=json.loads((c.parent/'source_manifest.json').read_bytes())
def rec(p):return {{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size_bytes':p.stat().st_size}}
assert all(rec(s/n)==e for n,e in initial.items())
f=s/'{FIXTURE}';assert not f.exists()
with f.open('xb') as out:out.write((c/'test_body_projection.py').read_bytes())
expected=json.loads((c/'source_manifest.json').read_bytes())
actual={{str(p.relative_to(s)):rec(p) for p in s.rglob('*') if p.is_file()}}
assert actual==expected and len(actual)==492
print(json.dumps({{'existing_491_bytes_unchanged':True,'added_only':'{FIXTURE}','final_source_count':492}}))
'''
    run('final_source_completion', SSH + ['/usr/bin/python3', '-'], payload.encode())
    run('final_cpu_validation', SSH + ['/data/yuyangliu/envs/isaaclab6_validation/bin/python', '-B', REMOTE + '/remote_cpu.py'], timeout=1800)
    run('final_download', ['scp', 'humanoid-server:' + REMOTE + '/verification_results.tar', str(HERE / 'verification_results.final.tar')], timeout=180)
    last = json.loads((HERE / 'final_cpu_validation.stdout.txt').read_text().strip().splitlines()[-1])
    archive = HERE / 'verification_results.final.tar'
    assert record(archive)['sha256'] == last['download_archive_sha256']
    downloaded = HERE / 'downloaded_final'; downloaded.mkdir(exist_ok=False)
    with tarfile.open(archive) as tf:
        for item in tf:
            assert item.isfile() and not Path(item.name).is_absolute() and '..' not in Path(item.name).parts
            raw = tf.extractfile(item).read()
            out = Path('\\\\?\\' + str((downloaded / item.name).absolute()))
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open('xb') as stream: stream.write(raw)
            assert record(out)['sha256'] == hashlib.sha256(raw).hexdigest()
    verification = read(downloaded / 'control/verification.json')
    assert verification['source_files'] == 492 and verification['source_unchanged_after_tests_and_preflights']
    assert all(record(HERE / 'source' / n) == e for n, e in final.items())
    receipt = {'status': 'SOURCE_FROZEN_CPU_STATIC_PREFLIGHT_PASS_NOT_CAPTURE_RELEASE',
        'source_manifest': str(HERE / 'source_manifest.final.json'),
        'source_manifest_sha256': record(HERE / 'source_manifest.final.json')['sha256'], 'source_files': 492,
        'source_bundle': str(HERE / 'source.final.tar.gz'), 'source_bundle_sha256': record(HERE / 'source.final.tar.gz')['sha256'],
        'local_source': str(HERE / 'source'), 'remote_source': SOURCE,
        'commands': str(HERE / 'commands.candidate.json'), 'commands_sha256': record(HERE / 'commands.candidate.json')['sha256'],
        'verification': str(downloaded / 'control/verification.json'),
        'verification_sha256': record(downloaded / 'control/verification.json')['sha256'],
        'verification_archive_sha256': record(archive)['sha256'],
        'already_passed_tests_initial_source': 26, 'remaining_tests_final_source': verification['test_count'],
        'tests': 26 + verification['test_count'], 'initial_import_failure_preserved': True,
        'completion_record': str(HERE / 'source_completion.json'),
        'old_491_existing_files_byte_unchanged': True, 'old_source_directory_gained_one_fixture': True,
        'final_492_full_source_verified_before_and_after_cpu': True,
        'source_unchanged_after_execution': True, 'preflight_exit_codes': {'stair': 0, 'push': 0},
        'actual_model_calls': 0, 'actual_gpu_runs': 0, 'actual_capture_runs': 0,
        'run_contract_status': 'NOT_RELEASED', 'final_capture_card_created': False,
        'known_limits': verification['limitations']}
    save('source_freeze_receipt.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__': main()
