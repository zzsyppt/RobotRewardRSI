"""Deploy a new source tree and inspect imports/help only; never use capture data."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

BASE = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
INPUT = BASE / 'n5_postprocess_source_r1_inputs'
SOURCE = BASE / 'n5_postprocess_source_r1'
PYTHON = '/data/yuyangliu/envs/isaaclab6_validation/bin/python'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    path = Path(path)
    return dict(path=str(path), sha256=sha(path), size_bytes=path.stat().st_size)


def save(name, value):
    with (INPUT / name).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    assert not SOURCE.exists(), 'independent source root must be new'
    receipt = json.loads((INPUT / 'local_freeze_receipt.json').read_bytes())
    manifest = json.loads((INPUT / 'source_manifest.json').read_bytes())
    assert sha(INPUT / 'source_manifest.json') == receipt['source_manifest']['sha256']
    assert sha(INPUT / 'source.tar.gz') == receipt['bundle']['sha256']
    old_deps = json.loads((INPUT / 'remote_dependency_check.stdout.json').read_bytes())
    assert all(sha(row['path']) == row['sha256'] for row in old_deps['files'])
    SOURCE.mkdir()
    with tarfile.open(INPUT / 'source.tar.gz') as archive:
        members = archive.getmembers()
        assert {m.name for m in members} == set(manifest['files'])
        for entry in members:
            assert entry.isfile() and not entry.name.startswith('/') and '..' not in Path(entry.name).parts
            raw = archive.extractfile(entry).read()
            assert hashlib.sha256(raw).hexdigest() == manifest['files'][entry.name]['sha256']
            target = SOURCE / entry.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(raw)
    expected_env = dict(PYTHONPATH=str(SOURCE / 'src') + ':' + str(SOURCE / 'PLAN/feedback_sys/P3NEW')
        + ':' + str(SOURCE / 'PLAN/feedback_sys/P3R') + ':' + str(SOURCE / 'PLAN/feedback_sys/P3'),
        PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
    env = {**os.environ, **expected_env}
    checks = INPUT / 'checks'
    checks.mkdir(exist_ok=False)
    commands = []

    def check(name, argv):
        result = subprocess.run(argv, cwd=SOURCE, env=env, capture_output=True, timeout=90)
        stdout, stderr = checks / (name + '.stdout.txt'), checks / (name + '.stderr.txt')
        stdout.write_bytes(result.stdout)
        stderr.write_bytes(result.stderr)
        commands.append(dict(name=name, argv=argv, cwd=str(SOURCE), environment=expected_env,
            returncode=result.returncode, stdout=ref(stdout), stderr=ref(stderr)))

    helpers = ('extract_dual_view_evidence.py', 'prepare_n5_media_selection.py',
        'prepare_native_observer.py', 'export_confirmation_raw.py', 'prepare_review_task_contract.py',
        'prepare_confirmation_materials.py', 'prepare_confirmation_media.py',
        'verify_stair_confirmation_arithmetic.py', 'verify_push_confirmation_arithmetic.py',
        'run_confirmation_review.py', 'run_sol_review.py', 'audit_delivery_offline.py')
    for name in helpers:
        check(name.removesuffix('.py') + '_help',
              [PYTHON, '-B', str(SOURCE / 'PLAN/feedback_sys/P3NEW' / name), '--help'])
    for relative in ('PLAN/feedback_sys/P3R/author_ceiling_probe.py', 'PLAN/feedback_sys/P3/author_experiment.py'):
        check(Path(relative).stem + '_help', [PYTHON, '-B', str(SOURCE / relative), '--help'])
    import_script = INPUT / 'check_imports.py'
    import_script.write_text('''from pathlib import Path
import importlib,json,sys
root=Path(sys.argv[1]).resolve()
names=('true_rl_agent','prepare_confirmation_materials','prepare_confirmation_media',
       'prepare_native_observer','prepare_text_author_smoke','prepare_native_author_smoke',
       'prepare_native_author_final_slots','audit_delivery_offline',
       'confirmation_context','author_ceiling_probe','author_experiment')
modules={name:importlib.import_module(name) for name in names}
modules['confirmation_context._p3']=modules['confirmation_context']._p3(root)
modules['author_ceiling_probe.load_p3']=modules['author_ceiling_probe'].load_p3(root/'PLAN/feedback_sys/P3')
paths={name:str(Path(module.__file__).resolve()) for name,module in modules.items()}
assert all(Path(path).is_relative_to(root) for path in paths.values()),paths
for name in ('prepare_native_author_smoke','prepare_native_author_final_slots','prepare_text_author_smoke'):
 module=modules[name]
 for attr in ('ROOT','SOURCE_ROOT'):
  if hasattr(module,attr): assert Path(getattr(module,attr)).resolve()==root,(name,attr)
runtime={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items()
         if name.startswith('true_rl_agent') and getattr(module,'__file__',None)}
assert all(Path(path).is_relative_to(root/'src') for path in runtime.values())
print(json.dumps(dict(status='PASS',source_root=str(root),module_files=paths,
    runtime_imports_checked=len(runtime),provider_instances_created=0,
    provider_requests=0,gpu_jobs=0,new_n5_results_read=False),indent=2))
''', encoding='utf-8')
    check('real_file_imports', [PYTHON, '-B', str(import_script), str(SOURCE)])
    actual = {p.relative_to(SOURCE).as_posix(): sha(p) for p in SOURCE.rglob('*') if p.is_file()}
    assert actual == {p: row['sha256'] for p, row in manifest['files'].items()}
    assert all(sha(row['path']) == row['sha256'] for row in old_deps['files'])
    ok = all(row['returncode'] == 0 for row in commands)
    save('remote_deployment_receipt.json', dict(status='CPU_SOURCE_READY' if ok else 'CPU_CHECK_FAILED',
        created_at_utc=datetime.now(timezone.utc).isoformat(), source_root=str(SOURCE),
        source_manifest=ref(INPUT / 'source_manifest.json'), source_bundle=ref(INPUT / 'source.tar.gz'),
        source_files_verified=len(actual), all_source_bytes_unchanged_after_checks=True,
        old_three_dependency_originals_unchanged=True, commands=commands,
        check_import_script=ref(import_script), deployment_script=ref(Path(__file__)),
        dependency_supplement_provenance=ref(INPUT / 'dependency_supplement_provenance.json'),
        running_capture_source_never_opened_or_modified=True, new_n5_results_read=False,
        provider_requests=0, gpu_jobs=0, training_jobs=0, captures_started=0))
    result_tar = INPUT / 'cpu_verification_results.tar'
    with tarfile.open(result_tar, 'w') as archive:
        for p in [INPUT / 'remote_deployment_receipt.json', import_script, *sorted(checks.iterdir())]:
            archive.add(p, arcname=p.relative_to(INPUT).as_posix(), recursive=False)
    print(json.dumps(dict(status='CPU_SOURCE_READY' if ok else 'CPU_CHECK_FAILED',
        source_root=str(SOURCE), source_files=len(actual), command_count=len(commands),
        failed_checks=[row['name'] for row in commands if row['returncode']],
        receipt=ref(INPUT / 'remote_deployment_receipt.json'), results_tar=ref(result_tar))))


if __name__ == '__main__':
    main()
