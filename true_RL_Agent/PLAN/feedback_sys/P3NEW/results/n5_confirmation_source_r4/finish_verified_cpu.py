"""Bind root's diagnostic-message-only fix and finish pending CPU validation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile

from deploy_cpu import HERE, SOURCE, SSH, run
from complete_cpu_source import ROOT, read, record, save

RUNNER = 'PLAN/feedback_sys/P3NEW/run_confirmation_review.py'
REMOTE = '/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_confirmation_source_r4_final_control/verified'


def main():
    previous = read(HERE / 'source_manifest.final.json')
    assert len(previous) == 492 and all(record(HERE / 'source' / n) == e for n, e in previous.items())
    target = HERE / 'source' / RUNNER
    old, new = target.read_bytes(), (ROOT / RUNNER).read_bytes()
    before = b'            require((reviewer["model"], reviewer["carrier"]) == (MODEL, "deepseek_api"), "DeepSeek assignment differs")'
    after = b'            require((reviewer["model"], reviewer["carrier"]) == (MODEL, "deepseek_api"),\n                    "DeepSeek assignment model/carrier differs from identity")'
    assert old.count(before) == 1 and new == old.replace(before, after), 'Only root-approved diagnostic wording may differ'
    target.write_bytes(new)
    current = {**previous, RUNNER: record(target)}
    save('source_manifest.verified.json', dict(sorted(current.items())))
    with tarfile.open(HERE / 'source.verified.tar.gz', 'x:gz') as t:
        for n in sorted(current): t.add(HERE / 'source' / n, arcname=n, recursive=False)
    prep = read(HERE / 'source_preparation.final.json')
    prep.update(source_manifest_sha256=record(HERE / 'source_manifest.verified.json')['sha256'],
                bundle=record(HERE / 'source.verified.tar.gz'), phase_test_patterns=['test_sol_cli_carrier.py'],
                completed_tests_not_repeated={'task_visual': 26, 'stair_body_projection': 2, 'reference_runner_methods': 13},
                remaining_unit_command='Only the two previously failing assignment-message assertions')
    save('source_preparation.verified.json', prep)
    correction = {'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'CPU_PACKAGING_AND_DIAGNOSTIC_WORDING_ONLY',
        'changed_only': RUNNER, 'before': previous[RUNNER], 'after': current[RUNNER],
        'replacement_before': before.decode(), 'replacement_after': after.decode(), 'all_other_491_current_files_unchanged': True,
        'initial_manifest_preserved': record(HERE / 'source_manifest.json'), 'initial_bundle_preserved': record(HERE / 'source.tar.gz'),
        'fixture_stage_manifest_preserved': record(HERE / 'source_manifest.final.json'), 'fixture_stage_bundle_preserved': record(HERE / 'source.final.tar.gz'),
        'initial_failure_log': record(HERE / 'cpu_validation.stdout.txt'), 'wording_failure_log': record(HERE / 'final_cpu_validation.stdout.txt'),
        'production_logic_changed': False, 'model_calls': 0, 'capture_runs': 0}
    save('source_wording_correction.json', correction)
    text = (HERE / 'remote_cpu.final.py').read_text()
    text = text.replace('CONTROL = BASE / "n5_confirmation_source_r4_final_control/final"',
                        'CONTROL = BASE / "n5_confirmation_source_r4_final_control/verified"')
    text = text.replace('"PYTHONPATH": str(SOURCE / "src") + ":" + str(SOURCE),',
                        '"PYTHONPATH": str(SOURCE / "src") + ":" + str(SOURCE) + ":" + str(SOURCE / "PLAN/feedback_sys/P3NEW"),')
    a = text.index('    jobs = '); b = text.index('    for index, pattern', a)
    text = text[:a] + '''    jobs = [("remaining_assignment_methods", [PYTHON, "-B", "-m", "unittest",
             "test_reference_runner.ReferenceRunnerTests.test_missing_wrong_or_changed_assignment_never_falls_back",
             "test_reference_runner.ReferenceBlockRunnerTests.test_missing_wrong_or_changed_assignment_never_falls_back"])]
''' + text[b:]
    compile(text, str(HERE / 'remote_cpu.verified.py'), 'exec')
    (HERE / 'remote_cpu.verified.py').write_text(text, encoding='utf-8', newline='\n')
    run('verified_remote_control', SSH + ['/usr/bin/python3', '-'],
        f"from pathlib import Path\nPath('{REMOTE}').mkdir(exist_ok=False)\n".encode())
    uploads = {'source_manifest.verified.json':'source_manifest.json', 'source_preparation.verified.json':'source_preparation.json',
               'source_wording_correction.json':'source_wording_correction.json', 'commands.candidate.json':'commands.candidate.json',
               'history_cohorts.before.json':'history_cohorts.before.json', 'r3_capture_card.readonly.json':'r3_capture_card.readonly.json',
               'remote_cpu.verified.py':'remote_cpu.py', 'source.verified.tar.gz':'source.tar.gz'}
    bundle = HERE / 'verified_control.tar'
    with tarfile.open(bundle, 'x') as t:
        for local, remote in uploads.items(): t.add(HERE / local, arcname=remote, recursive=False)
        t.add(target, arcname='run_confirmation_review.py', recursive=False)
    run('verified_upload', ['scp', str(bundle), 'humanoid-server:' + REMOTE + '/verified_control.tar'], timeout=180)
    payload = f'''import json,hashlib,tarfile
from pathlib import Path
c=Path('{REMOTE}');s=Path('{SOURCE}')
with tarfile.open(c/'verified_control.tar') as t:
 for m in t.getmembers():assert m.isfile() and '/' not in m.name and '\\\\' not in m.name
 t.extractall(c)
previous=json.loads((c.parent/'final/source_manifest.json').read_bytes())
def rec(p):return {{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size_bytes':p.stat().st_size}}
assert all(rec(s/n)==e for n,e in previous.items())
target=s/'{RUNNER}';old=target.read_bytes();new=(c/'run_confirmation_review.py').read_bytes()
assert new==old.replace({before!r},{after!r})
target.write_bytes(new)
expected=json.loads((c/'source_manifest.json').read_bytes())
actual={{str(p.relative_to(s)):rec(p) for p in s.rglob('*') if p.is_file()}}
assert actual==expected and len(actual)==492
print(json.dumps({{'changed_only':'{RUNNER}','diagnostic_wording_only':True,'final_source_count':492}}))
'''
    run('verified_source_update', SSH + ['/usr/bin/python3', '-'], payload.encode())
    run('verified_cpu_validation', SSH + ['/data/yuyangliu/envs/isaaclab6_validation/bin/python', '-B', REMOTE + '/remote_cpu.py'], timeout=1800)
    run('verified_download', ['scp', 'humanoid-server:' + REMOTE + '/verification_results.tar', str(HERE / 'verification_results.verified.tar')], timeout=180)
    last = json.loads((HERE / 'verified_cpu_validation.stdout.txt').read_text().strip().splitlines()[-1])
    archive = HERE / 'verification_results.verified.tar'
    assert record(archive)['sha256'] == last['download_archive_sha256']
    downloaded = HERE / 'downloaded_verified'; downloaded.mkdir(exist_ok=False)
    with tarfile.open(archive) as tf:
        for item in tf:
            assert item.isfile() and not Path(item.name).is_absolute() and '..' not in Path(item.name).parts
            raw = tf.extractfile(item).read(); out = Path('\\\\?\\' + str((downloaded / item.name).absolute()))
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open('xb') as f: f.write(raw)
            assert record(out)['sha256'] == hashlib.sha256(raw).hexdigest()
    verification = read(downloaded / 'control/verification.json')
    assert verification['source_files'] == 492 and verification['test_count'] == 5
    assert verification['source_unchanged_after_tests_and_preflights']
    assert all(record(HERE / 'source' / n) == e for n, e in current.items())
    receipt = {'status':'SOURCE_FROZEN_CPU_STATIC_PREFLIGHT_PASS_NOT_CAPTURE_RELEASE',
        'source_manifest':{'path':str(HERE/'source_manifest.verified.json'), **record(HERE/'source_manifest.verified.json')}, 'source_manifest_sha256':record(HERE/'source_manifest.verified.json')['sha256'],
        'source_files':492, 'source_bundle':{'path':str(HERE/'source.verified.tar.gz'), **record(HERE/'source.verified.tar.gz')}, 'source_bundle_sha256':record(HERE/'source.verified.tar.gz')['sha256'],
        'local_source':str(HERE/'source'), 'remote_source':SOURCE, 'commands':str(HERE/'commands.candidate.json'),
        'commands_sha256':record(HERE/'commands.candidate.json')['sha256'], 'verification':{'path':str(downloaded/'control/verification.json'), **record(downloaded/'control/verification.json')},
        'verification_sha256':record(downloaded/'control/verification.json')['sha256'], 'verification_archive_sha256':record(archive)['sha256'],
        'tests':46, 'test_accounting':{'task_visual_initial_pass':26,'stair_projection_fixture_complete_pass':2,
             'reference_runner_prewording_pass':13,'reference_runner_corrected_two_methods_pass':2,'cli_carrier_pass':3,
             'no_repeated_passes_added':True},
        'initial_import_failure_and_later_wording_failures_preserved':True,
        'fixture_completion_record':str(HERE/'source_completion.json'), 'wording_correction_record':str(HERE/'source_wording_correction.json'),
        'prior_manifests_bundles_logs_preserved':True, 'source_directory_evolution':'Initial 491 files; added one existing fixture; then changed only root-approved runner error string. Do not claim initial whole directory is unchanged.',
        'final_492_full_source_verified_before_and_after_cpu':True, 'source_unchanged_after_execution':True,
        'preflight_exit_codes':{'stair':0,'push':0}, 'actual_model_calls':0,'actual_gpu_runs':0,'actual_capture_runs':0,
        'run_contract_status':'NOT_RELEASED','final_capture_card_created':False,'known_limits':verification['limitations']}
    save('source_freeze_receipt.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__ == '__main__':main()
