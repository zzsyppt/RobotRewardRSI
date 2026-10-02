"""One r3 CPU/static preflight; no simulation, model, training, or rollout call."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

BASE = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
INPUT = BASE / 'n5_preflight_r3_inputs'
SOURCE = BASE / 'source_n5_preflight_r3'
OUTPUT = BASE / 'n5_preflight_r3'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ref(path):
    path = Path(path)
    return dict(path=str(path), sha256=sha(path), size_bytes=path.stat().st_size)


def write(name, value):
    path = OUTPUT / name
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    return ref(path)


def resources():
    def command(argv):
        p = subprocess.run(argv, text=True, capture_output=True, timeout=20)
        return dict(argv=argv, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    process_rows = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            comm = (p / 'comm').read_text().strip()
            args = (p / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            scripts = [Path(a).name for a in args if a.endswith('.py')]
            if any(word in ' '.join([comm, *scripts]).lower() for word in ('python', 'isaac', 'kit', 'train', 'replay')):
                process_rows.append(dict(pid=int(p.name), comm=comm, script_basenames=scripts))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    d = shutil.disk_usage(BASE)
    return dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),
        disk=dict(total_bytes=d.total, used_bytes=d.used, free_bytes=d.free),
        gpu=command(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader']),
        compute_processes=command(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,process_name,used_memory', '--format=csv,noheader']),
        processes=process_rows, process_scope='Current /proc python/Isaac/kit/train/replay names; no arguments or environment recorded')


def cohort_history():
    roots = [Path('/data/yuyangliu/AGENT') / n for n in (
        'feedback_p2_20260923', 'feedback_p2_audit_20260924', 'feedback_p3_20260924',
        'feedback_p3r_20260929', 'feedback_p3new_20260929', 'true_RL_Agent/outputs')]
    tokens = ('public_dev_diagnostic_v1', '1500000000', '1500010000', '1500010001', '1500010002')
    checked, matches, skipped_large, errors = [], [], [], []
    for root in roots:
        if not root.exists():
            continue
        for folder, dirs, names in os.walk(root):
            dirs[:] = [d for d in dirs if not (d.startswith('source') or d in (
                '.git', '__pycache__', 'materials', 'reference_source', 'eval_template', 'node_modules', 'video', 'videos')
                or 'preflight' in d.lower())]
            for name in names:
                if name not in ('summary.json', 'public_dev_result.json', 'manifest.json', 'result.json', 'numeric_evidence_manifest.json'):
                    continue
                p = Path(folder) / name
                if p.stat().st_size > 10 * 1024**2:
                    skipped_large.append(str(p)); continue
                try:
                    raw = p.read_bytes()
                    value = json.loads(raw)
                    checked.append(dict(path=str(p), sha256=hashlib.sha256(raw).hexdigest()))
                    found = [t for t in tokens if t in raw.decode('utf-8')]
                    if found:
                        matches.append(dict(path=str(p), tokens=found, status=value.get('status') if isinstance(value, dict) else None))
                except (OSError, ValueError) as exc:
                    errors.append(dict(path=str(p), error=type(exc).__name__))
    return dict(checked_at_utc=datetime.now(timezone.utc).isoformat(), scope_roots=list(map(str, roots)),
        scope='Runtime summary/result/manifest names; omit source/material/reference copies and preflight trees. Text matches are candidates, not proof of rollout execution.',
        tokens=list(tokens), files_checked=len(checked), checked_files=checked, matching_runtime_candidates=matches,
        skipped_large=skipped_large, read_errors=errors, global_server_nonuse_claim=False)


def assets():
    expected = []
    for name in ('unitree_go2_12dof_isaaclab_v1', 'unitree_g1_37dof_isaaclab_v2'):
        path = SOURCE / 'configs/robots' / (name + '.json')
        profile = json.loads(path.read_bytes())
        closure = profile['runtime_facts']['asset_closure']
        for entry in closure['entries']:
            expected.append(dict(profile=name, profile_ref=ref(path), **entry, local_matches=[]))
    # Cached copies only. No remote asset fetch, simulator, or Kit startup.
    roots = [Path('/data/yuyangliu/cache/xdg/ov'), Path('/data/yuyangliu/.cache/isaac_sim_6_validation')]
    sizes = {e['size_bytes'] for e in expected}
    candidates, errors = 0, []
    for root in roots:
        if not root.exists():
            continue
        for folder, _, names in os.walk(root):
            for name in names:
                p = Path(folder) / name
                try:
                    if p.stat().st_size not in sizes:
                        continue
                    candidates += 1
                    digest = sha(p)
                    for e in expected:
                        if e['sha256'] == digest:
                            e['local_matches'].append(str(p))
                except (OSError, PermissionError) as exc:
                    errors.append(dict(path=str(p), error=type(exc).__name__))
    return dict(expected=expected, searched_cache_roots=list(map(str, roots)), candidate_size_files_hashed=candidates,
        errors=errors, all_declared_asset_bytes_found=all(e['local_matches'] for e in expected),
        boundary='Checks cached asset closure bytes against current robot profiles; does not start resolver or prove runtime loadability.')


def main():
    assert not SOURCE.exists() and not OUTPUT.exists(), 'fresh r3 source/output required'
    card = json.loads((INPUT / 'preflight_card.json').read_bytes())
    manifest = json.loads((INPUT / 'source_manifest.json').read_bytes())
    assert sha(INPUT / 'source.tar.gz') == card['source']['bundle']['sha256']
    assert sha(INPUT / 'source_manifest.json') == card['source']['manifest']['sha256']
    SOURCE.mkdir()
    with tarfile.open(INPUT / 'source.tar.gz') as archive:
        entries = archive.getmembers()
        assert {e.name for e in entries} == set(manifest['files'])
        for e in entries:
            p = SOURCE / e.name
            assert e.isfile() and not e.name.startswith('/') and '..' not in Path(e.name).parts
            assert hashlib.sha256(archive.extractfile(e).read()).hexdigest() == manifest['files'][e.name]
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open('xb') as stream:
                stream.write(archive.extractfile(e).read())
    assert all(sha(SOURCE / p) == h for p, h in manifest['files'].items())
    OUTPUT.mkdir()
    before = write('resources_before.json', resources())
    inputs = {}
    for task, row in card['tasks'].items():
        inputs[task] = {}
        for flag, expected in row['inherited_source_input_references'].items():
            p = Path(expected['path'])
            if flag == '--protocol':
                p = Path(row['current_local_protocol_reference']['path'])
            actual = ref(p)
            assert actual['sha256'] == expected['sha256'], (task, flag, actual)
            inputs[task][flag] = dict(actual, matches_inherited_sha256=True)
        summary_key = '--train-summary' if task == 'stair' else '--checkpoint-summary'
        summary = json.loads(Path(inputs[task][summary_key]['path']).read_bytes())
        inputs[task]['training_receipt_fields'] = {k: summary.get(k) for k in (
            'status', 'subject_kind', 'subject_id', 'resolved_cell_manifest_hash', 'training_binding_hash', 'checkpoint')}
        assert summary['status'] == 'pass' and summary['subject_kind'] == 'reference'
    input_ref = write('source_input_verification.json', inputs)
    task_results = {}
    for task, command in card['commands'].items():
        argv = command['preflight_argv_not_executed']
        assert argv[argv.index('--mode') + 1] == 'preflight'
        assert argv[1].startswith(str(SOURCE) + '/scripts/')
        assert argv[argv.index('--output') + 1] == str(OUTPUT / task)
        env = os.environ.copy()
        env.update(command['environment'])
        with (OUTPUT / (task + '.stdout.txt')).open('xb') as stdout, (OUTPUT / (task + '.stderr.txt')).open('xb') as stderr:
            p = subprocess.run(argv, cwd=SOURCE, env=env, stdout=stdout, stderr=stderr, timeout=180)
        result_path = OUTPUT / task / 'result.json'
        result = json.loads(result_path.read_bytes()) if result_path.exists() else None
        task_results[task] = dict(argv=argv, environment=command['environment'], returncode=p.returncode, result=result,
            stdout=ref(OUTPUT / (task + '.stdout.txt')), stderr=ref(OUTPUT / (task + '.stderr.txt')))
        if result is not None:
            task_results[task]['result_ref'] = ref(result_path)
            task_results[task]['training_binding_matches_prior'] = result.get('preserved_training_binding_hash') == card['tasks'][task]['inherited_training_binding_hash']
            task_results[task]['training_manifest_matches_prior'] = result.get('source_training_manifest_hash') == card['tasks'][task]['inherited_source_training_manifest_hash']
            task_results[task]['evaluation_binding_matches_prior'] = result.get('evaluation_binding') == card['tasks'][task]['prior_evaluation_binding_reference_not_current_proof']
    tasks_ref = write('task_preflight_results.json', task_results)
    asset_ref = write('asset_cache_check.json', assets())
    history_ref = write('cohort_history_check.json', cohort_history())
    after = write('resources_after.json', resources())
    assert all(sha(SOURCE / p) == h for p, h in manifest['files'].items())
    refs = [ref(p) for p in OUTPUT.rglob('*') if p.is_file()]
    receipt = dict(status='STATIC_PREFLIGHT_ONLY_NOT_CAPTURE_RELEASE', source_root=str(SOURCE),
        source_manifest=ref(INPUT / 'source_manifest.json'), source_bundle=ref(INPUT / 'source.tar.gz'),
        source_file_count=len(manifest['files']), all_remote_source_hashes_equal_before_after=True,
        source_input_verification=input_ref, task_results=tasks_ref, assets=asset_ref, cohort_history=history_ref,
        resources_before=before, resources_after=after, files=refs, remote_script=ref(Path(__file__)),
        provider_requests=0, gpu_jobs=0, training_jobs=0, captures_started=0, formal_n5_evaluation_started=False,
        created_at_utc=datetime.now(timezone.utc).isoformat())
    write('remote_static_receipt.json', receipt)
    with tarfile.open(INPUT / 'remote_results.tar', 'w') as archive:
        for p in OUTPUT.rglob('*'):
            if p.is_file():
                archive.add(p, arcname=p.relative_to(OUTPUT).as_posix(), recursive=False)
    print(json.dumps(dict(receipt=ref(OUTPUT / 'remote_static_receipt.json'), results_tar=ref(INPUT / 'remote_results.tar'),
        tasks={k:dict(returncode=v['returncode'], status=(v['result'] or {}).get('status')) for k,v in task_results.items()})))


if __name__ == '__main__':
    main()
