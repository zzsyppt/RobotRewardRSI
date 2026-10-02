"""Fixed PNG candidate list and read-only dependency check; never unlinks."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
B = '/data/yuyangliu/AGENT/feedback_p3new_20260929'
GROUPS = ('push_body_grounding_dev_r2', 'push_dual_view_dev_r3')
ROOTS = [f'{B}/{name}{suffix}/video' for name in GROUPS for suffix in ('', '_export')]


def record(path):
    return {'path': str(path), 'size_bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def save(name, data):
    with (HERE / name).open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
        f.write('\n')


def main():
    receipt = json.loads((HERE / 'archive_verification.json').read_bytes())
    assert receipt['status'] == 'ARCHIVE_VERIFIED_REMOTE_UNLINK_NOT_PERFORMED'
    rows = [r for line in (HERE / 'remote_verification.jsonl').read_text().splitlines()
            if (r := json.loads(line))['record_type'] == 'verified_pair']
    assert len(rows) == receipt['verified_members'] == 3740
    assert record(HERE / 'remote_verification.jsonl')['sha256'] == receipt['remote_verification_sha256']
    candidates = []
    for r in rows:
        assert all(str(Path(r[k]).parent).replace('\\', '/') in ROOTS for k in ('source_path', 'export_path'))
        assert r['source_path'].endswith('.png') and r['export_path'].endswith('.png')
        assert r['source_stat'] == r['export_stat'] and r['source_stat']['nlink'] == 2
        candidates.append({k: r[k] for k in ('group', 'relative_path', 'source_path', 'export_path', 'sha256', 'source_stat', 'export_stat')})
    assert len({p for r in candidates for p in (r['source_path'], r['export_path'])}) == 7480
    # Reuse the prior fixed archive's exact manifest and dependency-check helpers.
    tree = ast.parse((HERE / 'archive_two_finished_png_groups.py').read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'remote' for t in n.targets))
    literal = ast.literal_eval(node.value.func.value)
    remote = literal.split('dependency_check("before_hash")', 1)[0].replace('__MANIFEST_SHA__', receipt['manifest_sha256'])
    remote += '\nexpected = json.loads(' + repr(json.dumps(candidates)) + ')\n'
    remote += r'''
dependency_check("pre_unlink_review")
for root in roots:
 actual={str(p) for p in pathlib.Path(root).rglob("*.png")}
 wanted={r[k] for r in expected for k in ("source_path","export_path") if str(pathlib.Path(r[k]).parent)==root}
 assert actual==wanted,(root,"PNG membership changed")
for r in expected:
 for key,statkey in (("source_path","source_stat"),("export_path","export_stat")):
  p=pathlib.Path(r[key]);assert stat.S_ISREG(p.lstat().st_mode)
  assert signature(p)==r[statkey],(str(p),"stat changed since verified archive stream")
current=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  args=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
  cwd=''
  try:cwd=os.readlink(p/'cwd')
  except (PermissionError,FileNotFoundError):pass
  r4=any(any(s in value for s in ('n5_confirmation_20261001_r4','n5_execution_20261001_r4','n5_confirmation_source_r4_final')) for value in args+[cwd])
  cli=any('codex' in pathlib.Path(a).name or 'run_sol_review.py' in a for a in args if a)
  if r4 or cli:
   current.append({'pid':int(p.name),'comm':(p/'comm').read_text().strip(),'r4_path_seen':r4,'cli_name_seen':cli,
                   'old_png_root_in_visible_cmdline_or_cwd':any(any(r in a for r in roots) for a in args+[cwd])})
 except (PermissionError,FileNotFoundError):pass
emit({'record_type':'pre_unlink_validation','observed_unix':time.time(),'verified_pairs':len(expected),'candidate_paths':2*len(expected),
 'four_roots':roots,'membership_exact':True,'current_file_stats_match_archive_pre_stream_and_per_member_checks':True,
 'png_content_sha_last_verified_during_archive':True,'new_png_content_rehash_performed':False,
 'visible_current_r4_or_cli_processes':current,'remote_mutations':0,
 'limit':'Visible process references only. No claim about inaccessible processes or future reads; rerun the live check at any separately authorized unlink.'})
'''
    result = subprocess.run(['ssh', 'humanoid-server', '/data/yuyangliu/envs/isaaclab6_validation/bin/python', '-B', '-'],
                            input=remote.encode(), capture_output=True, timeout=180)
    (HERE / 'pre_unlink_remote.stdout.txt').write_bytes(result.stdout)
    (HERE / 'pre_unlink_remote.jsonl').write_bytes(result.stderr)
    assert result.returncode == 0, result.stderr[-4000:].decode(errors='replace')
    checks = [json.loads(line) for line in result.stderr.decode().splitlines()]
    assert checks[-1]['record_type'] == 'pre_unlink_validation'
    assert not checks[0]['target_matches']
    save('unlink_candidates.json', {'status':'CANDIDATES_ONLY_NOT_EXECUTED','created_at_utc':datetime.now(timezone.utc).isoformat(),
         'four_roots':ROOTS,'pairs':3740,'png_paths':7480,'archive_receipt':record(HERE/'archive_verification.json'),
         'archive_sha256':receipt['archive_sha256'],'remote_latest_check':record(HERE/'pre_unlink_remote.jsonl'),
         'expected_reclaim_if_both_names_of_all_pairs_unlinked_bytes':receipt['remote_final']['expected_reclaim_only_after_both_names_unlinked_bytes'],
         'metadata_npz_responses_and_non_png_files_not_candidates':True,'remote_unlinks_performed':0,'candidates':candidates})
    save('pre_unlink_review.json', {'status':'READY_FOR_ROOT_REVIEW_NOT_UNLINK_AUTHORIZATION','observed_at_utc':datetime.now(timezone.utc).isoformat(),
         'candidate_manifest':record(HERE/'unlink_candidates.json'),'archive_receipt':record(HERE/'archive_verification.json'),
         'remote_check':record(HERE/'pre_unlink_remote.jsonl'),'checks':checks,'remote_unlinks_performed':0,
         'preserve':'All non-PNG paths, metadata, NPZ and model original responses; no operation outside the four explicit video roots.'})
    print(json.dumps({'status':'CANDIDATES_READY_NO_UNLINK','pairs':3740,'png_paths':7480,'checks':checks}),flush=True)


if __name__ == '__main__':
    main()
