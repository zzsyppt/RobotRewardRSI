"""One-shot exact PNG unlink. Uses the existing conditional cleanup authorization after verified archival and --execute."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

from prepare_unlink_review import B, HERE, ROOTS

CANDIDATE_SHA = 'PENDING_ARCHIVE_COMPLETION_DO_NOT_EXECUTE'
ARCHIVE_SHA = 'PENDING_ARCHIVE_COMPLETION_DO_NOT_EXECUTE'
REMOTE_CONTROL = B + '/cleanup_20261001_stair_cold_r2/unlink_ended_stair_png_once'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('No action taken. --execute may be used only after the archive and fixed-scope review are complete under the existing conditional authorization.')
    raw = (HERE / 'unlink_candidates.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == CANDIDATE_SHA
    candidate = json.loads(raw)
    assert candidate['four_roots'] == ROOTS and candidate['pairs'] == 6802 and candidate['png_paths'] == 13604
    assert candidate['archive_sha256'] == ARCHIVE_SHA
    locator_raw = (HERE / 'archive_restore_locator.json').read_bytes()
    locator = json.loads(locator_raw)
    assert locator['archive_sha256'] == ARCHIVE_SHA and locator['candidate_manifest_sha256'] == CANDIDATE_SHA
    assert Path(locator['archive_on_windows_host']).stat().st_size == locator['archive_size_bytes']
    assert len(locator['members']) == 6802
    # Reuse the archive's exact four-root membership, stat and visible-dependency helpers.
    tree = ast.parse((HERE / 'archive_two_finished_png_groups.py').read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'remote' for t in n.targets))
    literal = ast.literal_eval(node.value.func.value)
    remote = literal.split('dependency_check("before_hash")', 1)[0].replace('__MANIFEST_SHA__', locator['original_manifest_sha256'])
    remote += '\nexpected=json.loads(' + repr(json.dumps(candidate['candidates'])) + ')\n'
    remote += 'locator_raw=' + repr(locator_raw) + '\n'
    remote += 'control=pathlib.Path(' + repr(REMOTE_CONTROL) + ')\n'
    remote += 'preserved=json.loads(' + repr((HERE/'non_png_before.json').read_text()) + ')\n'
    remote += r'''
import shutil
def check_preserved():
 actual={str(p) for root in roots for p in pathlib.Path(root).rglob('*') if p.is_file() and p.suffix!='.png'}
 assert actual=={r['path'] for r in preserved['non_png_files']},'non-PNG membership changed'
 for r in preserved['non_png_files']+preserved['capture_completion']:
  p=pathlib.Path(r['path']);assert p.resolve()==p and p.is_relative_to(B)
  assert digest(p)==r['sha256'],(str(p),'preserved content changed')
check_preserved()
dependency_check("immediately_before_unlink")
assert len(expected)==6802 and len({r[k] for r in expected for k in ('source_path','export_path')})==13604
for root in roots:
 actual={str(p) for p in pathlib.Path(root).rglob('*.png')}
 wanted={r[k] for r in expected for k in ('source_path','export_path') if str(pathlib.Path(r[k]).parent)==root}
 assert actual==wanted,(root,'PNG membership changed')
for r in expected:
 for key,statkey in (('source_path','source_stat'),('export_path','export_stat')):
  p=pathlib.Path(r[key]);assert str(p.parent) in roots and p.suffix=='.png'
  assert p.resolve()==p and p.is_relative_to(B)
  assert stat.S_ISREG(p.lstat().st_mode) and signature(p)==r[statkey]
 assert r['source_stat']==r['export_stat'] and r['source_stat']['nlink']==2
assert control.resolve().is_relative_to(B) and control.parent==B/'cleanup_20261001_stair_cold_r2'
control.mkdir(parents=True,exist_ok=False)
with (control/'archive_restore_locator.json').open('xb') as f:
 f.write(locator_raw);f.flush();os.fsync(f.fileno())
assert hashlib.sha256((control/'archive_restore_locator.json').read_bytes()).hexdigest()==hashlib.sha256(locator_raw).hexdigest()
before=shutil.disk_usage(B).free
deleted=0;complete_pairs=0;blocks=0
with (control/'deleted_png_names.jsonl').open('x',encoding='utf-8') as journal:
 try:
  for r in expected:
   p=pathlib.Path(r['source_path']);q=pathlib.Path(r['export_path'])
   assert signature(p)==r['source_stat'] and signature(q)==r['export_stat']
   for target in (p,q):
    target.unlink()
    deleted+=1
    journal.write(json.dumps({'path':str(target),'original_sha256':r['sha256']})+'\n');journal.flush()
   complete_pairs+=1;blocks+=r['source_stat']['allocated_bytes']//512
  os.fsync(journal.fileno())
 finally:
  after=shutil.disk_usage(B).free
  receipt={'status':'UNLINK_COMPLETE' if deleted==13604 and complete_pairs==6802 else 'PARTIAL_STOP_NO_AUTOMATIC_RETRY',
   'finished_unix':time.time(),'four_roots':roots,'deleted_png_names':deleted,'completed_inode_pairs':complete_pairs,
   'allocated_512B_blocks_of_completed_pairs':blocks,'allocated_bytes_of_completed_pairs':blocks*512,
   'disk_free_before_bytes':before,'disk_free_after_bytes':after,
   'disk_free_net_delta_bytes':after-before,
   'space_limit':'Net free-space change also includes concurrent filesystem activity; allocated blocks are attribution, not a claim of exact immediate free-space gain.',
   'archive_locator_sha256':hashlib.sha256(locator_raw).hexdigest(),'directories_removed':0,
   'non_png_paths_unlinked':0,'metadata_npz_original_responses_unlinked':0,
   'root_visible_dependency_check_only':True,'no_automatic_retry':True}
  with (control/'unlink_receipt.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2);f.write('\n')
assert deleted==13604 and all(not pathlib.Path(r[k]).exists() for r in expected for k in ('source_path','export_path'))
assert all(pathlib.Path(root).is_dir() for root in roots)
check_preserved()
emit({'record_type':'preservation_check','non_png_contents_and_capture_receipts_unchanged':True,'non_png_files_rehashed':len(preserved['non_png_files']),'capture_receipts_rehashed':len(preserved['capture_completion'])})
emit({'record_type':'unlink_finished',**receipt})
print(json.dumps(receipt),flush=True)
'''
    # Exclusive local outputs also prevent accidentally rerunning the operation.
    with (HERE / 'unlink_execution.stdout.json').open('xb') as out, (HERE / 'unlink_execution.stderr.jsonl').open('xb') as err:
        result = subprocess.run(['ssh','humanoid-server','/data/yuyangliu/envs/isaaclab6_validation/bin/python','-B','-'],
                                input=remote.encode(),stdout=out,stderr=err,timeout=300)
    assert result.returncode == 0, 'Inspect preserved logs and remote journal; do not retry automatically.'
    receipt = json.loads((HERE / 'unlink_execution.stdout.json').read_bytes())
    assert receipt['status']=='UNLINK_COMPLETE' and receipt['deleted_png_names']==13604
    print(json.dumps(receipt),flush=True)


if __name__ == '__main__':
    main()
