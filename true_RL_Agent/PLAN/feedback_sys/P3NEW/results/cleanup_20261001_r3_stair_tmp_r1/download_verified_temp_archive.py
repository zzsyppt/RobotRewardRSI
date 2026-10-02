"""After the earlier WAN archive completes, copy the fixed R3 tar to D and fully verify."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

HERE = Path(__file__).resolve().parent
EARLIER = HERE.parent/'cleanup_20261001_stair_cold_r2/archive_verification.json'
LOCATOR = HERE/'remote_control/archive_restore_locator.json'
LOCATOR_SHA = '8ce8071fe8d3bda8ca185d22db3dddb0958726f620e8faa7436570f164ff1ec9'
ARCHIVE_SHA = '17db7006c711c5a6e431bcac23b3da1932654016ab4131ae56d26290b344fd45'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(name, value):
    with (HERE/name).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def main():
    earlier = json.loads(EARLIER.read_bytes())
    assert earlier['status'] == 'ARCHIVE_VERIFIED_REMOTE_UNLINK_NOT_PERFORMED'
    assert sha(LOCATOR) == LOCATOR_SHA
    locator = json.loads(LOCATOR.read_bytes())
    assert locator['archive_sha256'] == ARCHIVE_SHA and locator['verified_members'] == 3874
    expected = {r['archive_member']: r for r in locator['members']}
    assert len(expected) == 3874
    archive = HERE/'r3_stair_png_sources.tar'
    partial = archive.with_suffix('.tar.partial')
    assert not archive.exists() and not partial.exists()
    save('local_archive_registration.json', {'status': 'COPY_REGISTERED_AFTER_EARLIER_WAN_VERIFIED',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'remote_archive': locator['archive'],
        'archive_sha256': ARCHIVE_SHA, 'expected_size_bytes': locator['archive_size_bytes'],
        'local_archive': str(archive), 'member_locator_sha256': LOCATOR_SHA,
        'earlier_archive_verification_sha256': sha(EARLIER), 'remote_deletions_authorized': False})
    remote = r'''
from pathlib import Path
import hashlib,json,shutil,stat,sys
p=Path('/var/tmp/feedback_p3new_20261001_r3_stair_cold_yuyangliu_r1/r3_stair_png_sources.tar')
locator=p.parent/'archive_restore_locator.json'
assert p.resolve()==p and p.parent.resolve()==p.parent and stat.S_ISREG(p.lstat().st_mode)
assert hashlib.sha256(locator.read_bytes()).hexdigest()=='8ce8071fe8d3bda8ca185d22db3dddb0958726f620e8faa7436570f164ff1ec9'
def signature(p):
 s=p.stat();return dict(device=s.st_dev,inode=s.st_ino,nlink=s.st_nlink,size=s.st_size,mtime_ns=s.st_mtime_ns,allocated_bytes=s.st_blocks*512)
before=signature(p);assert before['size']==5855959040
print(json.dumps({'phase':'before_stream','archive':str(p),'stat':before}),file=sys.stderr,flush=True)
with p.open('rb') as f:shutil.copyfileobj(f,sys.stdout.buffer,1024*1024)
sys.stdout.buffer.flush()
assert signature(p)==before
print(json.dumps({'phase':'after_stream','stat_unchanged':True,'stat':signature(p),'remote_mutations':0}),file=sys.stderr,flush=True)
'''
    h = hashlib.sha256()
    size = 0
    last = time.monotonic()
    with (HERE/'local_archive_remote_stream.jsonl').open('xb') as log, partial.open('xb') as target:
        p = subprocess.Popen(['ssh', 'humanoid-server', '/data/yuyangliu/envs/isaaclab6_validation/bin/python', '-B', '-'],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log)
        p.stdin.write(remote.encode())
        p.stdin.close()
        while chunk := p.stdout.read(1024*1024):
            target.write(chunk)
            h.update(chunk)
            size += len(chunk)
            if time.monotonic()-last >= 15:
                print(json.dumps({'received_bytes': size}), flush=True)
                last = time.monotonic()
        rc = p.wait()
        target.flush()
        os.fsync(target.fileno())
    assert rc == 0, 'Preserve partial and log, no automatic retry.'
    assert size == locator['archive_size_bytes'] and h.hexdigest() == ARCHIVE_SHA
    seen = set()
    png_bytes = 0
    with tarfile.open(partial, 'r:') as tf:
        for member in tf:
            assert member.isfile() and member.name in expected and member.name not in seen
            r = expected[member.name]
            assert member.size == r['size_bytes']
            with tf.extractfile(member) as f:
                assert hashlib.file_digest(f, 'sha256').hexdigest() == r['sha256']
            seen.add(member.name)
            png_bytes += member.size
    assert seen == set(expected) and png_bytes == locator['verified_png_bytes']
    assert sha(partial) == ARCHIVE_SHA
    partial.rename(archive)
    save('archive_restore_locator.local_D.json', {'status': 'LOCAL_D_ARCHIVE_FULLY_VERIFIED',
        'archive_on_windows_host': str(archive), 'archive_sha256': ARCHIVE_SHA,
        'archive_size_bytes': size, 'retained_remote_temp_archive': locator['archive'],
        'original_remote_locator_sha256': LOCATOR_SHA, 'restore_rule': locator['restore_rule'],
        'members': [{k: r[k] for k in ('archive_member','source_path','export_path','size_bytes','sha256')} for r in locator['members']]})
    receipt = {'status': 'LOCAL_D_ARCHIVE_AND_ALL_MEMBERS_VERIFIED', 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
        'archive': str(archive), 'archive_sha256': ARCHIVE_SHA, 'archive_size_bytes': size,
        'verified_members': len(seen), 'verified_png_bytes': png_bytes,
        'full_disk_archive_sha_recomputed_after_member_verification': True,
        'member_paths_sizes_sha_and_exact_count_match_registered_remote_locator': True,
        'remote_stream_log_sha256': sha(HERE/'local_archive_remote_stream.jsonl'),
        'local_restore_locator_sha256': sha(HERE/'archive_restore_locator.local_D.json'),
        'remote_temp_archive_retained': True, 'remote_deletions': 0, 'api_calls': 0, 'gpu_jobs': 0}
    save('local_archive_verification.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
