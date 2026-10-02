"""One-shot authorized R3 Stair PNG cold relocation on the server; no model or GPU."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tarfile
import time

B = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
ROOTS = (B/'n5_confirmation_20260930_r3/stair/video', B/'n5_confirmation_20260930_r3/exports/stair/video')
COLD = Path('/var/tmp/feedback_p3new_20261001_r3_stair_cold_yuyangliu_r1')
EXPECTED_COUNT = 3874


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def signature(p):
    s = p.stat()
    return dict(device=s.st_dev, inode=s.st_ino, nlink=s.st_nlink, size=s.st_size,
                allocated_bytes=s.st_blocks*512, mtime_ns=s.st_mtime_ns)


def emit(value):
    print(json.dumps(value), flush=True)


def save(name, value):
    p = COLD/name
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    return {'path': str(p), 'sha256': sha(p), 'size_bytes': p.stat().st_size}


def dependency_check(phase):
    roots = tuple(map(str, ROOTS))
    matches = []
    denied = checked = 0
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
            checked += 1
            if any(any(root in arg for root in roots) for arg in args):
                matches.append({'pid': int(proc.name), 'kind': 'cmdline_target_path'})
            for p in [proc/'cwd', *(proc/'fd').iterdir()]:
                try:
                    target = os.readlink(p)
                    if any(target == root or target.startswith(root+'/') for root in roots):
                        matches.append({'pid': int(proc.name), 'kind': 'cwd_or_open_fd', 'target': target})
                except FileNotFoundError:
                    pass
        except PermissionError:
            denied += 1
        except FileNotFoundError:
            pass
    value = {'phase': phase, 'observed_unix': time.time(), 'visible_processes_checked': checked,
             'permission_limited_processes': denied, 'target_matches': matches,
             'limit': 'Current visible cmdline/cwd/open-FD only, not inaccessible or future reads.'}
    emit({'record_type': 'dependency_snapshot', **value})
    assert not matches, 'Current target dependency; stop without unlink'
    return value


def retained_contents():
    paths = {p for root in ROOTS for p in root.rglob('*') if p.is_file() and p.suffix != '.png'}
    paths.add(B/'n5_confirmation_20260930_r3/stair/result.json')
    return [{'path': str(p), 'size_bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(paths)]


def main():
    assert B.resolve() == B
    assert all(root.resolve() == root and root.is_relative_to(B) for root in ROOTS)
    assert COLD.parent.resolve() == Path('/var/tmp') and not COLD.exists()
    assert COLD.parent.stat().st_dev != B.stat().st_dev
    members = []
    source = {p.name: p for p in ROOTS[0].glob('*.png')}
    export = {p.name: p for p in ROOTS[1].glob('*.png')}
    assert source.keys() == export.keys() and len(source) == EXPECTED_COUNT
    for name, p in sorted(source.items()):
        q = export[name]
        assert p.resolve() == p and q.resolve() == q
        assert stat.S_ISREG(p.lstat().st_mode) and stat.S_ISREG(q.lstat().st_mode)
        a, b = signature(p), signature(q)
        assert a == b and a['nlink'] == 2
        members.append({'archive_member': str(p.relative_to(B)), 'source_path': str(p),
                        'export_path': str(q), 'size_bytes': a['size'], 'source_stat': a, 'export_stat': b})
    expected_bytes = sum(r['size_bytes'] for r in members)
    assert shutil.disk_usage(COLD.parent).free > expected_bytes + 10*1024**3
    completed = json.loads((B/'n5_confirmation_20260930_r3/stair/result.json').read_bytes())
    assert completed['status'] == 'PASS' and completed['final_jobs'] == 0
    COLD.mkdir(mode=0o700)
    assert COLD.stat().st_uid == os.getuid() and stat.S_IMODE(COLD.stat().st_mode) == 0o700
    registration = save('registration.json', {'status': 'FIXED_SCOPE_BEFORE_HASH_AND_ARCHIVE',
        'created_unix': time.time(), 'source_roots': list(map(str, ROOTS)), 'cold_directory': str(COLD),
        'data_device': B.stat().st_dev, 'cold_device': COLD.stat().st_dev, 'pairs': EXPECTED_COUNT,
        'logical_png_bytes': expected_bytes, 'members': members,
        'authorization': 'Root explicitly authorized this exact temporary cold relocation, followed by local D archival before task closeout.'})
    snapshots = [dependency_check('before_hash')]
    preserved = retained_contents()
    for i, r in enumerate(members):
        p, q = Path(r['source_path']), Path(r['export_path'])
        assert signature(p) == r['source_stat'] and signature(q) == r['export_stat']
        r['sha256'] = sha(p)
        assert sha(q) == r['sha256']
        assert signature(p) == r['source_stat'] and signature(q) == r['export_stat']
        if (i+1) % 500 == 0:
            emit({'record_type': 'hash_progress', 'verified_pairs': i+1})
    member_index = save('member_index.json', {'status': 'FULL_HASH_INDEX_BEFORE_ARCHIVE',
        'registration': registration, 'source_roots': list(map(str, ROOTS)), 'pairs': EXPECTED_COUNT,
        'logical_png_bytes': expected_bytes, 'members': members, 'preserved_non_png_and_result': preserved})
    expected = {r['archive_member']: r for r in members}
    assert len(expected) == EXPECTED_COUNT
    snapshots.append(dependency_check('before_archive'))
    partial = COLD/'r3_stair_png_sources.tar.partial'
    archive = COLD/'r3_stair_png_sources.tar'
    with tarfile.open(partial, 'w', format=tarfile.PAX_FORMAT) as tf:
        for r in members:
            p, q = Path(r['source_path']), Path(r['export_path'])
            assert signature(p) == r['source_stat'] and signature(q) == r['export_stat']
            tf.add(p, arcname=r['archive_member'], recursive=False)
            assert signature(p) == r['source_stat'] and signature(q) == r['export_stat']
    with partial.open('rb') as f:
        os.fsync(f.fileno())
    seen = set()
    verified_bytes = 0
    with tarfile.open(partial, 'r:') as tf:
        for item in tf:
            assert item.isfile() and item.name in expected and item.name not in seen
            r = expected[item.name]
            assert item.size == r['size_bytes']
            with tf.extractfile(item) as f:
                assert hashlib.file_digest(f, 'sha256').hexdigest() == r['sha256']
            seen.add(item.name)
            verified_bytes += item.size
    assert seen == set(expected) and verified_bytes == expected_bytes
    archive_sha = sha(partial)
    partial.rename(archive)
    locator = save('archive_restore_locator.json', {'status': 'TEMP_ARCHIVE_FULLY_VERIFIED_BEFORE_UNLINK',
        'archive': str(archive), 'archive_sha256': archive_sha, 'archive_size_bytes': archive.stat().st_size,
        'member_index': member_index, 'verified_members': len(seen), 'verified_png_bytes': verified_bytes,
        'restore_rule': 'Restore each exact archive_member to source_path after SHA/size verification, then hardlink export_path to source_path. Do not overwrite an existing different file or any non-PNG. Original inode numbers are not restored.',
        'temporary_storage_limit': 'Same server, separate filesystem, not a redundant or permanent backup. Transfer archive to local D and verify all members before task closeout; do not remove this temporary archive now.',
        'members': members})
    emit({'record_type': 'archive_verified', 'archive_sha256': archive_sha, 'members': len(seen)})
    snapshots.append(dependency_check('immediately_before_unlink'))
    assert retained_contents() == preserved
    for root, key in zip(ROOTS, ('source_path', 'export_path')):
        assert {str(p) for p in root.glob('*.png')} == {r[key] for r in members}
    for r in members:
        for key, skey in (('source_path', 'source_stat'), ('export_path', 'export_stat')):
            p = Path(r[key])
            assert p.resolve() == p and p.is_relative_to(B) and p.parent in ROOTS and p.suffix == '.png'
            assert stat.S_ISREG(p.lstat().st_mode) and signature(p) == r[skey]
    free_before = shutil.disk_usage(B).free
    deleted = pairs = blocks = 0
    with (COLD/'deleted_png_names.jsonl').open('x', encoding='utf-8') as journal:
        try:
            for r in members:
                p, q = Path(r['source_path']), Path(r['export_path'])
                assert signature(p) == r['source_stat'] and signature(q) == r['export_stat']
                for target in (p, q):
                    target.unlink()
                    deleted += 1
                    journal.write(json.dumps({'path': str(target), 'original_sha256': r['sha256']})+'\n')
                    journal.flush()
                pairs += 1
                blocks += r['source_stat']['allocated_bytes']//512
            os.fsync(journal.fileno())
        finally:
            free_after = shutil.disk_usage(B).free
            receipt = {'status': 'UNLINK_COMPLETE' if pairs == EXPECTED_COUNT and deleted == 2*EXPECTED_COUNT else 'PARTIAL_STOP_NO_AUTOMATIC_RETRY',
                'finished_unix': time.time(), 'source_roots': list(map(str, ROOTS)), 'cold_directory': str(COLD),
                'archive': str(archive), 'archive_sha256': archive_sha, 'archive_size_bytes': archive.stat().st_size,
                'archive_locator': locator, 'completed_inode_pairs': pairs, 'deleted_png_names': deleted,
                'allocated_512B_blocks_of_completed_pairs': blocks, 'allocated_bytes_of_completed_pairs': blocks*512,
                'data_free_before_bytes': free_before, 'data_free_after_bytes': free_after,
                'data_free_net_delta_bytes': free_after-free_before, 'cold_fs_free_after_bytes': shutil.disk_usage(COLD).free,
                'directories_removed': 0, 'non_png_paths_unlinked': 0, 'gpu_jobs': 0, 'api_calls': 0,
                'dependency_snapshots': snapshots, 'space_limit': 'Allocated blocks identify removed inodes; net free also includes concurrent filesystem activity.',
                'local_D_transfer': 'PENDING_EXISTING_WAN_ARCHIVE_COMPLETION', 'temporary_archive_removed': False}
            save('unlink_receipt.json', receipt)
    assert pairs == EXPECTED_COUNT and all(not Path(r[k]).exists() for r in members for k in ('source_path', 'export_path'))
    assert all(root.is_dir() for root in ROOTS) and retained_contents() == preserved
    save('preservation_check.json', {'status': 'PASS', 'non_png_and_capture_result_files_rehashed': len(preserved),
        'contents_and_membership_unchanged': True, 'source_and_export_directories_retained': True})
    emit({'record_type': 'complete', **receipt})


if __name__ == '__main__':
    main()
