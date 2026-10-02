"""One-off cleanup: retain both historical paths, replace duplicate PNG storage."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
from datetime import datetime, timezone

ROOT = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
HERE = Path(__file__).resolve().parent


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def signature(path):
    value = path.lstat()
    assert stat.S_ISREG(value.st_mode) and path.resolve() == path
    return {'device': value.st_dev, 'inode': value.st_ino, 'size': value.st_size,
            'mtime_ns': value.st_mtime_ns, 'mode': stat.S_IMODE(value.st_mode),
            'uid': value.st_uid, 'gid': value.st_gid}


def main():
    manifest = Path(sys.argv[1]).resolve()
    expected_sha = sys.argv[2]
    assert digest(manifest) == expected_sha
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    pairs = [r for r in records if r.get('record_type') == 'file_pair']
    assert len(pairs) == 15238
    assert len({r['export_path'] for r in pairs}) == len(pairs)
    # Recheck the complete inspected set before replacing a single directory entry.
    for row in pairs:
        source, target = Path(row['source_path']), Path(row['export_path'])
        assert source.is_relative_to(ROOT) and target.is_relative_to(ROOT)
        assert source.suffix == target.suffix == '.png'
        assert row['same_bytes'] and row['unchanged_during_hash'] and row['hardlink_candidate']
        assert row['source_sha256'] == row['export_sha256']
        assert row['source_stat']['device'] == row['export_stat']['device']
        assert row['export_stat']['nlink'] == target.stat().st_nlink == 1
        for path, recorded in ((source, row['source_stat']), (target, row['export_stat'])):
            assert all(recorded[k] == v for k, v in signature(path).items()), str(path)
        assert all(row['source_stat'][k] == row['export_stat'][k] for k in ('size', 'mode', 'uid', 'gid'))
    before = shutil.disk_usage(ROOT).free
    started = datetime.now(timezone.utc).isoformat()
    journal = HERE / 'png_merge_operations.jsonl'
    with journal.open('x', encoding='utf-8') as log:
        for index, row in enumerate(pairs, 1):
            source, target = Path(row['source_path']), Path(row['export_path'])
            temporary = target.with_name(target.name + '.dedup-link')
            assert not temporary.exists()
            os.link(source, temporary)
            os.replace(temporary, target)
            assert source.samefile(target)
            log.write(json.dumps({'source': str(source), 'target': str(target),
                                  'sha256': row['source_sha256'], 'shared_inode': target.stat().st_ino}) + '\n')
            if index % 500 == 0:
                log.flush()
                print(json.dumps({'merged': index}), flush=True)
    # Rehash retained data, and verify both old paths refer to those exact bytes.
    for row in pairs:
        source, target = Path(row['source_path']), Path(row['export_path'])
        assert source.samefile(target)
        assert digest(source) == row['source_sha256']
    receipt = {
        'status': 'COMPLETED_VERIFIED', 'started_utc': started,
        'finished_utc': datetime.now(timezone.utc).isoformat(),
        'manifest_sha256': expected_sha, 'pairs_merged': len(pairs),
        'expected_reclaimed_allocated_bytes': sum(r['potential_reclaim_allocated_bytes'] for r in pairs),
        'disk_free_before': before, 'disk_free_after': shutil.disk_usage(ROOT).free,
        'all_paths_preserved': True, 'all_original_bytes_rehashed_unchanged': True,
        'non_png_files_changed': 0, 'api_calls': 0, 'gpu_jobs': 0,
        'semantics': 'Export PNGs now share source inodes; these paths are not independent backups. Export mtimes become original source mtimes; content identities and all non-PNG records are unchanged.',
        'operation_journal_sha256': digest(journal), 'script_sha256': digest(Path(__file__)),
    }
    with (HERE / 'png_merge_receipt.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
