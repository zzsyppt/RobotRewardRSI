"""Read-only, task-scoped PNG comparison. Emits evidence; never mutates remote files."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time

ROOT = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
PAIRS = [
    ('R1_push','n5_confirmation_20260930/push/video','n5_confirmation_20260930/exports/push/video'),
    ('R1_stair','n5_confirmation_20260930/stair/video','n5_confirmation_20260930/exports/stair/video'),
    ('dev_push_body','push_body_grounding_dev_r2/video','push_body_grounding_dev_r2_export/video'),
    ('dev_push_dual','push_dual_view_dev_r3/video','push_dual_view_dev_r3_export/video'),
    ('dev_stair_dual','stair_dual_view_dev_r3/video','stair_dual_view_dev_r3_export/video'),
]


def emit(obj):
    print(json.dumps(obj,ensure_ascii=False),flush=True)


def scan(base):
    result = {}
    for folder, dirs, names in os.walk(base, followlinks=False):
        dirs[:] = [d for d in dirs if not Path(folder,d).is_symlink()]
        for name in names:
            p = Path(folder,name)
            if p.suffix.lower() == '.png':
                assert stat.S_ISREG(p.lstat().st_mode), str(p)
                result[p.relative_to(base).as_posix()] = p
    return result


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def snapshot(s):
    return {'device':s.st_dev,'inode':s.st_ino,'nlink':s.st_nlink,'size':s.st_size,
            'allocated_bytes':s.st_blocks*512,'mtime_ns':s.st_mtime_ns,'mode':stat.S_IMODE(s.st_mode),
            'uid':s.st_uid,'gid':s.st_gid}


def compare(job):
    name, rel, source, export = job
    a,b=source.stat(),export.stat()
    ah,bh=digest(source),digest(export)
    unchanged=snapshot(a)==snapshot(source.stat()) and snapshot(b)==snapshot(export.stat())
    same_inode=(a.st_dev,a.st_ino)==(b.st_dev,b.st_ino)
    eligible=(unchanged and ah==bh and a.st_size==b.st_size and a.st_dev==b.st_dev
              and not same_inode and b.st_nlink==1)
    return {'record_type':'file_pair','group':name,'relative_path':rel,
            'source_path':str(source),'export_path':str(export),'source_sha256':ah,'export_sha256':bh,
            'source_stat':snapshot(a),'export_stat':snapshot(b),'unchanged_during_hash':unchanged,
            'same_bytes':ah==bh and a.st_size==b.st_size,'same_inode':same_inode,
            'hardlink_candidate':eligible,
            'potential_reclaim_allocated_bytes':b.st_blocks*512 if eligible else 0,
            'reference_policy':'preserve both absolute paths and all bytes; retain all non-PNG metadata',
            'execution_status':'NOT_EXECUTED'}


start=time.time()
emit({'record_type':'header','schema':'task_png_duplicate_candidates.v1','root':str(ROOT),
      'started_unix':start,'read_only':True,'hash':'sha256','workers':4,
      'scope':'R1 and three explicit same-source dev/export video pairs only; R2 excluded',
      'action_proposed':'replace export PNG storage with hardlink to verified source; approval/execution separate',
      'backup_semantics':'retains paths/content, ceases to be an independent physical copy'})
jobs=[];groups={}
for name,a,b in PAIRS:
    left,right=scan(ROOT/a),scan(ROOT/b)
    groups[name]={'source_dir':str(ROOT/a),'export_dir':str(ROOT/b),'source_png_count':len(left),
                  'export_png_count':len(right),'only_source':sorted(set(left)-set(right)),
                  'only_export':sorted(set(right)-set(left)),'paired':len(set(left)&set(right)),
                  'same_bytes_pairs':0,'eligible_pairs':0,'changed_pairs':0,'potential_reclaim_bytes':0}
    jobs.extend((name,rel,left[rel],right[rel]) for rel in sorted(set(left)&set(right)))
print('Read-only SHA inventory: '+str(len(jobs))+' file pairs.',file=sys.stderr,flush=True)
with ThreadPoolExecutor(max_workers=4) as pool:
    for i,result in enumerate(pool.map(compare,jobs),1):
        d=groups[result['group']]
        d['same_bytes_pairs']+=result['same_bytes']
        d['eligible_pairs']+=result['hardlink_candidate']
        d['changed_pairs']+=not result['unchanged_during_hash']
        d['potential_reclaim_bytes']+=result['potential_reclaim_allocated_bytes']
        emit(result)
        if i%1000==0:print('Read-only SHA inventory: '+str(i)+'/'+str(len(jobs)),file=sys.stderr,flush=True)
complete=all(not d['only_source'] and not d['only_export'] and d['paired']==d['same_bytes_pairs']==d['eligible_pairs']
             and not d['changed_pairs'] for d in groups.values())
emit({'record_type':'summary','complete':complete,'finished_unix':time.time(),
      'elapsed_s':time.time()-start,'groups':groups,
      'potential_reclaim_allocated_bytes':sum(d['potential_reclaim_bytes'] for d in groups.values()),
      'execution_status':'NOT_EXECUTED'})
assert complete
