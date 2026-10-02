"""Fixed two-group archival to local D:, with no remote writes or deletions."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
MANIFEST = HERE.parent / "storage_duplicate_audit_20260930_r1/png_duplicate_candidates.jsonl"
B = "/data/yuyangliu/AGENT/feedback_p3new_20260929"
GROUPS = ("dev_push_body", "dev_push_dual")
REMOTE_PYTHON = "/data/yuyangliu/envs/isaaclab6_validation/bin/python"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(name, value):
    with (HERE / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


rows = [r for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if (r := json.loads(line)).get("record_type") == "file_pair" and r.get("group") in GROUPS]
assert len(rows) == 3740 and all(sum(r["group"] == g for r in rows) == 1870 for g in GROUPS)
expected = {str(PurePosixPath(r["source_path"]).relative_to(B)): r for r in rows}
assert len(expected) == len(rows) and len({r["source_sha256"] for r in rows}) == len(rows)
assert shutil.disk_usage(HERE).free > sum(r["source_stat"]["size"] for r in rows) + 1024**3
archive = HERE / "two_finished_push_png_sources.tar"
partial = archive.with_suffix(".tar.partial")
assert not archive.exists() and not partial.exists()
manifest_sha = sha(MANIFEST)
remote = r'''
import hashlib,json,os,pathlib,stat,sys,tarfile,time
B=pathlib.Path("/data/yuyangliu/AGENT/feedback_p3new_20260929")
MF=B/"cleanup_20260930/png_duplicate_candidates.jsonl"
EXPECTED_MANIFEST="__MANIFEST_SHA__"
raw=MF.read_bytes();assert hashlib.sha256(raw).hexdigest()==EXPECTED_MANIFEST
groups={"dev_push_body":"push_body_grounding_dev_r2","dev_push_dual":"push_dual_view_dev_r3"}
rows=[r for line in raw.decode().splitlines() if (r:=json.loads(line)).get("record_type")=="file_pair" and r.get("group") in groups]
assert len(rows)==3740 and all(sum(r["group"]==g for r in rows)==1870 for g in groups)
roots=[str(B/(n+s)/"video") for n in groups.values() for s in ("","_export")]
def emit(value): print(json.dumps(value),file=sys.stderr,flush=True)
def digest(p):
 with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def signature(p):
 s=p.stat();return dict(device=s.st_dev,inode=s.st_ino,nlink=s.st_nlink,size=s.st_size,allocated_bytes=s.st_blocks*512,mtime_ns=s.st_mtime_ns)
def relevant(value):return any(value==r or value.startswith(r+"/") for r in roots)
def dependency_check(phase):
 matches=[];denied=0;checked=0
 for p in pathlib.Path("/proc").iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:
   args=(p/"cmdline").read_bytes().decode(errors="replace").split("\0")
   checked+=1
   if any(any(r in a for r in roots) for a in args):matches.append({"pid":int(p.name),"kind":"cmdline_target_path"})
   for name in ("cwd",):
    try:
     target=os.readlink(p/name)
     if relevant(target):matches.append({"pid":int(p.name),"kind":name,"target":target})
    except FileNotFoundError:pass
   try:
    for fd in (p/"fd").iterdir():
     try:
      target=os.readlink(fd)
      if relevant(target):matches.append({"pid":int(p.name),"kind":"open_fd","target":target})
     except FileNotFoundError:pass
   except FileNotFoundError:pass
  except PermissionError:denied+=1
  except FileNotFoundError:pass
 emit({"record_type":"dependency_snapshot","phase":phase,"observed_unix":time.time(),"visible_processes_checked":checked,"permission_limited_processes":denied,"target_matches":matches,"limit":"Point-in-time visible cmdline/cwd/open-FD check; not a guarantee about future or inaccessible dependencies."})
 assert not matches
dependency_check("before_hash")
for g,n in groups.items():
 for suffix,key in (("","source_path"),("_export","export_path")):
  root=B/(n+suffix)/"video"
  actual={str(p) for p in root.rglob("*.png")}
  assert actual=={r[key] for r in rows if r["group"]==g},(g,key,"membership mismatch")
states={};allocated=0;non_two_links=[]
for i,r in enumerate(sorted(rows,key=lambda x:x["source_path"])):
 p=pathlib.Path(r["source_path"]);q=pathlib.Path(r["export_path"])
 assert p.parent==B/groups[r["group"]]/"video" and q.parent==B/(groups[r["group"]]+"_export")/"video"
 assert p.name==q.name==r["relative_path"] and p.suffix==".png"
 assert stat.S_ISREG(p.lstat().st_mode) and stat.S_ISREG(q.lstat().st_mode)
 a,b=signature(p),signature(q)
 assert a["device"]==b["device"] and a["inode"]==b["inode"]
 assert digest(p)==r["source_sha256"]==digest(q)==r["export_sha256"]
 assert a==signature(p) and b==signature(q) and a["size"]==r["source_stat"]["size"]
 states[str(p)]=(a,b)
 if a["nlink"]==2:allocated+=a["allocated_bytes"]
 else:non_two_links.append(str(p))
 emit({"record_type":"verified_pair","group":r["group"],"relative_path":r["relative_path"],"source_path":str(p),"export_path":str(q),"sha256":r["source_sha256"],"source_stat":a,"export_stat":b})
dependency_check("before_stream")
emit({"record_type":"stream_start","pid":os.getpid(),"members":len(rows),"bytes":sum(r["source_stat"]["size"] for r in rows)})
with tarfile.open(fileobj=sys.stdout.buffer,mode="w|",format=tarfile.PAX_FORMAT) as tf:
 for r in sorted(rows,key=lambda x:x["source_path"]):
  p=pathlib.Path(r["source_path"]);q=pathlib.Path(r["export_path"]);a,b=states[str(p)]
  assert signature(p)==a and signature(q)==b
  tf.add(p,arcname=str(p.relative_to(B)),recursive=False)
  assert signature(p)==a and signature(q)==b
sys.stdout.buffer.flush()
dependency_check("after_stream")
emit({"record_type":"remote_complete","status":"PNG_ONLY_STREAM_COMPLETED_NO_REMOTE_MUTATIONS","pid":os.getpid(),"members":len(rows),"manifest_sha256":EXPECTED_MANIFEST,"expected_reclaim_only_after_both_names_unlinked_bytes":allocated,"nlink_not_two":non_two_links,"remote_deletions":0,"remote_archive_files":0,"non_png_files_changed":0})
'''.replace("__MANIFEST_SHA__", manifest_sha)

started = datetime.now(timezone.utc).isoformat()
save("archive_registration.json", {"started_at_utc": started, "pid": os.getpid(), "manifest": str(MANIFEST),
     "manifest_sha256": manifest_sha, "groups": GROUPS, "expected_members": len(rows),
     "expected_png_bytes": sum(r["source_stat"]["size"] for r in rows), "archive": str(archive),
     "remote_mutations_authorized": False, "scope": "Archive only two explicitly named completed-development PNG source directories"})
with (HERE / "remote_verification.jsonl").open("xb") as log, partial.open("xb") as target:
    proc = subprocess.Popen(["ssh", "humanoid-server", REMOTE_PYTHON, "-B", "-"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=log)
    print(json.dumps({"local_pid": os.getpid(), "ssh_pid": proc.pid, "status": "REMOTE_VERIFY_THEN_STREAM"}), flush=True)
    proc.stdin.write(remote.encode()); proc.stdin.close()
    h = hashlib.sha256(); size = 0; last = time.monotonic()
    while chunk := proc.stdout.read(1024 * 1024):
        target.write(chunk); h.update(chunk); size += len(chunk)
        if time.monotonic() - last >= 15:
            print(json.dumps({"archive_bytes_received": size, "ssh_pid": proc.pid}), flush=True); last = time.monotonic()
    rc = proc.wait(); target.flush(); os.fsync(target.fileno())
assert rc == 0, f"Remote exit {rc}; preserve partial and verification log for inspection"
remote_records = [json.loads(line) for line in (HERE / "remote_verification.jsonl").read_text().splitlines()]
complete = remote_records[-1]
assert complete["record_type"] == "remote_complete" and complete["members"] == len(rows)
seen = set(); verified_bytes = 0
with tarfile.open(partial, "r:") as tf:
    for member in tf:
        assert member.isfile() and member.name in expected and member.name not in seen, member.name
        r = expected[member.name]; assert member.size == r["source_stat"]["size"]
        with tf.extractfile(member) as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == r["source_sha256"], member.name
        seen.add(member.name); verified_bytes += member.size
assert seen == set(expected)
partial.rename(archive)
save("archive_verification.json", {"status": "ARCHIVE_VERIFIED_REMOTE_UNLINK_NOT_PERFORMED", "started_at_utc": started,
     "finished_at_utc": datetime.now(timezone.utc).isoformat(), "archive": str(archive), "archive_sha256": h.hexdigest(),
     "archive_size_bytes": size, "verified_members": len(seen), "verified_png_bytes": verified_bytes,
     "member_paths_counts_sizes_and_sha_match_prior_manifest": True, "manifest_sha256": manifest_sha,
     "remote_verification_sha256": sha(HERE / "remote_verification.jsonl"), "remote_final": complete,
     "dependency_snapshots": [r for r in remote_records if r["record_type"] == "dependency_snapshot"],
     "source_and_export_bytes_retained": True, "metadata_npz_responses_retained": True,
     "remote_deletions": 0, "gpu_jobs": 0, "api_calls": 0,
     "release_condition": "Root review of this evidence is required before any later remote unlink; recheck live dependencies then."})
print(json.dumps({"status": "VERIFIED", "members": len(seen), "archive_sha256": h.hexdigest(), "archive_bytes": size,
                  "expected_reclaim_after_both_names_unlinked_bytes": complete["expected_reclaim_only_after_both_names_unlinked_bytes"]}), flush=True)
