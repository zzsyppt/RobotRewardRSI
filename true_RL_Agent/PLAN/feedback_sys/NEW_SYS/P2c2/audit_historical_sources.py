"""Read only the three approved numeric jobs and their q capture source snapshots."""
import hashlib
import json
from pathlib import Path
import resource
import time

started, cpu = time.perf_counter(), time.process_time()
paths = [
    ("k1", "/data/yuyangliu/AGENT/feedback_p2_audit_20260924/source",
     "/data/yuyangliu/AGENT/feedback_p2_audit_20260924/stair_parent_on",
     "0389a10d7cb9dbe14fff063f43c56482fcc6fbd918281a37beacb59c043db8d3"),
    ("k2", "/data/yuyangliu/AGENT/feedback_p2_20260923/source_r2",
     "/data/yuyangliu/AGENT/feedback_p2_20260923/r2_gen8_on",
     "fe488cb848dbe61454c0b9cc4fa6d8f8fc278deb1157065170577d9e91be5f20"),
    ("k3", "/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_confirmation_source_r4_final",
     "/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_confirmation_20261001_r4/stair",
     "8c5f2d68db3df151c850beb6e2147ced671fb00acc802e23427f188c0d7e9d83"),
]


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


rows = []
for sid, source_name, job_name, digest in paths:
    source, directory = Path(source_name), Path(job_name)
    archive = directory/"numeric_evidence/manifest.json"
    assert sha(archive) == digest
    job_path, result_path = directory/"public_dev_job.json", directory/"result.json"
    job, result = json.loads(job_path.read_text()), json.loads(result_path.read_text())
    assert job["metadata"]["work_dir"] == str(source)
    assert result["numeric_evidence_ref"]["manifest_sha256"] == digest
    files = []
    for name in ("isaaclab_runtime_env.py", "runtime_projection.py", "numeric_capture.py", "legacy_transition_reader.py"):
        path = source/"src/true_rl_agent/tasks/stair_climbing"/name
        text = path.read_text()
        anchors = []
        lines = text.splitlines()
        selected = set()
        for i, line in enumerate(lines):
            if any(token in line for token in ("joint_pos", "find_joints(", "preserve_order=True", "stacked =", "clone()")):
                selected.update(range(max(0, i-1), min(len(lines), i+3)))
        anchors = [{"line": i+1, "text": lines[i]} for i in sorted(selected)]
        if name == "isaaclab_runtime_env.py":
            assert '"joint_pos": robot_data.joint_pos[:, resolved.joint_ids].detach().clone()' in text
        if name == "runtime_projection.py":
            assert "ids, resolved_names = method(list(names), preserve_order=True)" in text
            assert "if tuple(resolved_names) != names or len(ids) != len(names)" in text
        files.append({"path": str(path), "sha256": sha(path), "anchors": anchors})
    rows.append({"source": sid, "archive_path": str(archive), "archive_sha256": digest,
                 "job_path": str(job_path), "job_sha256": sha(job_path), "job_work_dir": job["metadata"]["work_dir"],
                 "result_path": str(result_path), "result_sha256": sha(result_path),
                 "numeric_evidence_ref": result["numeric_evidence_ref"], "files": files})
print(json.dumps({"status": "PASS", "sources": rows, "server_writes": 0,
                  "wall_seconds": time.perf_counter()-started, "cpu_seconds": time.process_time()-cpu,
                  "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}, indent=2))
