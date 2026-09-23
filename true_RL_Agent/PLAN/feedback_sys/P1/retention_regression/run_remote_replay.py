"""CPU-only replay of frozen sessions in two isolated source snapshots.

Inputs: an overlay ZIP with the original audit driver, sync_manifest.json, and
only the changed source files. No training, simulator, or model invocation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    scratch = args.scratch.resolve()
    scratch.mkdir(parents=True, exist_ok=False)
    repo = Path("/data/yuyangliu/AGENT")
    python = "/data/yuyangliu/envs/isaaclab6_validation/bin/python"
    baseline = "fe96455ddd1a8c6ba30ee16e6a3a015422afe8d3"
    base = scratch / "baseline"
    base.mkdir()
    with subprocess.Popen(["git", "-C", str(repo), "archive", baseline, "true_RL_Agent"], stdout=subprocess.PIPE) as proc:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as archive:
            for member in archive:
                target = (base / member.name).resolve()
                if not target.is_relative_to(base):
                    raise ValueError("archive path escapes scratch")
                archive.extract(member, base)
        if proc.wait() != 0:
            raise RuntimeError("git archive failed")
    current = scratch / "current"
    shutil.copytree(base, current)
    with zipfile.ZipFile(args.bundle) as bundle:
        manifest = json.loads(bundle.read("sync_manifest.json"))
        if manifest["baseline_commit"] != baseline:
            raise ValueError("baseline mismatch")
        for item in manifest["source_files"]:
            target = (current / item["path"]).resolve()
            if not target.is_relative_to(current / "true_RL_Agent/src"):
                raise ValueError("source overlay outside src")
            content = bundle.read(item["path"])
            if hashlib.sha256(content).hexdigest() != item["sha256"]:
                raise ValueError("overlay SHA mismatch")
            target.write_bytes(content)
            item["remote_sha256"] = sha(target)
            item["baseline_sha256"] = sha(base / item["path"])
        driver = scratch / "p1_replay_audit.py"
        driver.write_bytes(bundle.read("p1_replay_audit.py"))
        if sha(driver) != manifest["driver_sha256"]:
            raise ValueError("driver SHA mismatch")
    manifest.update(scratch=str(scratch), python=python, original_driver_unmodified=True)
    write(scratch / "sync_manifest.json", manifest)
    source_paths = {
        "stair": repo / "true_RL_Agent/outputs/formal_go2_stair_15round_20260901_c6ab8ba",
        "push": repo / "true_RL_Agent/PLAN/2new_world/Go2_push/attempt_08/session",
    }
    identity_names = ("ledger.jsonl", "session_genesis.json", "resolved_cell_manifest.json", "generation_session_contract.json")
    before = {f"{task}/{name}": sha(source / name) for task, source in source_paths.items() for name in identity_names}
    results = []
    for label, snapshot in (("baseline", base), ("current", current)):
        for task, source in source_paths.items():
            output = scratch / f"{label}_{task}"
            frames = "13,16" if task == "stair" else "6"
            command = [python, str(driver), "--root", str(snapshot / "true_RL_Agent"), "--task", task,
                       "--source", str(source), "--output", str(output), "--frames", frames,
                       "--label", f"retention_{label}_{task}", "--context", "scratch" if task == "stair" else "session",
                       "--memory-variant", "full"]
            started = time.monotonic()
            with (scratch / f"{label}_{task}.log").open("w") as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            report = json.loads((output / "report.json").read_text())
            item = {"label": label, "task": task, "command": command, "process_exit_code": process.returncode,
                    "elapsed_s": round(time.monotonic() - started, 3), "report": str(output / "report.json"),
                    "report_sha256": sha(output / "report.json"), "status": report["status"], "payload_measurements": []}
            for generation in frames.split(","):
                payload_path = output / f"payload_before_g{int(generation):02d}.json"
                if not payload_path.exists():
                    continue
                memory = json.loads(payload_path.read_text())["public_history"]["experiment_memory"]
                rows = [row for entry in memory["experiments"] for row in entry.get("trial_outcomes", [])]
                item["payload_measurements"].append({"next_generation": int(generation), "path": str(payload_path),
                    "sha256": sha(payload_path), "runtime_column_values": dict(Counter(row[9] for row in rows if len(row) > 9 and isinstance(row[9], str))),
                    "rows": len(rows), "runtime_string_rows": sum(len(row) > 9 and isinstance(row[9], str) for row in rows),
                    "top_runtime_observation_count": sum("runtime_observation" in entry for entry in memory["experiments"])})
            results.append(item)
            write(scratch / "run_summary.json", {"runs": results})
            print(json.dumps({"label": label, "task": task, "status": report["status"], "elapsed_s": item["elapsed_s"], "error": report.get("error")}), flush=True)
    after = {f"{task}/{name}": sha(source / name) for task, source in source_paths.items() for name in identity_names}
    write(scratch / "run_summary.json", {"runs": results, "session_identity_hashes_before": before,
          "session_identity_hashes_after": after, "eight_session_identity_files_unchanged": before == after,
          "historical_write_boundary": "Driver uses frozen sessions as input; Push composition read_only=True; Stair composition and every output are under scratch. Hash check covers eight identity files only, not all session files.",
          "no_training_simulator_or_api_invocation": True})
    if any(item["status"] != "PASS" or item["process_exit_code"] for item in results) or before != after:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
