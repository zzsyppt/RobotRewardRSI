"""R3 CPU tests, exact preflight commands and source/cohort verification only."""
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import time

BASE = Path("/data/yuyangliu/AGENT/feedback_p3new_20260929")
SOURCE = BASE / "n5_confirmation_source_r3"
CONTROL = BASE / "n5_confirmation_source_r3_control"
PREFLIGHT = BASE / "n5_confirmation_preflight_r3"
PYTHON = "/data/yuyangliu/envs/isaaclab6_validation/bin/python"


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (CONTROL / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def source_check():
    actual = {str(path.relative_to(SOURCE)): {"sha256": digest(path), "size_bytes": path.stat().st_size}
              for path in SOURCE.rglob("*") if path.is_file()}
    assert actual == read(CONTROL / "source_manifest.json"), "frozen source drift"
    return len(actual)


def main():
    commands = read(CONTROL / "commands.candidate.json")
    preparation = read(CONTROL / "source_preparation.json")
    assert commands["source_root"] == str(SOURCE) and commands["preflight_root"] == str(PREFLIGHT)
    assert commands["capture_authorized_or_executed"] is False
    assert not PREFLIGHT.exists(), "keep prior preflight output; no overwrite"
    count = source_check()
    environment = {**os.environ, "PYTHONPATH": str(SOURCE / "src") + ":" + str(SOURCE),
                   "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
    jobs = [("unit_tests", [PYTHON, "-B", "-m", "unittest", *preparation["cpu_test_modules"]])]
    for index, pattern in enumerate(preparation["phase_test_patterns"], 1):
        jobs.append((f"phase_tests_{index}", [PYTHON, "-B", "-m", "unittest", "discover", "-s",
                     str(SOURCE / "PLAN/feedback_sys/P3NEW"), "-p", pattern]))
    for task, row in commands["tasks"].items():
        argv = row["preflight_argv"]
        assert argv[argv.index("--mode")+1] == "preflight"
        jobs.append((task + "_preflight", argv))
    records = []
    for name, argv in jobs:
        started = time.time()
        with (CONTROL / (name + ".stdout.txt")).open("xb") as stdout, (CONTROL / (name + ".stderr.txt")).open("xb") as stderr:
            result = subprocess.run(argv, cwd=SOURCE, env=environment, stdout=stdout, stderr=stderr, timeout=900)
        text = (CONTROL / (name + ".stderr.txt")).read_text()
        matches = re.findall(r"Ran (\d+) tests?", text)
        record = {"name": name, "argv": argv, "cwd": str(SOURCE), "exit_code": result.returncode,
                  "environment_set": {key: environment[key] for key in ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "CUDA_VISIBLE_DEVICES")},
                  "wall_seconds": time.time()-started, "test_count": int(matches[-1]) if matches else None,
                  "stdout": str(CONTROL / (name + ".stdout.txt")), "stderr": str(CONTROL / (name + ".stderr.txt"))}
        save(name + ".execution.json", record)
        records.append(record)
        print(json.dumps({"name": name, "exit_code": result.returncode, "test_count": record["test_count"]}), flush=True)
        if result.returncode:
            print(text[-6000:], flush=True)
            raise SystemExit(result.returncode)
    save("cpu_execution_receipt.json", {"records": records, "models": 0, "gpu_runs": 0, "capture_runs": 0})
    sys.path[:0] = [str(SOURCE / "src"), str(SOURCE), str(SOURCE / "PLAN/feedback_sys/P3NEW")]
    from true_rl_agent.composition.cell_manifest import load_resolved_cell_manifest
    from true_rl_agent.tasks.stair_climbing.entrypoint_contract import _resolved_stair_config
    from true_rl_agent.tasks.stair_climbing.diagnostic_evaluation import _binding
    from true_rl_agent.tasks.stair_climbing.seed_policy import diagnostic_seed_namespace
    from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
    from true_rl_agent.tasks.push_recovery.seed_policy import derive_runner_seed
    history = read(CONTROL / "history_cohorts.before.json")
    assert not history["errors"]
    old_card = read(CONTROL / "r2_capture_card.readonly.json")
    rows = {}
    for task in ("stair", "push"):
        result = read(PREFLIGHT / task / "result.json")
        binding = result["evaluation_binding"]
        assert result["status"] == {"stair": "PREFLIGHT_PASS", "push": "PREPARED"}[task]
        assert result["mode"] == "preflight"
        assert result["checkpoint_sha256"] == "sha256:" + old_card["tasks"][task]["checkpoint"]["sha256"]
        prior = [item for item in history["files"] if item["task_id"] == "unitree_" + ("stair_climbing" if task == "stair" else "push_recovery")]
        assert binding["seed_cohort_hash"] not in {item["binding"]["seed_cohort_hash"] for item in prior}
        manifest = load_resolved_cell_manifest(PREFLIGHT / task / "composition/resolved_cell_manifest.json")
        if task == "stair":
            config = _resolved_stair_config(SOURCE / "configs/tasks/stair_go2_12dof_unified_protocol_v3.json", resolved_cell_manifest=manifest)
            resolver = config.seed_resolver()
            namespace = diagnostic_seed_namespace(commands["diagnostic_cohort"])
            assert _binding(config, namespace).to_public_dict() == binding and result["seed_namespace"] == namespace
            seeds = [resolver.derive_seed(split="public_dev", run_index=i, namespace=namespace) for i in range(1, 4)]
            previous = {cohort: [resolver.derive_seed(split="public_dev", run_index=i, namespace=diagnostic_seed_namespace(cohort))
                                for i in range(1, 4)] for cohort in ("v1", "n5_r2_20260930")}
            assert previous["v1"] == [1988007583, 1057377687, 1542212762]
            assert previous["n5_r2_20260930"] == old_card["tasks"][task]["runner_seeds"]
            extra = {"diagnostic_cohort": commands["diagnostic_cohort"], "namespace": namespace}
        else:
            config = PushRecoveryTaskConfig.from_manifest(manifest, eval_base_seed=commands["push_eval_base_seed"])
            assert config.seed_cohort_hash() == binding["seed_cohort_hash"]
            seeds = [derive_runner_seed(base_seed=commands["push_eval_base_seed"], split_name="public_dev", run_index=i) for i in range(1, 4)]
            previous = {str(base): [derive_runner_seed(base_seed=base, split_name="public_dev", run_index=i) for i in range(1, 4)]
                        for base in (1500000000, 1600000000)}
            assert previous["1600000000"] == old_card["tasks"][task]["runner_seeds"]
            extra = {"eval_base_seed": commands["push_eval_base_seed"]}
        assert len(set(seeds)) == 3 and all(not set(seeds) & set(values) for values in previous.values())
        rows[task] = {"actual_preflight_result": str(PREFLIGHT / task / "result.json"), "actual_evaluation_binding": binding,
                      "runner_seeds": seeds, "prior_runner_seeds": previous, "checkpoint_matches_r2": True,
                      "historical_job_commitments_compared": len(prior), "new_cohort_hash_absent_from_recorded_history": True, **extra}
    origins = {}
    for name in ("true_rl_agent", "true_rl_agent.tasks.stair_climbing.seed_policy", "true_rl_agent.tasks.stair_climbing.diagnostic_evaluation",
                 "true_rl_agent.tasks.stair_climbing.entrypoint_contract", "true_rl_agent.tasks.push_recovery.seed_policy",
                 "export_confirmation_raw", "prepare_confirmation_materials", "prepare_confirmation_media",
                 "confirmation_review_coverage", "run_confirmation_review"):
        path = Path(importlib.import_module(name).__file__).resolve()
        assert path.is_relative_to(SOURCE)
        origins[name] = {"__file__": str(path), "sha256": digest(path)}
    assert source_check() == count
    receipt = {"status": "CPU_STATIC_PREFLIGHT_PASS_NOT_CAPTURE_RELEASE", "finished_at_utc": datetime.now(timezone.utc).isoformat(),
               "source_files": count, "source_manifest_sha256": digest(CONTROL / "source_manifest.json"),
               "source_unchanged_after_tests_and_preflights": True, "module_origins": origins, "tasks": rows,
               "test_count": sum(record["test_count"] or 0 for record in records), "models": 0, "gpu_runs": 0, "capture_runs": 0,
               "limitations": ["Recorded public-dev commitment history is not an exhaustive inventory of all server experiments.",
                               "Same checkpoint and terrain/task suite; fresh reset trajectories do not establish new-terrain generalization.",
                               "Source and CPU preflight do not release capture, establish new semantic qualification, or complete the final capture card.",
                               "Reviewer roster assigns roles only; actual packet task routing is verified after the new raw material is packed."]}
    save("verification.json", receipt)
    with tarfile.open(CONTROL / "verification_results.tar", "x") as archive:
        for path in CONTROL.iterdir():
            if path.is_file() and path.name not in ("source.tar.gz", "verification_results.tar"):
                archive.add(path, arcname="control/"+path.name, recursive=False)
        for path in PREFLIGHT.rglob("*"):
            if path.is_file(): archive.add(path, arcname="preflight/"+str(path.relative_to(PREFLIGHT)), recursive=False)
    print(json.dumps({"status": receipt["status"], "source_files": count, "tests": receipt["test_count"], "tasks": rows,
                      "download_archive_sha256": digest(CONTROL / "verification_results.tar")}), flush=True)


if __name__ == "__main__":
    main()
