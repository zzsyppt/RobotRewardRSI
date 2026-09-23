"""Read-only CPU comparison of two training_numeric_smoke outputs.

Use the same synchronized source tree that executed the pair. Missing resource
measurements remain unavailable; exact checkpoint differences are evidence of
a failed invariance check, not proof that numeric capture caused the difference.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import traceback


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_train_argv(argv):
    removed_values = {"--summary-path", "--checkpoint-output", "--run-dir"}
    result, index = [], 0
    while index < len(argv):
        if argv[index] in removed_values:
            index += 2
        elif argv[index] == "--capture-training-numeric":
            index += 1
        else:
            result.append(argv[index]); index += 1
    return result


def compare_checkpoints(first: Path, second: Path):
    import torch
    left = torch.load(first, map_location="cpu", weights_only=True)
    right = torch.load(second, map_location="cpu", weights_only=True)
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise ValueError("checkpoint must be a state dictionary")
    keys = set(left) | set(right)
    model = sorted(key for key in keys if isinstance(key, str) and "state_dict" in key and "optimizer" not in key)
    optimizer = sorted(key for key in keys if isinstance(key, str) and "optimizer" in key)
    if not model or not optimizer:
        raise ValueError("checkpoint lacks recognized model or optimizer state")
    mismatch = []
    counts = {"model_tensors": 0, "optimizer_tensors": 0}
    def compare(a, b, path, group):
        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            counts[group + "_tensors"] += 1
            equal = isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
        elif isinstance(a, dict) and isinstance(b, dict):
            equal = set(a) == set(b)
            for key in set(a) & set(b):
                compare(a[key], b[key], path + "." + str(key), group)
        elif isinstance(a, (list, tuple)) and type(a) is type(b):
            equal = len(a) == len(b)
            for index, (av, bv) in enumerate(zip(a, b)):
                compare(av, bv, path + "." + str(index), group)
        elif a is None or isinstance(a, (str, int, float, bool)):
            equal = type(a) is type(b) and a == b
        else:
            raise ValueError("unsupported checkpoint state leaf: " + path)
        if not equal:
            mismatch.append(path)
    for group, selected in (("model", model), ("optimizer", optimizer)):
        for key in selected:
            if key not in left or key not in right:
                mismatch.append(key)
            else:
                compare(left[key], right[key], key, group)
    if not counts["model_tensors"] or not counts["optimizer_tensors"]:
        raise ValueError("model or optimizer tensor state is empty")
    return {"load_mode": "torch.load(map_location=cpu, weights_only=True)",
            "model_keys": model, "optimizer_keys": optimizer, **counts,
            "model_optimizer_exact": not mismatch,
            "checkpoint_iteration_equal": left.get("iter") == right.get("iter") and "iter" in left and "iter" in right,
            "checkpoint_iterations": [left.get("iter"), right.get("iter")],
            "mismatch_count": len(mismatch), "first_mismatches": mismatch[:20],
            "other_top_level_metadata_excluded": sorted(str(key) for key in keys - set(model) - set(optimizer) - {"iter"}),
            "file_hash_equal": sha(first) == sha(second)}


def resource_time(path: Path | None):
    if path is None:
        return {"status": "UNAVAILABLE", "reason": "no independent GNU time -v observation supplied"}
    content = path.read_text(encoding="utf-8")
    match = re.search(r"peak_rss_kib=(\d+)", content)
    elapsed = re.search(r"elapsed_s=([0-9.]+)", content)
    exit_match = re.search(r"exit=(\d+)", content)
    if match is None:
        match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", content)
    if not match:
        raise ValueError("time report lacks max RSS: " + str(path))
    if exit_match is not None and int(exit_match.group(1)) != 0:
        raise ValueError("measured training driver did not exit successfully: " + str(path))
    return {"status": "OBSERVED", "max_rss_bytes": int(match.group(1)) * 1024,
            "elapsed_s": float(elapsed.group(1)) if elapsed else None,
            "source": str(path), "source_sha256": sha(path),
            "scope": "GNU time per-process high-water RSS including waited children; not sum of simultaneous process-tree RSS"}


def resource_budget_checks(*, archive_bytes, chunk_raw_bytes, paired_rss_delta):
    """P2 limits; training driver wall time has no rollout-overhead gate."""
    checks = {
        "numeric_archive_within_2_gib": archive_bytes <= 2 * 1024**3,
        "numeric_chunks_within_64_mib": bool(chunk_raw_bytes) and max(chunk_raw_bytes) <= 64 * 1024**2,
    }
    if paired_rss_delta is not None:
        checks["paired_extra_rss_within_512_mib"] = paired_rss_delta <= 512 * 1024**2
    return checks


def verify(root: Path, on_dir: Path, off_dir: Path, *, on_time=None, off_time=None, expected_cell=None):
    sys.path.insert(0, str(root / "src"))
    from true_rl_agent.runner.numeric_evidence import load_numeric_evidence_manifest
    spec = importlib.util.spec_from_file_location("numeric_smoke_pair_reader", root / "scripts/training_numeric_smoke.py")
    smoke = importlib.util.module_from_spec(spec); spec.loader.exec_module(smoke)
    on, off = [read(folder / "result.json") for folder in (on_dir, off_dir)]
    summaries = [read(folder / "training_summary.json") for folder in (on_dir, off_dir)]
    on_summary, off_summary = summaries
    argv = [read(folder / "executed_train_argv.json") for folder in (on_dir, off_dir)]
    errors, checks = [], {}
    def check(name, condition):
        checks[name] = bool(condition)
        if not condition:
            errors.append(name)
    task = on.get("task")
    if task not in {"stair", "push"}:
        raise ValueError("unknown training task")
    check("completed_on_off", on.get("status") == off.get("status") == "PASS" and on.get("capture") == "on" and off.get("capture") == "off")
    for field in ("task", "source_argv_sha256", "program_source_checksum", "subject_kind", "diagnostic_workload", "device"):
        check("paired_" + field, on.get(field) == off.get(field) and on.get(field) is not None)
    check("execution_argv_only_capture_and_outputs_differ", canonical_train_argv(argv[0]) == canonical_train_argv(argv[1]))
    check("summary_files_match_driver_receipts", all(sha(folder / "training_summary.json") == result.get("training_summary_sha256")
           for folder, result in ((on_dir, on), (off_dir, off))))
    check("initial_policy_hash_equal", isinstance(on.get("initial_policy_state_hash"), str)
          and len(on["initial_policy_state_hash"]) == 64 and on["initial_policy_state_hash"] == off.get("initial_policy_state_hash")
          and all(s["training"].get("initial_policy_state_hash") == on["initial_policy_state_hash"] for s in summaries))
    expected_steps = 8 if task == "stair" else 24
    actual = []
    for summary in summaries:
        runtime = summary["runtime_activation_summary"]
        offset = int(summary["activation_preflight"]["real_transition_dispatcher_calls"]) if task == "stair" else 0
        control_steps = int(runtime["step_calls"]) - offset
        actual.append({"control_steps": control_steps, "excluded_preflight_steps": offset,
                       "env_transitions": control_steps * summary["training"]["num_envs"],
                       "latest_checkpoint_iteration": summary["training"]["latest_checkpoint_iteration"]})
    check("actual_training_steps", all(item["control_steps"] == expected_steps and item["env_transitions"] == expected_steps * 16 for item in actual))
    check("final_checkpoint_iteration", all(item["latest_checkpoint_iteration"] == (0 if task == "stair" else 2) for item in actual))
    check("runtime_reward_observations_equal", on_summary["runtime_activation_summary"] == off_summary["runtime_activation_summary"])
    evidence = on_summary["training"]["numeric_evidence"]
    check("on_receipt_equals_archive_reference", on.get("numeric_evidence") == evidence)
    check("off_capture_disabled", off_summary["training"]["numeric_evidence"] == {"status": "not_enabled"})
    manifest = load_numeric_evidence_manifest(evidence["reference"])
    coverage = evidence["coverage"]
    check("coverage_equals_sealed_manifest", manifest["metadata"] == coverage)
    check("planned_and_observed_control_steps", coverage["planned_control_steps"] == coverage["observed_control_steps"] == expected_steps)
    check("preflight_excluded", coverage["source_step_offset_excluded"] == actual[0]["excluded_preflight_steps"])
    expected_windows = [[1, 2], [3, 4, 5], [6, 7, 8]] if task == "stair" else [list(range(1, 9)), list(range(9, 17)), list(range(17, 25))]
    check("three_windows_exact", len(coverage["windows"]) == 3 and all(w["status"] == "sampled" and w["sampled_steps"] == expected
         and w["sampled_env_transitions"] == len(expected) * 16 for w, expected in zip(coverage["windows"], expected_windows)))
    expected_identity = smoke.expected_training_numeric_identity(on_summary, task=task, argv=argv[0])
    recompute = smoke.check_sampled_rewards(evidence, task=task, expected_identity=expected_identity, require_all_windows=True)
    check("sample_count_equals_coverage", recompute["verified_sampled_control_steps"] == sum(map(len, expected_windows)))
    identity = manifest["identity"]
    cell = identity["resolved_cell_identity"]["cell_variant_id"]
    if expected_cell is not None:
        check("expected_robot_cell", cell == expected_cell)
    option = lambda name: smoke.unique_option_value(argv[0], name)
    check("archive_program_identity", identity.get("program_source_checksum", identity.get("program", {}).get("program_source_checksum")) == on["program_source_checksum"])
    check("archive_training_binding_identity", identity["training_binding_hash"] == option("--training-binding-hash"))
    check("archive_cell_identity", identity["resolved_cell_manifest_hash"] == option("--resolved-cell-manifest-hash"))
    check("archive_backend_identity", identity["backend_execution_environment_verification"]["actual_manifest_hash"] == option("--backend-execution-environment-manifest-hash"))
    check("archive_initial_policy_identity", identity["initial_policy_state_hash"] == on["initial_policy_state_hash"])
    checkpoints = [folder / "smoke_checkpoint.pt" for folder in (on_dir, off_dir)]
    check("checkpoint_files_match_training_receipts", all(sha(path) == summary["checkpoint"]["checksum"] for path, summary in zip(checkpoints, summaries)))
    states = compare_checkpoints(*checkpoints)
    check("model_optimizer_exact", states["model_optimizer_exact"])
    check("checkpoint_iteration_equal", states["checkpoint_iteration_equal"])
    time_paths = [supplied or (Path(str(folder) + ".time") if Path(str(folder) + ".time").is_file() else None)
                  for supplied, folder in ((on_time, on_dir), (off_time, off_dir))]
    times = [resource_time(path) for path in time_paths]
    wall = [on["wall_seconds"], off["wall_seconds"]]
    if not all(isinstance(v, (float, int)) and math.isfinite(v) and v > 0 for v in wall):
        raise ValueError("driver wall seconds are not positive finite measurements")
    resource_complete = all(item["status"] == "OBSERVED" for item in times)
    manifest_path = Path(evidence["reference"]["manifest_path"])
    # The archive reader above checked each NPZ's actual size, dtype and shape.
    # Derive budget quantities from those verified entries, not aggregate claims.
    import numpy as np
    numeric_bytes = sum(chunk["stored_bytes"] for chunk in manifest["chunks"]) + manifest_path.stat().st_size
    chunk_raw_bytes = [sum(math.prod(array["shape"]) * np.dtype(array["dtype"]).itemsize
                           for array in chunk["arrays"].values()) for chunk in manifest["chunks"]]
    rss_delta = times[0]["max_rss_bytes"] - times[1]["max_rss_bytes"] if resource_complete else None
    budget_checks = resource_budget_checks(archive_bytes=numeric_bytes, chunk_raw_bytes=chunk_raw_bytes,
                                           paired_rss_delta=rss_delta)
    for name, passed in budget_checks.items():
        check(name, passed)
    output_bytes = [sum(path.stat().st_size for path in folder.rglob("*") if path.is_file() and not path.is_symlink())
                    for folder in (on_dir, off_dir)]
    return {"schema_version": "true_rl_agent.p2.training_pair_check.v1", "task": task, "cell_variant_id": cell,
            "status": "FAIL" if errors else "PASS" if resource_complete else "INCOMPLETE_RESOURCE_MEASUREMENT",
            "numeric_and_training_invariance_pass": not errors, "checks": checks, "failed_checks": errors,
            "on_dir": str(on_dir), "off_dir": str(off_dir), "actual_training": actual,
            "archive_identity": identity, "sampled_reward_recompute": recompute, "checkpoint_states": states,
            "cost": {"on_wall_seconds": wall[0], "off_wall_seconds": wall[1], "wall_seconds_delta": wall[0] - wall[1],
                "wall_ratio": wall[0] / wall[1], "rss": {"on": times[0], "off": times[1]},
                "max_rss_bytes_delta": rss_delta,
                "added_numeric_archive_bytes": numeric_bytes, "numeric_raw_bytes": sum(chunk_raw_bytes),
                "max_chunk_raw_bytes": max(chunk_raw_bytes), "resource_budget_checks": budget_checks,
                "resource_limits_bytes": {"archive": 2 * 1024**3, "raw_chunk": 64 * 1024**2, "paired_extra_rss": 512 * 1024**2},
                "on_output_tree_bytes": output_bytes[0], "off_output_tree_bytes": output_bytes[1],
                "output_tree_bytes_delta": output_bytes[0] - output_bytes[1],
                "wall_scope": "whole driver including child startup, training, teardown and numeric verification; not isolated training throughput",
                "training_wall_ratio_is_observation_only": True,
                "cost_is_single_pair_observation_not_overhead_estimate": True},
            "limitations": ["Short diagnostic training only; no policy qualification or scientific benefit claim.",
                "Exact-state mismatch fails invariance; one uncontrolled GPU pair cannot attribute causation to capture."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--on", type=Path, required=True)
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--on-time", type=Path)
    parser.add_argument("--off-time", type=Path)
    parser.add_argument("--cell", help="Expected actual cell identity for this pair, e.g. stair.g1_37dof.unified_v3")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = verify(args.project_root.resolve(), args.on.resolve(), args.off.resolve(), on_time=args.on_time, off_time=args.off_time,
                        expected_cell=args.cell)
    except Exception:
        result = {"schema_version": "true_rl_agent.p2.training_pair_check.v1", "status": "FAIL",
                  "numeric_and_training_invariance_pass": False, "failed_checks": ["verification_exception"],
                  "error": traceback.format_exc()}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "failed_checks": result["failed_checks"]}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
