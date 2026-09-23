"""CPU-only P2 check of one Push capture on/off pair and its sealed archives.

Import the exact acceptance checkout with --project-root. This neither executes
a policy nor upgrades an old receipt; recomputations are separate derived files.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import traceback


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_values(left, right, *, atol=1e-6):
    """Discrete values must match; floating observations use absolute tolerance."""
    mismatches, count, maximum = [], 0, 0.0
    def walk(a, b, path):
        nonlocal count, maximum
        if isinstance(a, dict) and isinstance(b, dict):
            equal = set(a) == set(b)
            for key in sorted(set(a) & set(b)):
                walk(a[key], b[key], path + "." + key)
        elif isinstance(a, list) and isinstance(b, list):
            equal = len(a) == len(b)
            for index, (av, bv) in enumerate(zip(a, b)):
                walk(av, bv, path + f"[{index}]")
        elif isinstance(a, float) and isinstance(b, float):
            error = abs(a - b)
            maximum = max(maximum, error) if math.isfinite(error) else maximum
            equal = math.isfinite(a) and math.isfinite(b) and error <= atol
        else:
            equal = type(a) is type(b) and a == b
        if not equal:
            count += 1
            if len(mismatches) < 40:
                mismatches.append(path)
    walk(left, right, "root")
    return {"equal": count == 0, "mismatch_count": count, "first_mismatch_paths": mismatches,
            "max_absolute_float_difference": maximum, "absolute_float_tolerance": atol}


def resource_time(path):
    if not Path(path).is_file():
        return {"status": "MISSING", "path": str(path)}
    match = re.search(r"peak_rss_kib=(\d+) elapsed_s=([\d.]+) exit=(\d+)", Path(path).read_text())
    if match is None or int(match[1]) <= 0 or int(match[3]) != 0 or not math.isfinite(float(match[2])) or float(match[2]) <= 0:
        raise ValueError("invalid or failed resource wrapper receipt: " + str(path))
    return {"status": "OBSERVED", "path": str(path), "sha256": sha(path),
            "peak_rss_bytes": int(match[1]) * 1024, "elapsed_s": float(match[2]), "exit": int(match[3])}


def verify(project_root: Path, on_dir: Path, off_dir: Path, *, output: Path):
    sys.path.insert(0, str(project_root / "src"))
    from true_rl_agent.evaluation.binding import canonical_json_hash
    from true_rl_agent.evaluation.phases import EvaluationPhaseResult
    from true_rl_agent.runner.numeric_evidence import load_numeric_evidence_manifest
    from true_rl_agent.tasks.push_recovery.numeric_capture import validate_push_numeric_reference, recompute_push_numeric_evidence
    from true_rl_agent.tasks.push_recovery.metrics import project_push_split

    errors, checks = [], {}
    def check(name, condition):
        checks[name] = bool(condition)
        if not condition:
            errors.append(name)
    reports = [read(folder / "result.json") for folder in (on_dir, off_dir)]
    on, off = reports
    check("completed_capture_on_off", all(r.get("status") == "PASS" and r.get("mode") == "run" for r in reports)
          and on.get("capture") == "on" and off.get("capture") == "off")
    for field in ("cell", "source_training_manifest_hash", "evaluation_manifest_hash", "training_summary_sha256",
                  "source_manifest_sha256", "checkpoint_sha256", "preserved_training_binding_hash", "evaluation_plan_hash"):
        check("paired_" + field, on.get(field) is not None and on.get(field) == off.get(field))
    check("checkpoint_admitted_and_bytes_checked", all(r.get("checkpoint_bytes_checked") is True
          and r.get("checkpoint_runner_admission") == "passed" for r in reports))
    check("scope_public_dev_only", all(r.get("train_jobs") == r.get("accept_jobs") == r.get("sealed_jobs")
          == r.get("provider_requests") == 0 and r.get("qualification_claimed") is False for r in reports))
    phase_paths = [folder / "evaluation/public_dev/summary.json" for folder in (on_dir, off_dir)]
    raw = [read(path) for path in phase_paths]
    private = [read(folder / "public_dev_result.json") for folder in (on_dir, off_dir)]
    jobs = [read(folder / "public_dev_job.json") for folder in (on_dir, off_dir)]
    phase_results = [EvaluationPhaseResult.from_private_mapping(value) for value in private]
    check("same_evaluation_plan", jobs[0] == jobs[1])
    seed_sets, public = [], []
    for side, report, phase, receipt, job, parsed in zip(("on", "off"), reports, raw, private, jobs, phase_results):
        plan = job["evaluation_plan"]
        expected_plan_hash = canonical_json_hash(plan)
        check(side + "_phase_identity", phase["status"] == "pass" and phase["evaluation_phase"] == receipt["phase"] == job["phase"] == "public_dev"
              and phase["evaluation_plan_hash"] == receipt["evaluation_plan_hash"] == report["evaluation_plan_hash"] == expected_plan_hash
              and phase["job_id"] == receipt["job_id"] == job["job_id"]
              and receipt["checkpoint_hash"] == plan["checkpoint"]["checksum"] == report["checkpoint_sha256"]
              and receipt["training_binding_hash"] == plan["training_binding_hash"] == report["preserved_training_binding_hash"]
              and receipt["program_source_checksum"] == plan["program_source_checksum"])
        check(side + "_sealed_task_evidence", phase["task_evidence"] == receipt["task_evidence"]
              and phase["task_evidence_hash"] == parsed.task_evidence_hash)
        check(side + "_execution_cost_bound", phase["evaluation_execution_cost"] == receipt["execution_cost"]
              and report["evaluation_execution_cost"] == receipt["execution_cost"])
        seeds = phase["task_evidence"]["seed_summaries"]
        seed_sets.append(seeds)
        check(side + "_seed_and_episode_denominators", len(seeds) == report["public_dev_seeds"] == plan["public_seed_count"] == 3
              and [s["run"]["run_index"] for s in seeds] == [1, 2, 3]
              and all(s["run"]["split"] == "public_dev" and s["status"] == "pass"
                  and s["environment"]["num_envs"] == report["eval_num_envs"] == plan["eval_num_envs"] == 64
                  and len(s["per_episode"]) == 64 and s["evidence"]["all_first_episodes_closed"] is True for s in seeds))
        projection_errors = []
        projected = project_push_split(seeds, split_name="public_dev", errors=projection_errors)
        public.append(projected)
        check(side + "_public_projection", not projection_errors and report["public_dev"] == projected)
    seed_comparisons = []
    for index, (captured, baseline) in enumerate(zip(*seed_sets), 1):
        filtered = [{k: v for k, v in seed.items() if k not in {"created_at_utc", "numeric_evidence_ref"}}
                    for seed in (captured, baseline)]
        comparison = compare_values(*filtered)
        seed_comparisons.append({"run_index": index, **comparison})
        check(f"seed_{index}_on_off_behavior", comparison["equal"])
    public_comparison = compare_values(*public)
    check("on_off_public_summary", public_comparison["equal"])
    check("off_has_no_numeric_reference", all("numeric_evidence_ref" not in s for s in seed_sets[1]))
    refs = [s["numeric_evidence_ref"] for s in seed_sets[0]]
    check("driver_reference_list_bound", read(on_dir / "numeric_references.json") == refs)

    archives, recomputes, failure_counts = [], [], Counter()
    for seed, reference in zip(seed_sets[0], refs):
        phase, plan = raw[0], jobs[0]["evaluation_plan"]
        expected = {key: phase[key] for key in ("job_id", "task_id", "subject_kind", "subject_id",
            "subject_manifest_hash", "evaluation_binding", "backend_execution_environment_verification") if key in phase}
        expected.update(evaluation_plan_hash=phase["evaluation_plan_hash"], checkpoint_hash=private[0]["checkpoint_hash"],
            training_binding_hash=plan["training_binding_hash"], program_source_checksum=plan["program_source_checksum"],
            runtime_dynamics_hash=phase["environment_source_preflight"]["runtime_dynamics_hash"],
            resolved_cell_manifest_hash=on["evaluation_manifest_hash"])
        validate_push_numeric_reference(reference, seed, expected_identity=expected)
        manifest = load_numeric_evidence_manifest(reference)
        recompute = recompute_push_numeric_evidence(reference)
        index = seed["run"]["run_index"]
        check(f"seed_{index}_numeric_recompute", recompute["matches_runtime"] is True
              and recompute["episode_coverage"] == "complete" and recompute["episodes"] == len(seed["per_episode"])
              and recompute["valid_control_steps"] == seed["execution_cost"]["closed_first_episode_control_steps"])
        recompute_path = output.with_name(output.stem + f"_seed_{index:02d}_numeric_recompute.json")
        recompute_path.write_text(json.dumps(recompute, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        recomputes.append({"run_index": index, "path": str(recompute_path), "sha256": sha(recompute_path),
                           "matches_runtime": recompute["matches_runtime"], "checks": recompute["checks"]})
        failure_counts.update(record["failure_label"] for record in seed["per_episode"])
        stored = manifest["stored_chunk_bytes"] + Path(reference["manifest_path"]).stat().st_size
        check(f"seed_{index}_archive_budget", stored <= 2 * 1024**3
              and all(c["raw_bytes"] <= 64 * 1024**2 for c in manifest["chunks"]))
        archives.append({"run_index": index, "reference": reference, "stored_bytes": stored, "raw_bytes": manifest["raw_bytes"],
            "capture_seconds": manifest["metadata"]["capture_seconds"], "staging_raw_bytes": manifest["metadata"]["staging_raw_bytes"],
            "array_dtypes": {name: sorted({c["arrays"][name]["dtype"] for c in manifest["chunks"]})
                             for name in manifest["chunks"][0]["arrays"]}})
    costs = [value["evaluation_execution_cost"]["splits"]["public_dev"] for value in raw]
    phase_seconds = [c["wall_time_s"] for c in costs]
    if not all(isinstance(v, (float, int)) and math.isfinite(v) and v > 0 for v in phase_seconds):
        raise ValueError("phase wall times are not finite positive observations")
    check("phase_rollout_overhead_within_25_percent", phase_seconds[0] <= phase_seconds[1] * 1.25)
    times = [resource_time(Path(str(folder) + ".time")) for folder in (on_dir, off_dir)]
    resources_observed = all(t["status"] == "OBSERVED" for t in times)
    rss_delta = times[0]["peak_rss_bytes"] - times[1]["peak_rss_bytes"] if resources_observed else None
    if resources_observed:
        check("paired_peak_rss_delta_within_512_mib", rss_delta <= 512 * 1024**2)
    capacity = [r["gpu_capacity_seconds"] for r in reports]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in capacity):
        raise ValueError("GPU capacity duration is not a positive measurement")
    return {"schema_version": "true_rl_agent.p2.push_pair_check.v1", "cell": on["cell"],
        "status": "FAIL" if errors else "PASS" if resources_observed else "INCOMPLETE_RESOURCE_MEASUREMENT",
        "checks": checks, "failed_checks": errors, "project_root": str(project_root),
        "numeric_reducer_sha256": sha(project_root / "src/true_rl_agent/tasks/push_recovery/numeric_capture.py"),
        "on_dir": str(on_dir), "off_dir": str(off_dir), "phase_summary_sha256": [sha(p) for p in phase_paths],
        "seed_comparisons": seed_comparisons, "public_comparison": public_comparison,
        "episodes": sum(len(s["per_episode"]) for s in seed_sets[0]), "failure_counts": dict(failure_counts),
        "numeric_recompute": recomputes, "archives": archives,
        "cost": {"phase_wall_s": {"on": phase_seconds[0], "off": phase_seconds[1]},
            "phase_wall_ratio": phase_seconds[0] / phase_seconds[1], "outer_resources": {"on": times[0], "off": times[1]},
            "paired_peak_rss_bytes_delta": rss_delta, "gpu_capacity_seconds": {"on": capacity[0], "off": capacity[1], "total": sum(capacity)},
            "total_stored_archive_bytes": sum(a["stored_bytes"] for a in archives),
            "measurement_scope": "phase wall includes rollout, archive compression/publication and summary aggregation; outer RSS is GNU time per-process high-water RSS including waited children, not sum of simultaneous process-tree RSS"},
        "limitations": ["One fixed-checkpoint pair per cell; no policy qualification or benefit claim.",
            "RSS difference is a paired GNU time high-water proxy, not simultaneous process-tree total or isolated allocation profiling.",
            "Recompute checks stored control-step pose operands and delivery measurements; it does not replay physics or unrecorded force substeps."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--on", type=Path, required=True)
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = verify(args.project_root.resolve(), args.on.resolve(), args.off.resolve(), output=args.output.resolve())
    except Exception:
        result = {"schema_version": "true_rl_agent.p2.push_pair_check.v1", "status": "FAIL",
                  "failed_checks": ["verification_exception"], "error": traceback.format_exc()}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "failed_checks": result["failed_checks"]}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
