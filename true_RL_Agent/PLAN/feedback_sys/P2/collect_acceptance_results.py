"""Collect compact, hash-linked P2 reports from the isolated acceptance root."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def linked_file(path):
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def manual_source_binding(root, case, paired_case):
    """Recover source evidence from sealed outputs, never invent launch argv."""
    result_path = root / case / "result.json"
    phase_path = root / case / "evaluation/public_dev/summary.json"
    paired_phase_path = root / paired_case / "evaluation/public_dev/summary.json"
    result, phase, paired_phase = map(read, (result_path, phase_path, paired_phase_path))
    reference = result.get("numeric_evidence_ref")
    if reference is None:
        return {"status": "not_available", "reason": "no bound numeric archive"}
    manifest_path = Path(reference["manifest_path"])
    manifest_link = linked_file(manifest_path)
    if manifest_link["sha256"] != reference["manifest_sha256"]:
        raise ValueError("manual run archive manifest checksum mismatch")
    identity = read(manifest_path)["identity"]
    environment = phase["backend_execution_environment_verification"]
    source = phase["environment_source_preflight"]["source_manifest_hash"]
    if not (identity["backend_execution_environment_verification"] == environment
            == paired_phase["backend_execution_environment_verification"]
            and source == paired_phase["environment_source_preflight"]["source_manifest_hash"]):
        raise ValueError("manual run source does not match its verified pair")
    return {"status": "bound_outputs_match_pair", "source_inferred_from":
            "archive identity and on/off shared environment source preflight; launch argv remains unrecorded",
            "environment_source_manifest_hash": source, "backend_execution_environment_verification": environment,
            "archive_manifest": manifest_link, "result": linked_file(result_path),
            "phase_summary": linked_file(phase_path), "paired_phase_summary": linked_file(paired_phase_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True)
    reports, paired_cases = [], {}
    for path in sorted(args.root.glob("*verification.json")):
        data = read(path)
        pair = data.get("source_paths") or {k: data.get(k + "_dir") for k in ("on", "off")}
        if all(pair.get(k) for k in ("on", "off")):
            on, off = (Path(pair[k]).name for k in ("on", "off"))
            paired_cases.update({on: off, off: on})
        data["full_report"] = linked_file(path)
        # Correct descriptions in derived reports; original reports stay hash-linked.
        if data.get("schema_version") == "true_rl_agent.p2.push_pair_check.v1":
            data["cost"]["measurement_scope"] = (
                "phase wall includes rollout, archive compression/publication and summary aggregation; "
                "outer RSS is GNU time per-process high-water RSS including waited children, "
                "not sum of simultaneous process-tree RSS")
            data["limitations"] = [line.replace("paired process-tree peak proxy", "paired GNU time high-water proxy")
                                   for line in data["limitations"]]
        if "behavior" in data:
            episodes = data["behavior"].pop("episodes")
            data["behavior"]["episode_details"] = {
                "count": len(episodes), "location": "full_report.behavior.episodes"}
        (args.output / path.name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        reports.append({"name": path.name, "status": data["status"], **linked_file(path)})
    runs = []
    launches = {p.name.removesuffix(".launch.json"): p for p in args.root.glob("*.launch.json")}
    names = set(launches) | {p.stem for p in args.root.glob("*.time")}
    for case in sorted(names):
        path = launches.get(case)
        run = read(path) if path else {"case": case, "launch_receipt": "not_recorded"}
        case = run["case"]
        timing = args.root / (case + ".time")
        match = re.search(r"peak_rss_kib=(\d+) elapsed_s=([\d.]+) exit=(\d+)", timing.read_text()) if timing.exists() else None
        run["resource_measurement"] = ({"peak_rss_kib": int(match[1]), "elapsed_s": float(match[2]),
            "exit_code": int(match[3]), **linked_file(timing)} if match else {"status": "not_finished_or_missing"})
        run["receipt"] = linked_file(path) if path else None
        run["role"] = "acceptance_attempt" if case in paired_cases else "not_used_for_acceptance"
        if path is None and case in paired_cases:
            run["source_binding_evidence"] = manual_source_binding(args.root, case, paired_cases[case])
        if run.get("returncode") or run["resource_measurement"].get("exit_code"):
            run["role"] = "failed_attempt_included_in_cost"
        run["logs"] = [linked_file(log) for log in (
            args.root / (case + ".log"), args.root / case / "process/stdout.log",
            args.root / case / "process/stderr.log") if log.is_file()]
        runs.append(run)
    elapsed = sum(run["resource_measurement"].get("elapsed_s", 0) for run in runs)
    result = {"root": str(args.root), "reports": reports, "runs": runs,
              "cost": {"measured_single_gpu_driver_seconds": elapsed,
                       "gpu_capacity_hours_upper_proxy": elapsed / 3600,
                       "scope": "sum of whole-driver GNU time elapsed, one reserved GPU per run; includes startup, teardown and failed/preliminary runs",
                       "provider_requests": 0, "model_cost": 0,
                       "incomplete_measurements": sum("elapsed_s" not in r["resource_measurement"] for r in runs)}}
    (args.output / "run_inventory.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"reports": len(reports), "runs": len(runs), "cost": result["cost"]}))


if __name__ == "__main__":
    main()
