"""CPU-only recheck of saved P2 v1 training pairs using a supplied current tree.

Reads original receipts, argv, archives and checkpoints. Writes only compact
derived reports under the separate --output-dir; never starts a training job.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--pair-verifier", type=Path, required=True)
    parser.add_argument("--legacy-reports", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(1)
    root = args.project_root.resolve(strict=True)
    output = args.output_dir.resolve()
    reports = [args.legacy_reports / f"training_{task}_{robot}_verification.json"
               for task in ("push", "stair") for robot in ("g1", "go2")]
    spec = importlib.util.spec_from_file_location("legacy_pair_verifier", args.pair_verifier)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    source_paths = [Path(__file__).resolve(), args.pair_verifier.resolve(),
                    root / "scripts/training_numeric_smoke.py",
                    root / "src/true_rl_agent/runner/training_numeric_capture.py",
                    root / "src/true_rl_agent/runner/numeric_evidence.py"]
    source_hashes = {str(path): sha(path) for path in source_paths}
    rows = []
    for report_path in reports:
        start = time.perf_counter()
        old = read(report_path)
        on, off = Path(old["on_dir"]), Path(old["off_dir"])
        if any(output == folder.resolve() or output.is_relative_to(folder.resolve()) for folder in (on, off)):
            raise ValueError("derived output must not be inside a legacy training directory")
        row = {"case": report_path.stem, "source_report": str(report_path),
               "source_report_sha256": sha(report_path), "on_dir": str(on), "off_dir": str(off),
               "cell_variant_id": old["cell_variant_id"], "new_simulator_runs": 0, "new_model_calls": 0}
        try:
            summary = read(on / "training_summary.json")
            evidence = summary["training"]["numeric_evidence"]
            manifest_path = Path(evidence["reference"]["manifest_path"])
            manifest = read(manifest_path)
            paths = [report_path, manifest_path]
            paths += [folder / name for folder in (on, off)
                      for name in ("result.json", "training_summary.json", "executed_train_argv.json")]
            on_time = Path(old["cost"]["rss"]["on"]["source"])
            off_time = Path(old["cost"]["rss"]["off"]["source"])
            paths += [on_time, off_time]
            before = {str(path): sha(path) for path in paths}
            current = verifier.verify(root, on, off, on_time=on_time, off_time=off_time,
                                      expected_cell=old["cell_variant_id"])
            comparisons = {key: current[key] == old[key] for key in (
                "actual_training", "archive_identity", "sampled_reward_recompute", "checkpoint_states", "checks")}
            unchanged = before == {str(path): sha(path) for path in paths}
            old_schema = manifest["metadata"]["schema_version"]
            row.update(archive_schema_version=old_schema, current_pair_status=current["status"],
                       checks=current["checks"], failed_checks=current["failed_checks"],
                       sampled_reward_recompute=current["sampled_reward_recompute"],
                       actual_training=current["actual_training"], checkpoint_states=current["checkpoint_states"],
                       equivalent_to_legacy_report=comparisons, original_file_hashes=before,
                       original_files_unchanged=unchanged, original_file_count=len(before))
            row["status"] = "PASS" if (current["status"] == "PASS" and all(comparisons.values())
                                      and unchanged and old_schema == "true_rl_agent.training_numeric_capture.v1") else "FAIL"
        except Exception:
            row.update(status="FAIL", error=traceback.format_exc())
        row["cpu_recheck_wall_seconds"] = time.perf_counter() - start
        output.mkdir(parents=True, exist_ok=True)
        write(output / f"{report_path.stem}_recheck.json", row)
        rows.append({key: row.get(key) for key in ("case", "status", "archive_schema_version",
                     "original_files_unchanged", "cpu_recheck_wall_seconds")})
        print(json.dumps(rows[-1]), flush=True)
    summary = {"schema_version": "true_rl_agent.p2.legacy_training_recheck.v1",
               "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
               "source_hashes": source_hashes, "cases": rows, "new_simulator_runs": 0, "new_model_calls": 0,
               "scope": "Recompute saved v1 data with current checker; not new training or v2 simulator coverage."}
    write(output / "summary.json", summary)
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
