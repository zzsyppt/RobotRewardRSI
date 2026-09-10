"""Measure independent replay of the frozen baseline. Original writer: Git cd48221."""

import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action", choices=("audit",), required=True)
    args = parser.parse_args()
    project = args.project.resolve(strict=True)
    output = args.output.resolve()
    sys.path[:0] = [str(project / "src"), str(project)]
    from audit_historical_fixture import audit_fixture

    counts = Counter()
    durations = Counter()
    started = {}
    measured = {
        "_build_formal_report", "collect_actual_cost_usage",
        "_validated_job_cost_artifacts", "_resource_summary_evidence",
        "project_public_history", "audit_generation_lifecycle",
        "load_trial_train_receipt", "load_trial_eval_phase_receipt",
        "replay_generation_guard_chain", "replay_stair_generation_guard_chain",
        "load_chain",
    }

    def profile(frame, event, arg):
        name = frame.f_code.co_name
        if name not in measured:
            return
        if event == "call":
            counts[name] += 1
            started[id(frame)] = time.perf_counter_ns()
        elif event == "return":
            begin = started.pop(id(frame), None)
            if begin is not None:
                durations[name] += time.perf_counter_ns() - begin

    scans = Counter()
    def count_access(name, original):
        def counted(path, *args, **kwargs):
            scans[name] += 1
            return original(path, *args, **kwargs)
        return counted

    # Count directory traversal requests once, at iterator construction. Audit
    # wall time includes the same instrumentation overhead in both revisions.
    begin = time.perf_counter_ns()
    completed = False
    try:
        with ExitStack() as stack:
            for name in ("glob", "rglob", "read_bytes"):
                stack.enter_context(patch.object(Path, name, count_access(name, getattr(Path, name))))
            sys.setprofile(profile)
            audit_fixture(project, output)
            completed = True
    finally:
        sys.setprofile(None)
        payload = {
            "schema_version": "p4.session_replay_measurement.v2",
            "project": str(project), "action": args.action,
            "completed": completed,
            "wall_time_ns": time.perf_counter_ns() - begin,
            "profiled_calls": dict(sorted(counts.items())),
            "inclusive_duration_ns": dict(sorted(durations.items())),
            "filesystem_requests": dict(sorted(scans.items())),
        }
        result = output.parent / f"{output.name}-{args.action}-measurement.json"
        result.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
