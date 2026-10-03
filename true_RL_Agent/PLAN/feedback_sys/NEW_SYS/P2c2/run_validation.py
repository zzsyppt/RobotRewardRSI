"""Record exact local CPU commands and original stdout/stderr/peak-memory receipts."""
import argparse
from contextlib import redirect_stdout
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("commands", Path(__file__).parents[1]/"P2b1/run_final.py")
commands = importlib.util.module_from_spec(spec); spec.loader.exec_module(commands)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["tests", "real"], required=True); a = p.parse_args()
    commands.OUT = a.output.resolve(); commands.OUT.mkdir(parents=True, exist_ok=False)
    if a.mode == "tests":
        jobs = [("limits", ["-m", "unittest", "tests.unit.test_joint_limits", "-v"]),
                ("compatibility", ["-m", "unittest", "tests.unit.test_trajectory_cycles", "tests.unit.test_stair_cycle_supplement",
                 "tests.unit.test_stair_trajectory_report", "tests.unit.test_stair_trajectory_pipeline",
                 "tests.unit.test_trajectory_card", "tests.unit.test_trajectory_measurement", "tests.unit.test_trajectory_cohort",
                 "tests.unit.test_push_trajectory_pipeline", "tests.unit.test_push_trajectory", "-v"]),
                ("frozen_artifacts", ["PLAN/feedback_sys/NEW_SYS/P2c1/verify_compatibility.py"])]
    else:
        jobs = []
        sources = json.loads((Path(__file__).parents[1]/"P2a/source_lock.json").read_text(encoding="utf-8"))["sources"]
        for s in sources:
            sid = s["source_id"]; out = commands.OUT/sid
            jobs += [("build_"+sid, ["scripts/build_stair_joint_limit_supplement.py", "--manifest", str(ROOT/s["manifest_relative_path"]),
                       "--manifest-sha256", s["manifest_sha256"], "--source-id", sid, "--output", str(out),
                       "--base-public", str(ROOT/"outputs/p2a"/sid/"public.json")]),
                     ("oracle_"+sid, ["PLAN/feedback_sys/NEW_SYS/P2c2/audit_raw.py", "--source", sid, "--directory", str(out)])]
    receipts = []
    for name, argv in jobs:
        with (commands.OUT/(name+".wrapper.txt")).open("x", encoding="utf-8") as f, redirect_stdout(f):
            result = commands.run(name, argv)
        receipts.append(result)
        (commands.OUT/"command_receipts.json").write_text(json.dumps(receipts, indent=2)+"\n", encoding="utf-8")
        print(name, "exit", result["exit_code"], "seconds", round(result["wall_seconds"], 3), flush=True)
        if result["exit_code"]:
            print((commands.OUT/(name+".stderr.txt")).read_text(encoding="utf-8"))
            sys.exit(result["exit_code"])
