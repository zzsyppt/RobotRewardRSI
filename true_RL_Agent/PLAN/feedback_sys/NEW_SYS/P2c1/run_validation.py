"""Capture real CPU command stdout, stderr, hashes and peak memory; no network."""
import argparse
from contextlib import redirect_stdout
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("p2b1_commands", Path(__file__).parents[1]/"P2b1/run_final.py")
commands = importlib.util.module_from_spec(spec)
spec.loader.exec_module(commands)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["tests","real"], required=True)
    args = parser.parse_args()
    commands.OUT = args.output.resolve()
    commands.OUT.mkdir(parents=True, exist_ok=False)
    if args.mode == "tests":
        jobs = [("cycles", ["-m","unittest","tests.unit.test_trajectory_cycles","tests.unit.test_stair_cycle_supplement","-v"]),
                ("compatibility", ["-m","unittest","tests.unit.test_stair_trajectory_report",
                    "tests.unit.test_stair_trajectory_pipeline","tests.unit.test_trajectory_card",
                    "tests.unit.test_trajectory_measurement","tests.unit.test_trajectory_cohort",
                    "tests.unit.test_push_trajectory_pipeline","tests.unit.test_push_trajectory","-v"])]
    else:
        jobs = []
        lock = json.loads((Path(__file__).parents[1]/"P2a/source_lock.json").read_text(encoding="utf-8"))
        for source in lock["sources"]:
            sid = source["source_id"]
            out = commands.OUT/sid
            jobs += [("build_"+sid, ["scripts/build_stair_cycle_supplement.py", "--manifest", str(ROOT/source["manifest_relative_path"]),
                "--manifest-sha256", source["manifest_sha256"], "--source-id", sid, "--output", str(out),
                "--base-public", str(ROOT/"outputs/p2a"/sid/"public.json")]),
                ("oracle_"+sid, ["PLAN/feedback_sys/NEW_SYS/P2c1/audit_raw.py", "--source", sid,"--directory",str(out)])]
    receipts = []
    for name, argv in jobs:
        with (commands.OUT/(name+".wrapper.txt")).open("w",encoding="utf-8") as stream, redirect_stdout(stream):
            receipt = commands.run(name,argv)
        receipts.append(receipt)
        (commands.OUT/"command_receipts.json").write_text(json.dumps(receipts,indent=2)+"\n",encoding="utf-8")
        print(name, "exit", receipt["exit_code"], "seconds", round(receipt["wall_seconds"],3),flush=True)
        if receipt["exit_code"]:
            print((commands.OUT/(name+".stderr.txt")).read_text(encoding="utf-8"))
            sys.exit(receipt["exit_code"])
