"""Record only the semantic repair's relevant CPU checks; preserve old receipts."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("p2b1_commands",Path(__file__).parents[1]/"P2b1/run_final.py")
commands = importlib.util.module_from_spec(spec)
spec.loader.exec_module(commands)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    commands.OUT = args.output.resolve()
    commands.OUT.mkdir(parents=True,exist_ok=False)
    source = json.loads((Path(__file__).parents[1]/"P2a/source_lock.json").read_text(encoding="utf-8"))["sources"][0]
    assert source["source_id"] == "k1"
    jobs = [("tests",["-m","unittest","tests.unit.test_trajectory_cycles","tests.unit.test_stair_cycle_supplement","-v"]),
        ("build_k1",["scripts/build_stair_cycle_supplement.py","--manifest",str(ROOT/source["manifest_relative_path"]),
            "--manifest-sha256",source["manifest_sha256"],"--source-id","k1","--output",str(commands.OUT/"k1_rebuild"),
            "--base-public",str(ROOT/"outputs/p2a/k1/public.json")]),
        ("semantic_replay",["PLAN/feedback_sys/NEW_SYS/P2c1/semantic_repair_replay.py","--output",str(commands.OUT)]),
        ("compatibility",["PLAN/feedback_sys/NEW_SYS/P2c1/verify_compatibility.py"])]
    receipts = []
    for name,argv in jobs:
        receipt = commands.run(name,argv)
        receipts.append(receipt)
        (commands.OUT/"command_receipts.json").write_text(json.dumps(receipts,indent=2)+"\n",encoding="utf-8")
        print(name,"exit",receipt["exit_code"],"seconds",round(receipt["wall_seconds"],3),flush=True)
        if receipt["exit_code"]:
            sys.exit(receipt["exit_code"])
