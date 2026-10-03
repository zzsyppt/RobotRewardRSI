"""Exercise actual archive/card identity and mandatory-budget rejection paths."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from true_rl_agent.runner.numeric_evidence import NUMERIC_EVIDENCE_REF_SCHEMA
from true_rl_agent.tasks.stair_climbing.trajectory_report import build_stair_trajectory_report


def main():
    sources = json.loads(Path(__file__).with_name("source_lock.json").read_text())["sources"]
    checks = []
    for index, source in enumerate(sources):
        out = ROOT / "outputs/p2a" / source["source_id"]
        replay = json.loads((out / "replay.json").read_text())
        ref = {"schema_version": NUMERIC_EVIDENCE_REF_SCHEMA,
               "manifest_path": str(ROOT / source["manifest_relative_path"]),
               "manifest_sha256": source["manifest_sha256"]}
        other = sources[(index + 1) % len(sources)]
        wrong_ref = {**ref, "manifest_path": str(ROOT / other["manifest_relative_path"]),
                     "manifest_sha256": other["manifest_sha256"]}
        for case, numeric, identity, budget, expected in [
            ("swapped_source", wrong_ref, other["identity"], 8192, "source/trial/checkpoint"),
            ("mandatory_budget", ref, source["identity"], replay["mandatory_bytes"] - 1, "mandatory facts exceed"),
        ]:
            try:
                build_stair_trajectory_report(replay["card_reference"], numeric_reference=numeric,
                    expected_identity=identity, max_bytes=budget)
            except ValueError as error:
                assert expected in str(error), str(error)
                checks.append({"source": source["source_id"], "case": case, "passed": True, "error": str(error)})
            else:
                raise AssertionError((source["source_id"], case, "did not reject"))
    result = {"passed": True, "checks": checks}
    with (ROOT / "outputs/p2a/real_guards.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2); stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
