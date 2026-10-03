"""Package executed receipts and immutable evidence references, no new measures."""
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).parent
OUT = ROOT/"outputs/p2c1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def reference(path):
    return {"path":path.relative_to(ROOT).as_posix(),"sha256":sha(path),"bytes":path.stat().st_size}


if __name__ == "__main__":
    receipts = []
    for directory in (OUT/"tests_01",OUT/"real_01",OUT/"compat"):
        for receipt in read(directory/"command_receipts.json"):
            assert receipt["exit_code"] == 0
            for channel in ("stdout","stderr"):
                path = directory/(receipt["name"]+"."+channel+".txt")
                assert sha(path) == receipt[channel+"_sha256"]
                receipt[channel] = reference(path)
            receipts.append(receipt)
    tests = {}
    for name in ("cycles","compatibility"):
        log = (OUT/"tests_01"/(name+".stderr.txt")).read_text(encoding="utf-8")
        count = int(re.search(r"Ran (\d+) tests? in",log).group(1))
        assert re.search(r"^OK$",log,re.M)
        tests[name] = {"tests":count,"pass":count,"fail":0,"error":0,"skip":0}
    sources = []
    for sid in ("k1","k2","k3"):
        directory = OUT/"real_01"/sid
        manifest, oracle = read(directory/"manifest.json"),read(directory/"oracle.json")
        assert oracle["status"] == "PASS"
        # The output records also retain exact role order and per-run provenance.
        raw_manifest = read(Path(manifest["source_manifest_ref"]["manifest_path"]))
        names = ["robot_0.support."+x for x in raw_manifest["metadata"]["runtime_projection"]["resolved_roles"]["support_effector"]]
        expected_entities = [[name] for name in names]+[[names[i],names[j]] for i in range(len(names)) for j in range(i+1,len(names))]
        run_counts = {}
        with (directory/"cycles.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                assert [r["entity_ids"] for r in row["records"]] == expected_entities
                seed = raw_manifest["metadata"]["seeds"][row["run_index"]-1]["runner_seed"]
                assert row["runner_seed"] == seed
                run_counts[row["run_index"]] = run_counts.get(row["run_index"],0)+1
        report = read(directory/"report.json")
        assert report["runs"] == [{"run_index":i,"runner_seed":raw_manifest["metadata"]["seeds"][i-1]["runner_seed"],"episodes":n}
                                  for i,n in run_counts.items()]
        for name,info in manifest["artifacts"].items():
            assert sha(directory/name) == info["sha256"]
        sources.append({"source_id":sid,"oracle":oracle,"manifest":manifest,
                        "summary":report["summary"],"preview":read(directory/"preview_receipt.json"),
                        "files":[reference(p) for p in sorted(directory.iterdir()) if p.is_file()]})
    old = read(HERE.parent/"P2b2/verification.json")
    old_log = ROOT/"outputs/p2b2/final/regression.stderr.txt"
    log = old_log.read_text(encoding="utf-8")
    errata = []
    for key, value in old["unresolved_or_skipped_tests"].items():
        parts = key.split(".")
        assert parts[-1] == parts[-2]
        fixed = ".".join(parts[:-1])
        module, cls, method = fixed.rsplit(".",2)
        path = ROOT/(module.replace(".","/")+".py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        node = next(c for c in tree.body if isinstance(c,ast.ClassDef) and c.name == cls)
        function = next(f for f in node.body if isinstance(f,ast.FunctionDef) and f.name == method)
        assert f"{method} ({fixed})" in log
        errata.append({"historical_key":key,"correct_test_id":fixed,"historical_status":value,
                       "source":path.relative_to(ROOT).as_posix(),"line":function.lineno})
    code = [ROOT/"src/true_rl_agent/evaluation/trajectory_translation/cycles.py",
            ROOT/"src/true_rl_agent/tasks/stair_climbing/trajectory_cycles.py",
            ROOT/"scripts/build_stair_cycle_supplement.py",
            ROOT/"tests/unit/cycle_oracle.py",ROOT/"tests/unit/test_trajectory_cycles.py",
            ROOT/"tests/unit/test_stair_cycle_supplement.py",*sorted(HERE.glob("*.py")),
            HERE/"FREEZE_ZH.md",HERE/"DEVELOPMENT_ZH.md",HERE/"REAL_EXAMPLES.json"]
    result = {"schema_version":"stair.p2c1.verification.v1","status":"development_pass_root_gate_pending",
        "baseline_commit":"ca140cfd1e6084118e0c36a4c28f8fe97a2dc294",
        "pre_raw_rule_freeze_commit":"c7aa200b62f697d1b16327c1278e2f89c067f60d",
        "branch":"codex/trajectory-translator-upgrade","only_writer":"root development thread",
        "stage":"P2c1 only; offline current-marker cycles/phase; no nominal-limit or terrain implementation",
        "catalog":"trajectory.p0.measurements.v3 additive to v2; old v2 default and replay preserved",
        "tests":tests,"commands":receipts,"sources":sources,
        "old_artifact_checks":read(OUT/"compat/old_artifacts.stdout.txt"),
        "P2b2_test_id_errata":{"old_verification":reference(HERE.parent/"P2b2/verification.json"),
            "original_log":reference(old_log),"corrections":errata,"historical_files_modified":False},
        "code_and_deliverables":[reference(p) for p in code], "key_diff":reference(OUT/"key_diff.patch"),
        "cost":{"external_model_calls":0,"gpu_runs":0,"training_runs":0,"server_writes":0,
                "measured_command_wall_seconds_sum":sum(r["wall_seconds"] for r in receipts),
                "maximum_python_peak_working_set_bytes":max(r["memory"]["peak_working_set_bytes"] for r in receipts)},
        "limits":["No K5 or P3 pilot behavior read; no statistical holdout/generalization claim.",
            "All four new optional sentences omitted in each real 8192B preview; new full offline report retained.",
            "P2c2/C3 and production delivery await separate root authorization; not permanently unavailable.",
            "Ruff absent in venv; no dependency installation. 154 applicable tests passed without skips.",
            "Windows normal-permission initial status clean; sandbox deep-path D was not deletion."],
        "parent_gate":"commit/push current branch then stop writing for independent review"}
    (HERE/"verification.json").write_bytes((json.dumps(result,indent=2,ensure_ascii=False)+"\n").encode("utf-8"))
    print(json.dumps({"status":result["status"],"tests":sum(r["tests"] for r in tests.values()),
                      "raw_episodes":sum(s["oracle"]["episodes"] for s in sources),
                      "raw_records":sum(s["oracle"]["measurement_records_checked"] for s in sources),
                      "cost":result["cost"]},indent=2))
