"""Add the missed shallow commitments; preserve the initial 14-file scan."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
argv = ["ssh", "-o", "BatchMode=yes", "humanoid-server", "/usr/bin/python3", "-"]
with (HERE / "history_cohorts.complete.json").open("xb") as stdout, (HERE / "history_cohorts.complete.stderr.txt").open("xb") as stderr:
    result = subprocess.run(argv, input=(HERE / "history_scan_complete.py").read_bytes(), stdout=stdout, stderr=stderr, timeout=120)
(HERE / "history_cohorts.complete.command.json").write_text(json.dumps({"argv": argv, "returncode": result.returncode}, indent=2)+"\n")
assert result.returncode == 0
complete = json.loads((HERE / "history_cohorts.complete.json").read_bytes())
assert not complete["errors"]
prior = json.loads((HERE.parent / "n5_confirmation_source_r2/history_cohorts.complete.json").read_bytes())
before = json.loads((HERE / "history_cohorts.before.json").read_bytes())
by_path = {row["path"]: row for row in complete["files"]}
assert all(by_path.get(row["path"]) == row for row in prior["files"])
assert all(by_path.get(row["path"]) == row for row in before["files"])
print(json.dumps({"scanned_files": complete["scanned_files"], "all_prior_28_commitments_unchanged": True,
                  "initial_14_scan_preserved": True, "sha256": hashlib.sha256((HERE / "history_cohorts.complete.json").read_bytes()).hexdigest()}))
