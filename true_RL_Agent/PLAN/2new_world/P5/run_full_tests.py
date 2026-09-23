"""Record one complete discovery run; no test filtering or retry aggregation."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[3]
label = sys.argv[1]
if label not in {"local", "target", "audit_local", "audit_target", "replay_local", "replay_target"}:
    raise SystemExit("unknown acceptance run label")
directory = Path(__file__).parent
receipt = directory / f"{label}_final_tests.json"
details = {
    "status": "RUNNING",
    "started_at": datetime.now(timezone.utc).isoformat(),
    "python": sys.executable,
    "command": "unittest discover -v -s tests -t .",
}
receipt.write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
start = time.monotonic()
log = directory / f"{label}_final_tests.log"
with log.open("w", encoding="utf-8") as stream:
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "unittest", "discover", "-v", "-s", "tests", "-t", "."],
        cwd=root, env={**os.environ, "PYTHONPATH": str(root / "src")},
        stdout=stream, stderr=subprocess.STDOUT,
    )
text = log.read_text(encoding="utf-8")
counts = re.search(r"Ran (\d+) tests? in", text)
terminal = text[text.rfind("\nRan "):]
details.update(
    status="PASS" if result.returncode == 0 and counts is not None else "FAIL",
    tests_run=int(counts[1]) if counts else None,
    returncode=result.returncode,
    **{key: (int(match[1]) if (match := re.search(key + r"=(\d+)", terminal)) else 0) if counts else None
       for key in ("failures", "errors", "skipped")},
    elapsed_s=time.monotonic() - start,
    finished_at=datetime.now(timezone.utc).isoformat(),
)
receipt.write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
print(json.dumps(details))
raise SystemExit(0 if details["status"] == "PASS" else 1)
