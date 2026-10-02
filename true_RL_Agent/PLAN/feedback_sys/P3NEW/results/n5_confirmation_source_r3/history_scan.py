"""Read recorded job commitments only; no trajectories, answers, or mutations."""
import hashlib
import json
from pathlib import Path

base = Path("/data/yuyangliu/AGENT/feedback_p3new_20260929")
paths = sorted(set(base.glob("*/*/public_dev_job.json")) | set(base.glob("*/*/*/public_dev_job.json")))
rows, errors = [], []
for path in paths:
    try:
        raw = path.read_bytes()
        data = json.loads(raw)
        plan = data.get("spec", data)["evaluation_plan"]
        rows.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                     "task_id": plan["task_id"], "binding": plan["evaluation_binding"]})
    except (KeyError, ValueError, OSError) as error:
        errors.append({"path": str(path), "error": str(error)})
print(json.dumps({"scope": "Recorded public-dev job commitments at B depth 2 and 3; raw trajectories and model answers not read",
                  "scanned_files": len(paths), "files": rows, "errors": errors}, indent=2))
