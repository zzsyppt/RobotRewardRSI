"""Read prior public-dev job metadata at the already used bounded depth before creating R4 preflight."""
import hashlib
import json
from pathlib import Path

base = Path("/data/yuyangliu/AGENT/feedback_p3new_20260929")
paths = set()
for pattern in ("*/public_dev_job.json", "*/*/public_dev_job.json", "*/*/*/public_dev_job.json"):
    paths.update(base.glob(pattern))
paths = sorted(paths)
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
print(json.dumps({"scope": "Recorded public-dev jobs at B / 1, 2 or 3 directory levels; includes actual R1/R2/R3 and old preflights; new R4 preflight does not exist yet; metadata only",
                  "scanned_files": len(paths), "files": rows, "errors": errors}, indent=2))
