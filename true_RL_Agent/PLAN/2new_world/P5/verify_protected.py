"""Read-only P0 inventory comparison for the protected full Go2 experiment."""
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
inventory = json.loads((ROOT / "PLAN/2new_world/P0/remote_inventory.json").read_text(encoding="utf-8-sig"))
prefix = "outputs/formal_go2_stair_15round_20260901_c6ab8ba/"
expected = {path: item for path, item in inventory["protected_artifacts"].items() if path.startswith(prefix)}
for item in inventory["checkpoint_inventory"]:
    if item["path"].startswith(prefix):
        expected[item["path"]] = item
changes = []
checkpoint_count = 0
for path, item in expected.items():
    file = ROOT / path
    if not file.is_file():
        changes.append({"path": path, "reason": "missing"})
        continue
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    if file.stat().st_size != item["bytes"] or digest != item["sha256"]:
        changes.append({"path": path, "reason": "content_changed"})
    checkpoint_count += path.endswith(".pt")
report = {"status": "PASS" if not changes else "FAIL", "checked_at_unix": time.time(),
          "scope": prefix, "file_count": len(expected), "checkpoint_count": checkpoint_count,
          "total_bytes": sum(item["bytes"] for item in expected.values()), "changes": changes}
(Path(__file__).parent / "protected_after.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
raise SystemExit(bool(changes))
