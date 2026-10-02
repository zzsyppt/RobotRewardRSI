"""Collect the verified per-item reviews from the two workflow journals (data transfer only, no scoring)."""
import json
import sys
from pathlib import Path

J = Path("C:/Users/lover/.claude/projects/D--humanoid-agent/af0e9d9a-08cb-4c4a-ae47-89f04e037846/subagents/workflows")
RUNS = ["wf_107a41ec-0b0", "wf_27d4c22f-6af"]
OUT = Path(sys.argv[1])

reviews = {}
for run in RUNS:
    rows = [json.loads(l) for l in (J / run / "journal.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    labels = {r["agentId"]: r["label"] for r in rows if r.get("type") == "started"}
    for r in rows:
        if r.get("type") != "result" or not r.get("result"):
            continue
        stage, pid = labels[r["agentId"]].split(":")
        reviews.setdefault(pid, {})[stage] = r["result"]
OUT.write_text(json.dumps(reviews, ensure_ascii=False, indent=1), encoding="utf-8")
missing = [p for p, v in reviews.items() if "verify" not in v]
print(len(reviews), "items;", "missing verify:", missing)
