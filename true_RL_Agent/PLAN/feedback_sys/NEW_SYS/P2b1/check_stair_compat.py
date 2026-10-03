"""Compare rebuilt frozen card and existing P2a artifacts without duplicating GB archives."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "outputs/p2b"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def main():
    frozen = {"evidence.jsonl": "58f1014f557847c5ee32a3c03e4bebf4becdc2bc35552c6ac948a5fcc576af6e",
        "policy_card.json": "d83ddf0b8ee28e8d00705de81029930474b47ba230532596d50aabd2e9b0d9a2",
        "policy_card_zh.md": "c5589b8b79efd605c5d5d305a84ba58f246de3fbba55a39d48cff0d7f90c90ac"}
    for name, expected in frozen.items():
        assert sha(OUT / "stair_frozen_card" / name) == expected
    audit = json.loads((ROOT / "PLAN/feedback_sys/NEW_SYS/P2a/verification.json").read_text(encoding="utf-8"))
    sources = []
    for source in audit["sources"]:
        checked = []
        for item in source["files"]:
            path = ROOT / item["path"]
            assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], path
            checked.append(item)
        sources.append({"source_id": source["name"], "files": checked})
    gates = json.loads((OUT / "stair_frozen_verification.json").read_text(encoding="utf-8"))
    result = {"status": "PASS", "frozen_card_sha256": frozen, "p2a_sources": sources,
        "frozen_facts": {k: {"counts": v["counts"], "fact_count": v["fact_count"], "compared_scalar_fields": v["compared_scalar_fields"]} for k, v in gates["reports"].items()},
        "scope": "frozen card rebuilt; 247/309 checked; P2a production artifacts hash-identical, no new three-source rebuild"}
    (OUT / "stair_compat.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "sources": len(sources), "files": sum(len(s["files"]) for s in sources), "facts": result["frozen_facts"]}, indent=2))


if __name__ == "__main__":
    main()
