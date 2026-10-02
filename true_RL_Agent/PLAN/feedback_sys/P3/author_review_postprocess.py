"""Offline review attachments only; leaves the frozen author/export code unchanged."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def attach(stage_root, output):
    review = output / "reviewer"
    withheld = output / "withhold"
    manifest_path = withheld / "postprocess_manifest.json"
    if withheld.exists():
        raise ValueError("review postprocessing destination already exists")
    mapping_path = output / "withhold_from_reviewer.json"
    mapping = read(mapping_path)["mapping"]
    if len(mapping) != 53 or len({r["result_path"] for r in mapping}) != 48:
        raise ValueError("requires the complete frozen 48 + 5 export")
    source = stage_root / "author_parent_reward_supplement_r7.json"
    supplement = read(source)
    if sha(source) != "803d8ecfcf491528d252506ffe1855d750e8301015f16064497f2ab86bd74a89":
        raise ValueError("review parent supplement changed")
    original = {r["result_file"]: r for r in supplement["items"]}
    rows = []
    for entry in mapping:
        item = original.get(entry["result_path"])
        if item is None:
            continue
        if item["result_sha256"] != entry["result_sha256"]:
            raise ValueError("supplement is not bound to this author result")
        parent = item["review_comparison_parent"]
        task = "stair" if item["case_id"].startswith("stair-") else "push"
        if sha(review / f"reference-{task}.py") != parent["parent_source_sha256"]:
            raise ValueError("review reference differs from the proven initial parent")
        index = entry["blind_id"].removeprefix("proposal-")
        rows.append({"blind_id": entry["blind_id"],
            "actual_last_prompt": {"file": f"seen-{index}.json",
                "current_best_source_present": False, "public_feedback_summary_present": False},
            "review_comparison_parent": {"file": f"reference-{task}.py",
                "parent_source_sha256": parent["parent_source_sha256"],
                "actual_first_prompt_sha256": parent["prompt_file_sha256"],
                "frozen_first_prompt_sha256": parent["frozen_first_prompt_sha256"],
                "first_prompt_equals_frozen": parent["frozen_first_prompt_object_equal"],
                "parent_hash_matches_result": parent["parent_hash_matches_result"],
                "use_limit": "Reviewer comparison only, from the bound initial prompt; not present in the actual last repair prompt."},
            "reward_checks": item["reward_checks"]})
    public = {"status": "REVIEW_ONLY_PARENT_REWARD_SUPPLEMENT", "items": rows,
        "source_supplement_sha256": sha(source),
        "functions": [{"symbol": r["symbol"], "file_sha256": r["file_sha256"],
            "function_source_utf8_sha256": r["function_source_utf8_sha256"]} for r in supplement["functions"]],
        "push_sample_limit": {"per_sample_signed_arithmetic": "NOT_AVAILABLE_NOT_PERSISTED",
            "reason": "Saved Push smoke receipts contain absolute extrema and counts, not signed per-sample raw/weighted values."},
        "human_value_gate": "PENDING_REVIEW", "semantic_support": "NOT_GRADED"}
    identity = re.compile(r"\blane-[ab]\b|\bgeneration-\d+\b|\barm[\s_:=\-\"']+[AT]\b|/data/|[A-Za-z]:[\\/]")
    if identity.search(json.dumps(public, ensure_ascii=False)):
        raise ValueError("private identity or path leaked into anonymous supplement")
    withheld.mkdir()
    shutil.copyfile(source, withheld / source.name)
    write(review / "machine_parent_reward_supplement.json", public)
    shutil.copytree(stage_root / "author_review_timelines", review / "timelines")
    changed = []
    for name in ("README_ZH.md", "INDEX_ZH.md"):
        path = review / name
        before = sha(path)
        with path.open("a", encoding="utf-8") as stream:
            stream.write("\n\n补充附件：[父程序与奖励符号核对](machine_parent_reward_supplement.json)。"
                "其中少数项的末次修复输入没有 current_best 或测量反馈，补充父本来自已绑定的首次输入，"
                "仅供审阅比较，不能算作作者末次看见的信息。Push 原回执未保存逐样本带符号数值，相关算术保持 N/A。\n"
                "\n只读数值时间轴位于 [timelines](timelines/)，含 4 张图及来源 sidecar；它们不是原生视觉输入。\n")
        changed.append({"path": "reviewer/" + name, "before_sha256": before, "after_sha256": sha(path)})
    write(manifest_path, {"status": "OFFLINE_REVIEW_ATTACHMENTS", "api_requests": 0,
        "candidate_executions": 0, "postprocessor": {"file": str(Path(__file__)), "sha256": sha(Path(__file__))},
        "frozen_export_mapping_sha256": sha(mapping_path), "source_supplement_sha256": sha(source),
        "anonymous_supplement_sha256": sha(review / "machine_parent_reward_supplement.json"),
        "unique_supplemented_episodes": len(original), "anonymous_supplemented_items": len(rows),
        "modified_generated_readmes": changed,
        "timeline_files": [{"path": "reviewer/timelines/" + p.name, "sha256": sha(p)}
            for p in sorted((review / "timelines").iterdir()) if p.is_file()],
        "frozen_code_modified": False, "original_partial_qa_modified": False,
        "human_value_gate": "PENDING_REVIEW"})
    return {"unique_supplemented_episodes": len(original), "anonymous_supplemented_items": len(rows),
        "timeline_pngs": len(list((review / "timelines").glob("*.png"))), "identity_warnings": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(attach(args.stage_root, args.output)))
