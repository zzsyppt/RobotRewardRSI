"""Bound old-development source-operand audit; no model/simulator invocation."""
import ast
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
P3NEW = HERE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def function_sha(path, name):
    source = path.read_text(encoding="utf-8")
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()


def main():
    target = HERE / "source_operand_link_batch.json"
    if target.exists(): raise ValueError("fresh result required")
    source = P3NEW / "verify_stair_confirmation_arithmetic.py"
    old_source = HERE / "verifier_before_operand_link.py"
    assert sha(old_source) == "efdcf46c370b0121e1819f67389d3b7c28082ed9586bd02b7e357165c0ad79ae"
    unchanged = {name: function_sha(source, name) for name in
                 ("decode_column", "decode_events", "mismatch_examples", "verify_stair_episode")}
    assert all(value == function_sha(old_source, name) for name, value in unchanged.items())
    spec = importlib.util.spec_from_file_location("stair_arithmetic", source)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    original = P3NEW / "results/n5_arithmetic_split_development_r1/stair"
    watched = {str(p.resolve()): sha(p) for p in original.iterdir() if p.is_file()}
    watched[str(source.resolve())] = sha(source)
    watched[str(old_source.resolve())] = sha(old_source)
    batch = json.loads((original / "batch_12_episodes.json").read_bytes())
    export = P3NEW / "results/raw_export_development_related_v1"
    packing = P3NEW / "results/n5_review_capacity_development_r2/packing_receipt.json"
    rows = []
    for old in batch["episodes"]:
        request = Path(old["request_path"])
        assert sha(request) == old["request_sha256"]
        watched[str(request.resolve())] = sha(request)
        raw, full = verifier.materials_from_http(request)
        assert verifier.digest(raw) == old["raw_numeric_material_sha256"]
        assert verifier.digest(full) == old["full_derivation_material_sha256"]
        arithmetic = verifier.verify_stair_episode(raw, full)
        expected = arithmetic.pop("recomputed_fields")
        assert arithmetic == old["result"] and verifier.digest(expected) == old["recomputed_fields_sha256"]
        linked = verifier.verify_stair_operand_sources(raw,
            episode_header_path=export / "episodes" / raw["episode"]["episode_id"] / "episode.json",
            raw_export_manifest_path=export / "manifest.json", packing_receipt_path=packing)
        watched.update(linked["input_sha256"])
        rows.append({"packet": old["packet"], "episode_id": old["episode_id"], "steps": old["steps"],
                     "cell": old["cell"], "surface_kind": old["surface_kind"], "request_sha256": old["request_sha256"],
                     "original_arithmetic_and_recomputed_fields_unchanged": True,
                     "original_arithmetic_status": arithmetic["status"], "source_operand_link": linked})
        print(json.dumps({"packet": old["packet"], "status": linked["status"],
                          "failed_n": linked.get("failed_n"), "unchecked": linked["unchecked"]}))
    assert len(rows) == 12
    assert all(sha(Path(p)) == value for p, value in watched.items())
    counts = Counter(row["source_operand_link"]["status"] for row in rows)
    passed = counts == {"PASS_SOURCE_OPERAND_LINK": 12}
    record = {"status": "PASS_SOURCE_OPERAND_LINK_WITH_ORIGINAL_ARITHMETIC_UNCHANGED" if passed else "REQUIRES_REVIEW",
              "scope": "12 fixed old-development Stair episodes; source NPZ to five exported model columns, not N5 acceptance",
              "episode_count": 12, "status_counts": dict(counts), "total_steps": sum(r["steps"] for r in rows),
              "direct_column_checks": sum(len(r["source_operand_link"].get("direct_operand_comparisons", [])) for r in rows),
              "direct_scalar_values_compared": sum(c["value_count"] for r in rows for c in r["source_operand_link"].get("direct_operand_comparisons", [])),
              "old_verifier_sha256": sha(old_source), "current_verifier_sha256": sha(source),
              "event_arithmetic_functions_source_unchanged": unchanged,
              "old_results_and_all_source_inputs_unchanged": True, "input_sha256": watched,
              "helper_sha256": sha(Path(__file__)), "new_api_calls": 0, "gpu_jobs": 0, "ssh_calls": 0,
              "closure": "Force norms and history max independently recomputed from original declared NPZ vectors; surface recomputed using task normative query and bound raw position/origin/geometry.",
              "remaining_boundaries": "No independent collision physics, asset identity, termination reclassification, definition semantics or N5 acceptance; original arithmetic receipts keep their historical unchecked declarations unchanged.",
              "episodes": rows}
    target.write_bytes(json.dumps(record, ensure_ascii=False, indent=2).encode() + b"\n")
    print(json.dumps({k: record[k] for k in ("status", "episode_count", "direct_column_checks", "direct_scalar_values_compared", "current_verifier_sha256")}))
    print("receipt_sha256", sha(target))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
