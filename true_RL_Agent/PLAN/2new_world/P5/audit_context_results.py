"""Compare actual historical prompts and verify retained measured evidence."""
from pathlib import Path
import hashlib
import json
import sys

PROJECT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(PROJECT / "src"), str(PROJECT)]
from true_rl_agent.agent_core.trial_receipts import TrialReceiptRef, load_trial_eval_phase_receipt
from true_rl_agent.tasks.stair_climbing.unified_scoring import build_unified_score_projection


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def wire(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())


def main():
    evidence = PROJECT / "PLAN/2new_world/P5"
    source = PROJECT / "outputs/formal_go2_stair_15round_20260901_c6ab8ba"
    genesis = read(source / "session_genesis.json")
    ledger = [json.loads(line) for line in (source / "ledger.jsonl").read_text().splitlines()]
    mask = genesis["anchor"]["public_result"]["metrics"]["objective_family_mask"]
    generation = ledger[2]
    trial = generation["trials"][0]
    assert trial["public_dev"] and generation["selection"] is None
    guard = next(item for item in generation["private_artifacts"]
                 if item["kind"] == "generation_guard_reference_phase_receipt")
    parent = load_trial_eval_phase_receipt(TrialReceiptRef(kind="eval_phase", path=guard["uri"],
        checksum=guard["checksum"], size_bytes=guard["size_bytes"])).result
    assert parent.checkpoint_hash == generation["parent_current_best"]["checkpoint_hash"]
    scores = [build_unified_score_projection(summary, objective_family_mask=mask)
              for summary in (trial["public_dev"]["public_summary"], parent.task_evidence["summary"])]
    expected_delta = round(scores[0]["j_primary"]["score_stats"]["mean"] - scores[1]["j_primary"]["score_stats"]["mean"], 6)
    expected_guard = round(sum(min(values) for values in zip(*(scores[0][key]["seed_macro_scores"]
        for key in ("j_ood", "c0_id", "c1_id")), strict=True)) / len(scores[0]["j_primary"]["seed_macro_scores"])
        - sum(min(values) for values in zip(*(scores[1][key]["seed_macro_scores"]
        for key in ("j_ood", "c0_id", "c1_id")), strict=True)) / len(scores[1]["j_primary"]["seed_macro_scores"]), 6)
    rows = []
    for label, generations in (("c6", (1, 4, 7, 9, 13, 15)), ("c8", (9,))):
        for number in generations:
            filename = f"generation_{number:02d}.json"
            current = read(evidence / f"replay_context_final_{label}" / filename)
            baseline_path = evidence / f"audit_history_after_{label}" / filename
            baseline = read(baseline_path) if baseline_path.exists() else None
            memory = current["public_history"]["experiment_memory"]
            assert wire(memory) <= 14 * 1024
            assert memory["omitted_experiment_count"] == 0
            assert [item["round_index"] for item in memory["experiments"]] == list(range(1, number))
            assert sum(len(item.get("trial_outcomes", [])) for item in memory["experiments"]) == (number - 1) * 2
            feedback = current["public_feedback_summary"]
            details = [item.get("candidate_id", "reference_anchor") for item in feedback["feedback_entries"]
                       if "stratum_diagnostics" in item["feedback"]["generalization_summary"]]
            assert len(details) <= 2
            if label == "c6" and number >= 4:
                result = next(item for item in memory["experiments"] if item["round_index"] == 3)["trial_outcomes"][0]
                assert result[5:7] == [expected_delta, expected_guard]
            if label == "c6" and number >= 13:
                reasons = [item[-1] for generation_row in memory["experiments"] for item in generation_row.get("trial_outcomes", [])]
                assert {"candidate_runtime_failed", "authoring_provider_failed"}.issubset(reasons)
            rows.append({"session": label, "generation": number, "bytes": wire(current),
                         "previous_bytes": wire(baseline) if baseline else None,
                         "feedback_bytes": wire(feedback), "memory_bytes": wire(memory),
                         "detailed_candidates": details, "retained_trials": (number - 1) * 2})
    branches = []
    for suffix in ("repair", "source_followup"):
        path = evidence / "replay_context_final_c6" / f"generation_13_{suffix}.json"
        value = read(path)
        if "public_feedback_summary" not in value:
            assert value["authoring_context_frame"]["frame_kind"] == "static_repair"
            assert value["provider_or_runtime_repair_feedback"] and value["source_payloads"]
            branches.append({"kind": suffix, "bytes": wire(value),
                             "feedback_scope": "current failed candidate; historical diagnostics not included"})
            continue
        feedback = value["public_feedback_summary"]
        definitions = feedback["diagnostic_table_definitions"]
        for entry in feedback["feedback_entries"]:
            for name in ("factor_diagnostics", "stratum_diagnostics"):
                table = entry["feedback"].get("generalization_summary", {}).get(name)
                if table:
                    assert name in definitions and "columns" in definitions[name]
                    assert all(len(row) == len(definitions[name]["columns"]) for row in table["rows"])
        branches.append({"kind": suffix, "bytes": wire(value), "shared_headers_retained": True})
    report = {"status": "PASS", "source_manifest_sha256": hashlib.sha256((evidence / "replay_source_files.json").read_bytes()).hexdigest(),
              "g3_a_public_dev_delta": expected_delta, "g3_a_guard_delta": expected_guard,
              "parent_checkpoint_hash": parent.checkpoint_hash, "frames": rows, "branches": branches,
              "provider_requests": 0, "training_jobs": 0}
    (evidence / "replay_context_results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
