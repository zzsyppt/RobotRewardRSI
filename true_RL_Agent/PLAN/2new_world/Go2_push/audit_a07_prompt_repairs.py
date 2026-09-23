"""Replay F10/F11 from immutable A06 evidence; no provider or simulation jobs."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from true_rl_agent.composition.cell_manifest import ResolvedCellManifestV1
from true_rl_agent.evaluation.acceptance_replay import project_public_history
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.prompting.authoring_frame import (
    build_authoring_context_frame, project_sections_for_authoring_frame, with_final_prompt_budget,
)
from true_rl_agent.prompting.payload import PROMPT_PAYLOAD_SECTION_ORDER
from true_rl_agent.prompting.public_safety import assert_public_only_prompt_payload
from true_rl_agent.prompting.serialization import canonical_prompt_json_bytes
from true_rl_agent.prompting.source_catalog import PROMPT_SOURCE_SECTION_KEYS, finalize_prompt_source_sections
from true_rl_agent.prompting.task_sections import validate_provider_task_prompt_receipt
from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
from true_rl_agent.tasks.push_recovery.phased_adapter import PushPhasedEvaluationAdapter
from true_rl_agent.tasks.push_recovery.prompt_adapter import PushRecoveryPromptAdapter

BASE = Path(__file__).resolve().parent
SESSION = BASE / "attempt_06/session"
OUTPUT = BASE / "attempt_07"
evidence = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    evidence[str(path.relative_to(ROOT))] = digest(path)
    return json.loads(path.read_text(encoding="utf-8"))


def finish_frame(sections, kind, omissions=()):
    values = deepcopy(sections)
    values.pop("authoring_context_frame", None)
    values["authoring_context_frame"] = build_authoring_context_frame(
        frame_kind=kind, section_values=values, section_order=PROMPT_PAYLOAD_SECTION_ORDER,
        omitted_sections=omissions)
    values["authoring_context_frame"] = with_final_prompt_budget(
        values["authoring_context_frame"], prompt_payload=values)
    assert_public_only_prompt_payload(values)
    validate_provider_task_prompt_receipt(
        values["authoring_context_frame"]["budget_metadata"]["provider_task_sections"], prompt_payload=values)
    return values


def prompt(generation, lane, attempt):
    return SESSION / f"authoring/go2-push-5rounds-20260923-a06-generation-{generation:03d}/lane-{lane}/authoring_attempts/attempt_{attempt:06d}/prompt_payload.json"


if __name__ == "__main__":
    OUTPUT.mkdir(exist_ok=True)
    manifest = ResolvedCellManifestV1.from_mapping(read(SESSION / "resolved_cell_manifest.json"))
    config = PushRecoveryTaskConfig.from_manifest(manifest)
    adapter = PushRecoveryPromptAdapter(config)
    genesis = read(SESSION / "session_genesis.json")
    ledger_path = SESSION / "ledger.jsonl"
    evidence[str(ledger_path.relative_to(ROOT))] = digest(ledger_path)
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-history-only", action="store_true")
    args = parser.parse_args()
    history_export = OUTPUT / "a06_gen01_public_history.json"
    if os.name == "nt":
        exported = read(history_export)
        assert exported["source_ledger_sha256"] == digest(ledger_path)
        assert exported["session_genesis_hash"] == genesis["session_genesis_hash"]
        records = exported["records"]
    else:
        history = project_public_history(ledger[:1],
            initial_current_best=CurrentBestEvalStats.from_mapping(ledger[0]["parent_current_best"]),
            expected_session_genesis_hash=genesis["session_genesis_hash"],
            acceptance_protocol=config.acceptance_protocol(),
            expected_initial_policy_state_hash=genesis["anchor"]["initial_policy_state_hash"],
            expected_authoring_mode=ledger[0]["authoring_mode"],
            expected_ledger_epoch_hash=ledger[0]["ledger_epoch_hash"],
            phased_evaluation_adapter=PushPhasedEvaluationAdapter(config, manifest.resolved_cell_task_protocol_hash))
        records = list(history.records)
        history_export.write_text(json.dumps({"source_ledger_sha256": digest(ledger_path),
            "session_genesis_hash": genesis["session_genesis_hash"], "records": records},
            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.export_history_only:
        print(json.dumps({"exported_history": str(history_export), "records": len(records)}))
        raise SystemExit(0)

    previous = read(prompt(2, "b", 1))
    state = read(prompt(2, "b", 1).with_name("attempt_result.json"))["continuation_state"]["selected_round_public_context"]
    task = adapter.build_prompt_sections(public_records=records, public_context=state)
    updated = deepcopy(previous)
    updated.update(task)
    current = task["public_feedback_summary"].pop("current_best")
    assert task["public_feedback_summary"] == previous["public_feedback_summary"]
    task["public_feedback_summary"]["current_best"] = current
    assert current["source_round_id"] == ledger[0]["resulting_current_best"]["accepted_round_id"]
    assert current["program_source_checksum"] == ledger[0]["resulting_current_best"]["program_source_checksum"]
    raw_dev = read(SESSION / f"runner/{current['source_round_id']}/eval/public_dev/summary.json")
    counts = Counter()
    strata = {}
    for seed in raw_dev["task_evidence"]["seed_summaries"]:
        counts.update(seed["aggregate"]["failure_breakdown"]["counts"])
        for name, row in seed["aggregate"]["stratum_results"].items():
            strata.setdefault(name, Counter()).update({key: row[key] for key in ("episodes", "successes")})
    assert current["public_dev"]["failure_breakdown"]["counts"] == dict(counts)
    assert current["latest_public_score"] == counts["success_recovered"] / sum(counts.values())
    assert {name: {key: row[key] for key in ("episodes", "successes")}
        for name, row in current["public_dev"]["stratum_results"].items()} == {name: dict(row) for name, row in strata.items()}
    updated = finish_frame(updated, "candidate_draft")
    changed_sections = [key for key in previous if previous[key] != updated[key]]
    assert changed_sections == ["public_feedback_summary", "authoring_context_frame"]
    current_text = json.dumps(current)
    for token in ("public_accept", "sealed_validation", "checkpoint", "/data/", "seed_results", "seed_summaries"):
        assert token not in current_text

    repair_rows = []
    for generation in (3, 5):
        previous_repair = read(prompt(generation, "a", 3))
        prior_sources = read(prompt(generation, "a", 2))["source_payloads"]
        repair_input = deepcopy(previous_repair)
        repair_input["source_payloads"] = [*prior_sources, *repair_input["source_payloads"]]
        projected, omissions = project_sections_for_authoring_frame(
            repair_input, frame_kind="static_repair", section_order=PROMPT_PAYLOAD_SECTION_ORDER)
        projected.update(finalize_prompt_source_sections({key: projected[key] for key in PROMPT_SOURCE_SECTION_KEYS}))
        repaired = finish_frame(projected, "static_repair", omissions)
        expected_ids = {item["logical_id"] for item in repair_input["source_payloads"]}
        assert {item["logical_id"] for item in repaired["source_payloads"]} == expected_ids
        assert len(expected_ids) == 3
        assert "public_feedback_summary" not in repaired and "public_history" not in repaired
        for item in repaired["source_payloads"]:
            assert item["content_sha256"] == hashlib.sha256(item["content"].encode()).hexdigest()
        feedback = repaired["provider_or_runtime_repair_feedback"]
        assert feedback == previous_repair["provider_or_runtime_repair_feedback"]
        assert feedback["stage"] == "historical_exact_exclusion"
        assert feedback["reason"] == "historical_exact_program_repeat"
        repair_rows.append({"generation": generation,
            "previous_prompt_bytes": canonical_prompt_json_bytes(previous_repair),
            "updated_prompt_bytes": canonical_prompt_json_bytes(repaired),
            "retained_source_ids": sorted(expected_ids),
            "retained_source_bytes": {item["logical_id"]: len(item["content"].encode()) for item in repaired["source_payloads"]},
            "historical_feedback_unchanged": True, "private_or_new_sources_added": False})
    assert all(digest(ROOT / path) == expected for path, expected in evidence.items())
    report = {"status": "PASS", "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "A06 real ledger public-history replay and production task projection/frame assembly; no provider/simulator execution",
        "history_replay": "Production ledger replay uses original Linux receipt URIs; Windows consumes its hash-bound public-history export.",
        "F10": {"changed_sections": changed_sections, "previous_prompt_bytes": canonical_prompt_json_bytes(previous),
            "updated_prompt_bytes": canonical_prompt_json_bytes(updated), "current_best": current,
            "latest_nonaccepted_feedback_unchanged": True, "parent_dev_matches_original_rollout": True,
            "parent_private_phase_or_checkpoint_fields_visible": False},
        "F11": {"replayed_repair_frames": repair_rows,
            "future_source_request_avoided": "Not claimed; the three required source bodies are now directly available.",
            "verifier_anchor_changed": False},
        "original_evidence_unchanged": True, "evidence_sha256": evidence,
        "product_sources_sha256": {str(path.relative_to(ROOT)): digest(path) for path in (
            ROOT / "src/true_rl_agent/tasks/push_recovery/prompt_adapter.py",
            ROOT / "src/true_rl_agent/prompting/authoring_frame.py")},
        "provider_calls": 0, "training_jobs": 0, "evaluation_jobs": 0}
    (OUTPUT / "prompt_repairs_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "F10_bytes": report["F10"]["updated_prompt_bytes"] - report["F10"]["previous_prompt_bytes"],
        "F11_bytes": {str(row["generation"]): row["updated_prompt_bytes"] - row["previous_prompt_bytes"] for row in repair_rows}, "original_evidence_files": len(evidence)}))
