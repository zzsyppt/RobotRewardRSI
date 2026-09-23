"""Read real c6 evidence through the shared host, scheduler and prompt builder.

No provider, runner, physics or session preparation is invoked. Outputs belong
only to P6; the complete historical source tree is checked before and after.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
SOURCE_FILES = (
    "agent_core/context.py", "agent_core/experiment_memory.py",
    "agent_core/generation_contract.py", "agent_core/generation_lanes.py",
    "agent_core/generation_record.py", "agent_core/generation_runtime.py",
    "agent_core/live_loop.py", "evaluation/acceptance_replay.py",
    "memory/context_builder.py", "prompting/source_catalog.py",
    "tasks/push_recovery/loop_audit.py", "tasks/push_recovery/prompt_adapter.py",
    "tasks/stair_climbing/exploration.py", "tasks/stair_climbing/generation_history.py",
    "tasks/stair_climbing/prompt_adapter.py", "tasks/stair_climbing/selection.py",
    "tasks/stair_climbing/unified_scoring.py",
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot(root):
    return {str(path.relative_to(root)): [path.stat().st_size, path.stat().st_mtime_ns, digest(path)]
            for path in sorted(root.rglob("*")) if path.is_file()}


def injection_routes():
    """Source inspection complements the actual shared-host call below."""
    routes = {}
    for name in ("agent_core/generation_runtime.py", "agent_core/generation_lanes.py",
                 "agent_core/live_loop.py", "tasks/stair_climbing/generation_history.py",
                 "tasks/push_recovery/loop_audit.py"):
        path = ROOT / "src/true_rl_agent" / name
        matches = [{"line": node.lineno, "adapter_expression": ast.unparse(keyword.value)}
                   for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                   if isinstance(node, ast.Call) for keyword in node.keywords
                   if keyword.arg == "phased_evaluation_adapter"]
        assert matches, f"missing bound phased adapter at {name}"
        routes[name] = matches
    return routes


def reconstruct(source, output, report):
    from true_rl_agent.agent_core.generation_contract import GenerationSessionContract
    from true_rl_agent.agent_core.generation_runtime import generation_public_history
    from true_rl_agent.agent_core.ledger import PROVIDER_AUTHORING_MODE, load_ledger, primary_ledger_epoch_hash
    from true_rl_agent.agent_core.trial_receipts import TrialReceiptRef, load_trial_eval_phase_receipt
    from true_rl_agent.candidates.prompt_projection import assert_public_only_prompt_payload
    from true_rl_agent.composition.base import TaskCompositionContext
    from true_rl_agent.evaluation.binding import EvaluationBindingV1, canonical_json_hash
    from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
    from true_rl_agent.memory.context_builder import PublicContextBuilder
    from true_rl_agent.prompt_serialization import (
        MAX_AUTHORING_PROMPT_PAYLOAD_BYTES, canonical_prompt_json, canonical_prompt_json_bytes,
    )
    from true_rl_agent.tasks.stair_climbing.composition import StairClimbingCompositionProvider

    genesis = read(source / "session_genesis.json")
    manifest = read(source / "resolved_cell_manifest.json")
    ledger = load_ledger(source / "ledger.jsonl")
    provider = StairClimbingCompositionProvider()
    comp = provider.resolve(
        values={**provider.cli_defaults(), "cell_variant_id": manifest["cell_variant_id"]},
        context=TaskCompositionContext(project_root=ROOT, output_dir=output / "composition"),
    )
    anchor = genesis["anchor"]
    public, reference = anchor["public_result"], anchor["reference_manifest"]
    binding = EvaluationBindingV1.from_mapping(genesis["evaluation_binding"])
    best = CurrentBestEvalStats.from_reference_public_result_mapping(
        reference_id=reference["reference_id"], reference_manifest_hash=reference["manifest_hash"],
        program_source_checksum=reference["source_checksum"],
        training_binding_hash=anchor["training_binding"]["binding_hash"],
        capability_profile_hash=reference["capability_profile_hash"],
        checkpoint_hash=anchor["checkpoint_hash"], public_result=public,
        acceptance_protocol=comp.task_plugin.config.acceptance_protocol,
        expected_evaluation_binding=binding,
    )
    evidence = {"anchor_kind": "reference_program", "public_result": public,
                "public_result_hash": canonical_json_hash(public)}
    comp.task_plugin.bind_genesis_evidence(evidence)
    reference_source = comp.reference_program_artifact.source_text
    assert "sha256:" + hashlib.sha256(reference_source.encode()).hexdigest() == reference["source_checksum"]
    contract = GenerationSessionContract.from_mapping(read(source / "generation_session_contract.json"))
    phased = comp.task_plugin.phased_evaluation_adapter()
    assert phased is not None
    # These are the existing read-only consumer surfaces of the bound foundation
    # and Genesis. Preparing a new foundation would touch leases/session files.
    foundation = SimpleNamespace(composition=comp, phased_adapter=phased, session_contract=contract)
    genesis_view = SimpleNamespace(initial_current_best=best,
        session_genesis_hash=genesis["session_genesis_hash"],
        expected_initial_policy_state_hash=anchor["initial_policy_state_hash"])
    builder = PublicContextBuilder(task_memory_adapter=comp.task_memory_adapter,
        task_prompt_adapter=comp.task_plugin.prompt_adapter(), phased_evaluation_adapter=phased,
        acceptance_protocol=comp.task_plugin.config.acceptance_protocol,
        initial_current_best=best, session_genesis_hash=genesis["session_genesis_hash"],
        candidate_capability_profile=comp.task_plugin.task_spec.candidate_capability_profile,
        initial_reference_program_source=reference_source, initial_anchor_public_evidence=evidence,
        expected_initial_policy_state_hash=anchor["initial_policy_state_hash"],
        expected_ledger_epoch_hash=primary_ledger_epoch_hash(genesis["session_genesis_hash"]),
        ledger_authoring_mode=PROVIDER_AUTHORING_MODE, generation_session_contract=contract)
    assert builder.phased_evaluation_adapter is foundation.phased_adapter
    report["injection_routes"] = {"evidence_kind": "source inspection, not five independent session launches",
                                  "routes": injection_routes()}
    report["adapter"] = type(phased).__name__
    report["frames"] = []
    for next_generation in (4, 13):
        prefix = ledger[:next_generation - 1]
        print(f"Replaying c6 history before G{next_generation}", flush=True)
        # Exactly the production prepare_generation_inputs history -> scheduler path.
        history = generation_public_history(foundation, genesis=genesis_view, ledger=prefix)
        exploration = comp.task_memory_adapter.build_generation_exploration_plan(
            history, generation_session_contract=contract)
        assert len(history.records) == len(prefix)
        g3 = history.records[2]["generation"]
        raw_g3 = prefix[2]
        assert g3["selection"] is None and g3["trials"][0]["public_dev"] is not None
        assert g3["trials"][1]["public_dev"] is None
        guard_ref = next(item for item in raw_g3["private_artifacts"]
                         if item["kind"] == "generation_guard_reference_phase_receipt")
        parent = load_trial_eval_phase_receipt(TrialReceiptRef(kind="eval_phase", path=guard_ref["uri"],
            checksum=guard_ref["checksum"], size_bytes=guard_ref["size_bytes"])).result
        assert parent.checkpoint_hash == raw_g3["parent_current_best"]["checkpoint_hash"]
        assert g3["parent_public_dev"] == phased.project_public_phase_summary(phase_result=parent)
        cards = {card.trial_id: card for card in exploration.index.cards}
        measured, unmeasured = [cards[trial["trial_id"]] for trial in g3["trials"]]
        summary = measured.public_evidence_summary
        assert measured.parent_current_best_hash == raw_g3["parent_current_best_hash"]
        assert measured.observed_effect == "positive" and measured.value_label == "medium"
        assert measured.revisit_disposition == "eligible"
        assert all(lane["inspired_by_trial_id"] != measured.trial_id for lane in exploration.lane_intents)
        assert "objective" not in unmeasured.public_evidence_summary
        assert "guard_floor" not in unmeasured.public_evidence_summary
        projected = builder.task_prompt_adapter.project_prompt_public_history(
            public_records=history.records, initial_anchor_public_evidence=evidence)
        prompt_g3 = projected.records[2]["generation"]
        prompt_parent = prompt_g3["public_dev_parent_reference"]
        assert prompt_parent["objective_score"] == summary["objective"]["parent"]
        assert prompt_parent["guard_floor_score"] == summary["guard_floor"]["parent"]
        expected = [round(summary[key]["delta"], 6) for key in ("objective", "guard_floor")]
        assert expected == [0.000336, 0.000504], "c6 G3-A measured differences changed"
        assert [prompt_g3["trials"][0]["prompt_outcome"][key]
                for key in ("objective_delta", "guard_delta")] == expected
        assert not {"objective_delta", "guard_delta"}.intersection(prompt_g3["trials"][1]["prompt_outcome"])
        payload = builder.build_prompt(prefix,
            authoring_capability_readiness=comp.authoring_capability_readiness.to_prompt_mapping(),
            candidate_verifier_prompt_rules=comp.candidate_admission_policy.authoring_prompt_rules()).provider_payload
        assert_public_only_prompt_payload(payload)
        memory = payload["public_history"]["experiment_memory"]
        assert canonical_prompt_json_bytes(memory) <= 14 * 1024
        assert canonical_prompt_json_bytes(payload) <= MAX_AUTHORING_PROMPT_PAYLOAD_BYTES
        assert memory["omitted_experiment_count"] == 0
        assert [item["round_index"] for item in memory["experiments"]] == list(range(1, next_generation))
        assert sum(len(item["trial_outcomes"]) for item in memory["experiments"]) == 2 * (next_generation - 1)
        row = next(item for item in memory["experiments"] if item["round_index"] == 3)
        by_lane = {values[0]: dict(zip(memory["trial_outcome_columns"], values, strict=True))
                   for values in row["trial_outcomes"]}
        assert [by_lane["a"][key] for key in ("public_dev_delta", "public_dev_guard_delta")] == expected
        assert by_lane["b"]["public_dev_delta"] is None and by_lane["b"]["public_dev_guard_delta"] is None
        (output / f"generation_{next_generation:02d}.json").write_text(canonical_prompt_json(payload), encoding="utf-8")
        item = {"next_generation": next_generation, "prompt_bytes": canonical_prompt_json_bytes(payload),
                "memory_bytes": canonical_prompt_json_bytes(memory), "retained_trials": 2 * (next_generation - 1),
                "g3_a_trial": measured.trial_id, "g3_b_trial": unmeasured.trial_id,
                "g3_a_delta": expected, "g3_parent_checkpoint": parent.checkpoint_hash,
                "g3_parent_summary_hash": canonical_json_hash(g3["parent_public_dev"]),
                "g3_a_scheduler_effect": measured.observed_effect,
                "g3_a_revisit_disposition": measured.revisit_disposition,
                "scheduler_lane_intents": list(exploration.lane_intents),
                "same_parent_and_deltas": True, "unmeasured_sibling_has_no_delta": True,
                "private_safety_passed": True}
        report["frames"].append(item)
        print(json.dumps(item), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if not output.is_relative_to(ROOT / "PLAN/2new_world/P6") or output.is_relative_to(source):
        parser.error("output must be a new P6 directory outside the historical source")
    output.mkdir(parents=True, exist_ok=False)
    before = snapshot(source)
    (output / "protected_before.json").write_text(json.dumps(before, indent=2) + "\n", encoding="utf-8")
    paths = [ROOT / "src/true_rl_agent" / name for name in SOURCE_FILES] + [Path(__file__).resolve()]
    report = {"status": "RUNNING", "source": str(source), "provider_requests": 0, "training_jobs": 0,
              "physical_evaluation_jobs": 0, "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in paths}}
    start = time.monotonic()
    try:
        reconstruct(source, output, report)
        report["status"] = "PASS"
    except Exception as exc:
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        after = snapshot(source)
        report.update(elapsed_s=time.monotonic() - start, protected_file_count=len(before),
                      protected_bytes=sum(item[0] for item in before.values()),
                      source_unchanged=after == before)
        if after != before:
            report["status"] = "FAIL"
        (output / "protected_after.json").write_text(json.dumps(after, indent=2) + "\n", encoding="utf-8")
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: report[key] for key in ("status", "source_unchanged", "elapsed_s")}), flush=True)
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
