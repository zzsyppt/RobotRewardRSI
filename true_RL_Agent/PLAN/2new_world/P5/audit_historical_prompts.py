"""Reconstruct production prompts from immutable historical evidence; no execution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


def reconstruct(root, output, generations, *, baseline=False, branch_frames=False):
    from true_rl_agent.agent_core.generation_contract import GenerationSessionContract
    from true_rl_agent.agent_core.ledger import primary_ledger_epoch_hash, PROVIDER_AUTHORING_MODE
    from true_rl_agent.composition.base import TaskCompositionContext
    from true_rl_agent.evaluation.binding import EvaluationBindingV1, canonical_json_hash
    from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
    from true_rl_agent.memory.context_builder import PublicContextBuilder
    from true_rl_agent.prompt_serialization import canonical_prompt_json, canonical_prompt_json_bytes
    from true_rl_agent.tasks.stair_climbing.composition import StairClimbingCompositionProvider
    from true_rl_agent.tasks.stair_climbing import prompt_adapter

    genesis = json.loads((root / "session_genesis.json").read_text())
    manifest = json.loads((root / "resolved_cell_manifest.json").read_text())
    ledger = [json.loads(line) for line in (root / "ledger.jsonl").read_text().splitlines()]
    provider = StairClimbingCompositionProvider()
    comp = provider.resolve(values={**provider.cli_defaults(), "cell_variant_id": manifest["cell_variant_id"]},
        context=TaskCompositionContext(project_root=ROOT, output_dir=output / "composition"))
    anchor = genesis["anchor"]
    public = anchor["public_result"]
    reference = anchor["reference_manifest"]
    binding = EvaluationBindingV1.from_mapping(genesis["evaluation_binding"])
    best = CurrentBestEvalStats.from_reference_public_result_mapping(
        reference_id=reference["reference_id"], reference_manifest_hash=reference["manifest_hash"],
        program_source_checksum=reference["source_checksum"],
        training_binding_hash=anchor["training_binding"]["binding_hash"],
        capability_profile_hash=reference["capability_profile_hash"],
        checkpoint_hash=anchor["checkpoint_hash"], public_result=public,
        acceptance_protocol=comp.task_plugin.config.acceptance_protocol, expected_evaluation_binding=binding)
    evidence = {"anchor_kind": "reference_program", "public_result": public,
                "public_result_hash": canonical_json_hash(public)}
    comp.task_plugin.bind_genesis_evidence(evidence)
    source = comp.reference_program_artifact.source_text
    assert "sha256:" + hashlib.sha256(source.encode()).hexdigest() == reference["source_checksum"]
    contract = GenerationSessionContract.from_mapping(json.loads((root / "generation_session_contract.json").read_text()))
    builder = PublicContextBuilder(task_memory_adapter=comp.task_memory_adapter,
        task_prompt_adapter=comp.task_plugin.prompt_adapter(), acceptance_protocol=comp.task_plugin.config.acceptance_protocol,
        initial_current_best=best, session_genesis_hash=genesis["session_genesis_hash"],
        candidate_capability_profile=comp.task_plugin.task_spec.candidate_capability_profile,
        initial_reference_program_source=source, initial_anchor_public_evidence=evidence,
        expected_initial_policy_state_hash=anchor["initial_policy_state_hash"],
        expected_ledger_epoch_hash=primary_ledger_epoch_hash(genesis["session_genesis_hash"]),
        ledger_authoring_mode=PROVIDER_AUTHORING_MODE, generation_session_contract=contract)
    report = {"source": str(root), "genesis_sha256": hashlib.sha256((root / "session_genesis.json").read_bytes()).hexdigest(),
              "ledger_sha256": hashlib.sha256((root / "ledger.jsonl").read_bytes()).hexdigest(), "frames": []}
    for generation in generations:
        def build(**kwargs):
            return builder.build_prompt(ledger[:generation - 1],
                authoring_capability_readiness=comp.authoring_capability_readiness.to_prompt_mapping(),
                candidate_verifier_prompt_rules=comp.candidate_admission_policy.authoring_prompt_rules(), **kwargs).provider_payload
        row = {"generation": generation}
        try:
            payload = build()
        except ValueError as exc:
            row["production_error"] = str(exc)
            if not baseline or "stored prompt feedback" not in str(exc):
                raise
            # Baseline-only A/B: expose the pre-fix memory cost independently of
            # the old redundant feedback equality check. Never used for execution.
            def derived(result, *, config, objective_family_mask_provider):
                return prompt_adapter.build_unified_stair_prompt_feedback(
                    suite=config.ordered_evaluation_suite, semantics=config.evaluation_semantics,
                    summary=result["metrics"]["public_dev"],
                    objective_family_mask=objective_family_mask_provider()) if result.get("metrics", {}).get("public_dev") else result.get("metrics", {}).get("prompt_feedback", {})
            with patch.object(prompt_adapter, "_validated_unified_prompt_feedback", derived):
                payload = build()
            row["diagnostic_only_feedback_derivation"] = True
        (output / f"generation_{generation:02d}.json").write_text(canonical_prompt_json(payload))
        memory = payload.get("public_history", {}).get("experiment_memory", {})
        row.update(bytes=canonical_prompt_json_bytes(payload),
            memory_wire_bytes=len(json.dumps(memory, ensure_ascii=False, separators=(",", ":")).encode()),
            memory_metered_bytes=canonical_prompt_json_bytes(memory),
            retained_generations=[e["round_index"] for e in memory.get("experiments", [])],
            omitted_generations=memory.get("omitted_experiment_count"),
            runtime_candidate=payload.get("runtime_feedback_summary", {}).get("source_candidate_id"),
            detailed_latest=[e.get("candidate_id") for e in payload.get("public_feedback_summary", {}).get("feedback_entries", [])
                             if "latest_nonaccepted" in e.get("selection_basis", [])],
            retained_trials=sum(len(e.get("trial_outcomes", e.get("lanes", []))) for e in memory.get("experiments", [])),
            source_identities=[i.get("feedback_identity") for i in payload.get("source_catalog", {}).get("items", [])
                               if "latest_nonaccepted" in i.get("logical_id", "")])
        report["frames"].append(row)
        if branch_frames and generation == 13:
            from true_rl_agent.agent_core.authoring_response import SourceRequestResponse
            from true_rl_agent.agent_core.source_request_service import resolve_source_request
            from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef

            # A recorded verifier failure supplies the real repair source/reason.
            attempt = root / "authoring" / ledger[generation - 1]["generation_id"] / "lane-b/authoring_attempts/attempt_000001/attempt_result.json"
            continuation = json.loads(attempt.read_text())["continuation_state"]
            candidate = continuation["latest_failed_candidate"]
            candidate_ref = CandidatePackageRef(candidate_id=candidate["candidate_id"], path=candidate["path"],
                change_set=CandidateChangeSet(frozenset(candidate["changed_components"])))
            repair = build(latest_failed_candidate=candidate_ref,
                candidate_repair_feedback=continuation["repair_feedback"], frame_kind="static_repair",
                recoverable_sources=tuple(payload["source_payloads"]))
            # There was no source request in these two experiments. Exercise the
            # real service using one allowed ID from this actual late prompt;
            # this is a controlled branch, not an invented historical event.
            catalog = payload["source_catalog"]["items"]
            logical_id = next(item["logical_id"] for item in catalog if item.get("requestable"))
            resolution = resolve_source_request(prompt_payload=payload,
                response=SourceRequestResponse(logical_ids=(logical_id,), rationale="Capacity audit of a real historical source",
                    source_catalog_logical_ids=tuple(item["logical_id"] for item in catalog)),
                source_request_count=0, source_request_bytes_returned=0)
            assert resolution.source_payloads
            followup = build(candidate_repair_feedback=resolution.feedback,
                source_payloads=resolution.source_payloads, frame_kind="source_followup",
                recoverable_sources=tuple(payload["source_payloads"]))
            report["branch_frames"] = []
            for label, value, provenance in (
                ("repair", repair, "actual recorded verifier failure and failed candidate source"),
                ("source_followup", followup, "controlled source request against this real late public history"),
            ):
                path = output / f"generation_{generation:02d}_{label}.json"
                path.write_text(canonical_prompt_json(value))
                report["branch_frames"].append({"file": path.name, "bytes": canonical_prompt_json_bytes(value), "provenance": provenance})
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generations", type=int, nargs="+", default=[1, 7, 9, 13, 15])
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--branch-frames", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = reconstruct(args.source, args.output, args.generations, baseline=args.baseline, branch_frames=args.branch_frames)
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
