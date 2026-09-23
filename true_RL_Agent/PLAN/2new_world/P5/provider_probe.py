"""P5 real authoring probe from frozen P3 public evidence, with no RL launches."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from true_rl_agent.agent_core.authoring_loop import CandidateAuthoringLoop
    from true_rl_agent.agent_core.openai_compatible_provider import OpenAICompatibleCandidateProvider, OpenAICompatibleProviderConfig
    from true_rl_agent.composition.base import TaskCompositionContext
    from true_rl_agent.evaluation.binding import EvaluationBindingV1, canonical_json_hash
    from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
    from true_rl_agent.memory.context_builder import PublicContextBuilder
    from true_rl_agent.prompt_serialization import canonical_prompt_json, canonical_prompt_json_bytes
    from true_rl_agent.tasks.stair_climbing.composition import StairClimbingCompositionProvider
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--call", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    origin = ROOT / "PLAN/2new_world/P3/results/cohort_alignment_20260909/go2_12dof_qualification"
    evidence = json.loads((origin / "result.json").read_text())
    summary = json.loads((origin / "assembly.json").read_text())["summary"]
    provider = StairClimbingCompositionProvider()
    comp = provider.resolve(values={**provider.cli_defaults(), "cell_variant_id": evidence["cell"]},
        context=TaskCompositionContext(project_root=ROOT, output_dir=output / "composition"))
    task = comp.task_plugin
    reference = comp.reference_program_artifact
    phase = evidence["phases"]["public_dev"]
    binding = EvaluationBindingV1.from_mapping(summary["evaluation_binding"])
    raw = comp.runner._public_result_from_summary(summary, job_id=summary["job_id"],
        expected_checkpoint_hash=evidence["checkpoint"], expected_eval_protocol_hash=phase["eval_protocol_hash"],
        expected_evaluation_binding_hash=phase["evaluation_binding_hash"],
        expected_sealed_seed_count=task.config.evaluation_protocol().sealed_validation_seed_count,
        evaluation_binding=binding, evaluation_subject=phase["subject"],
        subject_manifest_hash=phase["subject_manifest_hash"], program_source_checksum=phase["program_source_checksum"],
        training_binding_hash=phase["training_binding_hash"], capability_profile_hash=phase["capability_profile_hash"],
        expected_execution_device=summary["execution_device"])
    trained = json.loads(Path(evidence["training_summary"]).read_text())
    qualified = task.qualify_reference_result(public_result=raw, train_summary=trained)
    public = qualified.public_result.to_public_dict()
    best = CurrentBestEvalStats.from_reference_public_result_mapping(reference_id=reference.manifest.reference_id,
        reference_manifest_hash=reference.manifest.manifest_hash, program_source_checksum=reference.manifest.source_checksum,
        training_binding_hash=phase["training_binding_hash"], capability_profile_hash=phase["capability_profile_hash"],
        checkpoint_hash=evidence["checkpoint"], public_result=public,
        acceptance_protocol=task.config.acceptance_protocol, expected_evaluation_binding=binding)
    anchor = {"anchor_kind": "reference_program", "public_result": public, "public_result_hash": canonical_json_hash(public)}
    task.bind_genesis_evidence(anchor)
    builder = PublicContextBuilder(task_memory_adapter=comp.task_memory_adapter, task_prompt_adapter=task.prompt_adapter(),
        acceptance_protocol=task.config.acceptance_protocol, initial_current_best=best,
        candidate_capability_profile=task.task_spec.candidate_capability_profile,
        initial_reference_program_source=reference.source_text, initial_anchor_public_evidence=anchor,
        session_genesis_hash=canonical_json_hash({"purpose": "P5 diagnostic authoring only", "anchor": anchor}))
    built = builder.build_prompt([], authoring_capability_readiness=comp.authoring_capability_readiness.to_prompt_mapping(),
        candidate_verifier_prompt_rules=comp.candidate_admission_policy.authoring_prompt_rules())
    payload = built.provider_payload
    (output / "initial_payload.json").write_text(canonical_prompt_json(payload), encoding="utf-8")
    model_context = {"context_window_tokens": 1048576, "max_output_tokens": 393216,
        "tokenizer_count_source": "provider_usage_prompt_tokens",
        "specification_source": "https://api-docs.deepseek.com/quick_start/pricing"}
    policy = {"model": "deepseek-v4-flash", "max_tokens": 32768, "reasoning_mode": "adaptive",
        "reasoning_effort": "max", "empty_final_reasoning_retry": True, "model_context": model_context,
        "max_attempts": 5, "capacity_scope": "Fresh production frame; provider usage is measured, no bytes/4 token claim"}
    (output / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    print(json.dumps({"prepared_bytes": canonical_prompt_json_bytes(payload), "call": args.call}), flush=True)
    if not args.call:
        return
    live = OpenAICompatibleCandidateProvider(OpenAICompatibleProviderConfig(model=policy["model"], output_dir=output / "candidates",
        base_url="https://api.deepseek.com", env_file=Path("/data/yuyangliu/secrets/deepseek.env"), api_key_env="DEEPSEEK_API_KEY",
        max_tokens=policy["max_tokens"], timeout_s=240, reasoning_mode="adaptive", reasoning_effort="max",
        empty_final_reasoning_retry=True, model_context=model_context))
    loop = CandidateAuthoringLoop(context_builder=builder, provider=live, candidate_admission_policy=comp.candidate_admission_policy,
        round_dir=output / "authoring", max_attempts=policy["max_attempts"], component_smoke_adapter=task.component_smoke_adapter(),
        authoring_capability_readiness=comp.authoring_capability_readiness.to_prompt_mapping())
    result = loop.run_round(round_id="p5-provider-probe", records=[])
    report = {"purpose": "Provider/authoring verification only; no training or acceptance", "stop_reason": result.stop_reason,
        "candidate": result.candidate.to_public_dict() if result.candidate else None,
        "attempt_records": result.attempt_records, "prompt_paths": result.attempt_prompt_payload_paths}
    (output / "result.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({"stop_reason": result.stop_reason, "attempts": len(result.attempt_records)}), flush=True)
    if result.candidate is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
