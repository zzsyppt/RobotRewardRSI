"""Reference preparation and lane assembly for the shared formal session host."""

from __future__ import annotations

from true_rl_agent.agent_core.generation_attempt_scopes import (
    GenerationLaneOperationScopes, generation_job_attempt_scope,
    recover_generation_lane_operation_scopes,
)
from true_rl_agent.agent_core.generation_accounting import remaining_sealed_retry_entitlement
from true_rl_agent.agent_core.generation_contract import GENERATION_LANE_IDS
from true_rl_agent.agent_core.generation_lanes import build_generation_lane_inputs
from true_rl_agent.agent_core.generation_lifecycle import GenerationLifecycleIdentity
from true_rl_agent.agent_core.generation_provider_budget import CostGuardedProvider
from true_rl_agent.agent_core.ledger import PROVIDER_AUTHORING_MODE, load_ledger, primary_ledger_epoch_hash
from true_rl_agent.agent_core.openai_compatible_provider import (
    OpenAICompatibleCandidateProvider, OpenAICompatibleProviderConfig,
)
from true_rl_agent.agent_core.session_genesis import prepare_session_genesis, session_genesis_identity_hash
from true_rl_agent.evaluation.acceptance_replay import project_public_history
from true_rl_agent.evaluation.phases import EvaluationPhase
from true_rl_agent.runner.job_attempts import PreGenesisExecutionScope


def prepare_generation_genesis(foundation, *, require_existing=False):
    config, composition = foundation.config, foundation.composition
    snapshot = composition.genesis_identity_snapshot()
    common = dict(journal_root=config.output_dir / "job_attempts", session_id=config.session_id,
        genesis_identity_hash=session_genesis_identity_hash(snapshot),
        session_contract_hash=foundation.session_contract.contract_hash, lane_id="lane-a",
        budget_contract_hash=foundation.cost_contract.contract_hash,
        resource_binding_hash=foundation.gpu_bindings[0].resource_binding_hash)
    if config.imports_reference_genesis:
        execution = dict(source_reference_genesis_dir=config.source_reference_genesis_dir,
                         source_reference_session_id=config.source_reference_session_id)
    else:
        retries = foundation.cost_contract.retry_train_jobs
        execution = dict(gpu_process_binding=foundation.gpu_bindings[0],
            reference_train_attempt_scope=(None if config.reuses_reference_training else
                PreGenesisExecutionScope(**common,
                    invocation_segment_id="session-genesis", attempt_budget_units=1 + retries,
                    max_infrastructure_retries=retries)),
            reference_eval_phase_attempt_scopes={phase: PreGenesisExecutionScope(**common,
                invocation_segment_id="session-genesis-" + phase.value.replace("_", "-"),
                attempt_budget_units=1, max_infrastructure_retries=0) for phase in EvaluationPhase})
        if config.reuses_reference_training:
            execution.update(
                source_reference_training_genesis_dir=config.source_reference_training_genesis_dir,
                source_reference_training_session_id=config.source_reference_training_session_id,
            )
    genesis = prepare_session_genesis(output_dir=config.output_dir, session_id=config.session_id,
        task_plugin=composition.task_plugin, backend_adapter=composition.backend_adapter,
        runner=foundation.runner, policy_initialization=composition.policy_initialization,
        genesis_identity_snapshot=snapshot, baseline_manifest=composition.baseline_manifest,
        reference_program_artifact=composition.reference_program_artifact,
        bootstrap_initial_best_score=composition.bootstrap_initial_best_score,
        bootstrap_initial_policy_state_hash=composition.bootstrap_initial_policy_state_hash,
        bind_backend_execution_environment=True,
        generation_session_contract_ref=foundation.session_contract_ref,
        reference_phased_evaluation_adapter=foundation.phased_adapter,
        require_existing=require_existing, **execution)
    composition.task_plugin.bind_genesis_evidence(genesis.initial_anchor_public_evidence())
    return genesis


def generation_lifecycle_identity(foundation, *, genesis):
    return GenerationLifecycleIdentity(session_id=foundation.config.session_id,
        session_genesis_hash=genesis.session_genesis_hash,
        generation_session_contract_hash=foundation.session_contract.contract_hash,
        final_confirmation_plan_hash=foundation.final_confirmation_plan.plan_hash,
        ledger_epoch_hash=primary_ledger_epoch_hash(genesis.session_genesis_hash),
        output_root=str(foundation.config.output_dir.resolve()))


def generation_public_history(foundation, *, genesis, ledger):
    return project_public_history(ledger, initial_current_best=genesis.initial_current_best,
        expected_session_genesis_hash=genesis.session_genesis_hash,
        acceptance_protocol=foundation.composition.task_plugin.task_spec.acceptance_protocol,
        expected_initial_policy_state_hash=genesis.expected_initial_policy_state_hash,
        expected_authoring_mode=PROVIDER_AUTHORING_MODE,
        expected_ledger_epoch_hash=primary_ledger_epoch_hash(genesis.session_genesis_hash),
        phased_evaluation_adapter=foundation.phased_adapter)


def generation_provider_config(*, output_dir, env_file, policy):
    """Use the same configuration validation at preparation and execution."""
    return OpenAICompatibleProviderConfig(
        output_dir=output_dir, env_file=env_file, api_key_env=policy["api_key_env"],
        base_url=policy["base_url"], endpoint=policy["endpoint"], model=policy["model"],
        max_tokens=policy["max_output_tokens"], timeout_s=policy["timeout_s"],
        temperature=policy["temperature"], reasoning_mode=policy["reasoning_mode"],
        reasoning_effort=policy["reasoning_effort"],
        empty_final_reasoning_retry=policy["empty_final_reasoning_retry"],
        model_context=policy.get("model_context"))


def generation_provider(foundation, *, generation_id, lane_id):
    provider = OpenAICompatibleCandidateProvider(generation_provider_config(
        output_dir=foundation.config.output_dir / "provider_candidates" / generation_id / lane_id,
        env_file=foundation.config.provider_env_file, policy=foundation.provider_policy))
    return CostGuardedProvider(delegate=provider, output_dir=foundation.config.output_dir,
        generation_id=generation_id, lane_id=lane_id,
        provider_caps=foundation.cost_contract.to_dict()["provider"],
        max_generations=foundation.session_contract.max_generations,
        allow_conservative_interrupted_recovery=True)


def prepare_generation_inputs(foundation, *, genesis, guard, segment_id,
                              generation_index=1, parent=None):
    config, composition = foundation.config, foundation.composition
    ledger = load_ledger(config.output_dir / "ledger.jsonl")
    history = generation_public_history(foundation, genesis=genesis, ledger=ledger)
    exploration = composition.task_memory_adapter.build_generation_exploration_plan(
        history, generation_session_contract=foundation.session_contract)
    generation_id = f"{config.session_id}-generation-{generation_index:03d}"
    scopes = []
    for lane_id, binding in zip(GENERATION_LANE_IDS, foundation.gpu_bindings, strict=True):
        common = dict(genesis=genesis, generation_id=generation_id, lane_id=lane_id,
                      invocation_segment_id=segment_id, binding=binding)
        lane_scopes = GenerationLaneOperationScopes(
            train=generation_job_attempt_scope(foundation, **common),
            public_dev=generation_job_attempt_scope(foundation, **common),
            public_accept=generation_job_attempt_scope(foundation, **common),
            sealed_validation=generation_job_attempt_scope(foundation, **common,
                max_infrastructure_retries=remaining_sealed_retry_entitlement(foundation)))
        scopes.append(recover_generation_lane_operation_scopes(
            attempt_journal_root=config.output_dir / "job_attempts", generation_id=generation_id,
            lane_id=lane_id, current_invocation_segment_id=segment_id, current_scopes=lane_scopes))
    providers = {lane_id: generation_provider(foundation, generation_id=generation_id, lane_id=lane_id)
                 for lane_id in GENERATION_LANE_IDS}
    inputs = build_generation_lane_inputs(composition=composition, runner=foundation.runner,
        contract=foundation.session_contract, genesis=genesis, session_id=config.session_id,
        output_dir=config.output_dir, generation_index=generation_index, ledger=ledger,
        exploration=exploration, parent=parent, guard=guard, phased_adapter=foundation.phased_adapter,
        gpu_bindings=foundation.gpu_bindings, operation_scopes=scopes, providers=providers,
        authoring_attempts=foundation.provider_policy["max_attempts_per_lane"],
        record_authoring=lambda *, author, **_: author())
    return inputs, exploration


def prevalidate_generation_terminal(foundation, *, ledger,
                                   actual_cost_receipt, resource_evidence):
    count = foundation.session_contract.max_generations
    if not (len(ledger) == count
            and actual_cost_receipt.payload["cap_checks"]["all_passed"]
            and resource_evidence["generation_index"] == count
            and resource_evidence["attempt_chain"]["open_reservation_count"] == 0):
        raise ValueError("generation terminal is not closed within its frozen budget")


def generation_pause_result(foundation, *, genesis, ledger, segment_id, actual_cost_receipt, closure):
    return {"schema_version": "true_rl_agent.generation_pause_result.v1",
        "session_id": foundation.config.session_id, "task_id": foundation.composition.task_id,
        "closed_generation_count": len(ledger), "segment_id": segment_id,
        "actual_cost_receipt": {"path": str(actual_cost_receipt.path),
                                "payload_hash": actual_cost_receipt.payload_hash}, "closure": dict(closure),
        "status": "paused", "acceptance_authority": False}
