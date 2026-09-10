"""Prepare both generation lanes from resolved task capabilities.

Authoring, immutable round inputs and phase scopes are constructed once here.
The caller supplies the frozen exploration and provider budget policy.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from true_rl_agent.agent_core.authoring_loop import CandidateAuthoringLoop
from true_rl_agent.agent_core.exploration import GenerationExplorationPlan
from true_rl_agent.agent_core.generation import FixedWidthGenerationInputs, GenerationLaneRuntime
from true_rl_agent.agent_core.generation_attempt_scopes import GenerationLaneOperationScopes
from true_rl_agent.agent_core.generation_contract import (
    GENERATION_LANE_IDS, GENERATION_MANIFEST_V2_SCHEMA_VERSION,
    GenerationLaneAssignment, GenerationManifest, GenerationSessionContract,
    generation_training_initialization_hash,
)
from true_rl_agent.agent_core.ledger import PROVIDER_AUTHORING_MODE, primary_ledger_epoch_hash
from true_rl_agent.agent_core.round import RoundInputs
from true_rl_agent.agent_core.session_genesis import VerifiedSessionGenesis
from true_rl_agent.composition import ResolvedTaskComposition
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.evaluation.phased_adapter import PhasedEvaluationAdapter
from true_rl_agent.memory import ExperimentRecord, PublicContextBuilder
from true_rl_agent.runner.base import Runner
from true_rl_agent.runner.specs import GpuProcessBinding


def build_generation_lane_inputs(
    *, composition: ResolvedTaskComposition, runner: Runner,
    contract: GenerationSessionContract, genesis: VerifiedSessionGenesis,
    session_id: str, output_dir: Path, generation_index: int,
    ledger: list[dict[str, Any]], exploration: GenerationExplorationPlan,
    parent: CurrentBestEvalStats | None, guard: Any,
    phased_adapter: PhasedEvaluationAdapter,
    gpu_bindings: Sequence[GpuProcessBinding],
    operation_scopes: Sequence[GenerationLaneOperationScopes],
    providers: Mapping[str, Any], authoring_attempts: int,
    record_authoring: Callable[..., Any],
) -> FixedWidthGenerationInputs:
    ledger_path = output_dir / "ledger.jsonl"
    ledger_epoch_hash = primary_ledger_epoch_hash(genesis.session_genesis_hash)
    records = [ExperimentRecord.from_ledger_record(record) for record in ledger]
    context_builder = PublicContextBuilder(
        task_memory_adapter=composition.task_memory_adapter,
        task_prompt_adapter=composition.task_plugin.prompt_adapter(),
        acceptance_protocol=composition.task_plugin.task_spec.acceptance_protocol,
        initial_current_best=genesis.initial_current_best,
        session_genesis_hash=genesis.session_genesis_hash,
        candidate_capability_profile=(
            composition.task_plugin.task_spec.candidate_capability_profile
        ),
        initial_reference_program_source=genesis.reference_program_source(),
        initial_anchor_public_evidence=genesis.initial_anchor_public_evidence(),
        expected_initial_policy_state_hash=(genesis.expected_initial_policy_state_hash),
        ledger_authoring_mode=PROVIDER_AUTHORING_MODE,
        expected_ledger_epoch_hash=ledger_epoch_hash,
        generation_session_contract=contract,
    )
    base_context = context_builder.build(records)
    if (
        isinstance(generation_index, bool)
        or not isinstance(generation_index, int)
        or generation_index <= 0
        or generation_index > contract.max_generations
    ):
        raise ValueError("generation index exceeds the frozen session contract")
    generation_id = f"{session_id}-generation-{generation_index:03d}"
    generation_scoped_paths = contract.max_generations > 1
    assignments = tuple(
        GenerationLaneAssignment(
            lane_id=lane_id,
            execution_device=binding.process_execution_device,
            resource_binding_hash=binding.resource_binding_hash,
            lane_intent_hash=canonical_json_hash(intent),
            gpu_process_binding=binding,
        )
        for lane_id, binding, intent in zip(
            GENERATION_LANE_IDS,
            gpu_bindings,
            exploration.lane_intents,
            strict=True,
        )
    )
    if parent is None:
        parent = genesis.initial_current_best
    guard.assert_matches_parent(parent)
    manifest = GenerationManifest(
        schema_version=GENERATION_MANIFEST_V2_SCHEMA_VERSION,
        generation_id=generation_id,
        generation_index=generation_index,
        session_genesis_hash=genesis.session_genesis_hash,
        generation_session_contract_hash=contract.contract_hash,
        ledger_epoch_hash=ledger_epoch_hash,
        parent_program_source_checksum=parent.program_source_checksum,
        parent_checkpoint_hash=parent.checkpoint_hash,
        current_best_evidence_hash=canonical_json_hash(parent.to_public_dict()),
        base_public_context_hash=canonical_json_hash(base_context),
        evaluation_binding_hash=genesis.evaluation_binding.binding_hash,
        training_initialization_hash=generation_training_initialization_hash(
            policy_initialization_kind=composition.policy_initialization.kind,
            training_binding_hash=parent.training_binding_hash,
            expected_initial_policy_state_hash=(
                genesis.expected_initial_policy_state_hash
            ),
        ),
        guard_reference_evidence_hash=guard.evidence.evidence_hash,
        guard_reference_phase_receipt_hash=canonical_json_hash(
            guard.phase_receipt_ref.to_dict()
        ),
        lanes=assignments,  # type: ignore[arg-type]
    )
    readiness = composition.authoring_capability_readiness.to_prompt_mapping()
    smoke_adapter = composition.task_plugin.component_smoke_adapter()

    def author(lane_id: str, exclusion: Any) -> Any:
        lane_index = GENERATION_LANE_IDS.index(lane_id)
        trial_id = f"{generation_id}-{lane_id}"
        authoring_loop = CandidateAuthoringLoop(
            context_builder=context_builder,
            provider=providers[lane_id],
            candidate_admission_policy=composition.candidate_admission_policy,
            round_dir=(
                output_dir / "authoring" / generation_id / lane_id
            ),
            max_attempts=authoring_attempts,
            component_smoke_adapter=smoke_adapter,
            authoring_capability_readiness=readiness,
            sibling_semantic_exclusion=exclusion,
            generation_lane_intent=exploration.lane_intents[lane_index],
            blocked_verifier_equivalence_keys=(
                exploration.blocked_verifier_equivalence_keys
            ),
        )
        if not generation_scoped_paths:
            return authoring_loop.run_round(round_id=trial_id, records=records)
        return record_authoring(
            evidence_root=(
                output_dir / "generation_authoring_timing"
            ),
            generation_id=generation_id,
            lane_id=lane_id,
            author=lambda: authoring_loop.run_round(
                round_id=trial_id,
                records=records,
            ),
        )

    def round_inputs(lane_id: str, authored: Any) -> RoundInputs:
        lane_index = GENERATION_LANE_IDS.index(lane_id)
        binding = gpu_bindings[lane_index]
        scopes = operation_scopes[lane_index]
        trial_id = f"{generation_id}-{lane_id}"
        if authored.candidate is None:
            raise ValueError("verified generation authoring result has no candidate")
        return RoundInputs(
            round_id=trial_id,
            task_plugin=composition.task_plugin,
            backend_adapter=composition.backend_adapter,
            runner=runner,
            candidate_admission_policy=composition.candidate_admission_policy,
            candidate=authored.candidate,
            policy_initialization=composition.policy_initialization,
            ledger_path=(
                output_dir
                / "non_authoritative_lane_ledgers"
                / (generation_id if generation_scoped_paths else "")
                / f"{lane_id}.jsonl"
            ),
            public_context=authored.round_public_context,
            current_best_eval_stats=parent,
            session_genesis_hash=genesis.session_genesis_hash,
            evaluation_binding=genesis.evaluation_binding,
            authoring_mode=PROVIDER_AUTHORING_MODE,
            prompt_payload_ref=authored.prompt_payload_ref,
            authoring_evidence_ref=authored.authoring_evidence_ref,
            record_candidate_runtime_failures=True,
            candidate_artifact_root=(
                output_dir
                / "candidate_artifacts"
                / (generation_id if generation_scoped_paths else "")
                / lane_id
            ),
            expected_initial_policy_state_hash=(
                genesis.expected_initial_policy_state_hash
            ),
            parent_program_source_checksum=parent.program_source_checksum,
            ledger_epoch_hash=ledger_epoch_hash,
            backend_execution_environment_manifest_hash=(
                genesis.backend_execution_environment_manifest_hash
            ),
            trial_receipt_root=(
                output_dir
                / "trial_receipts"
                / generation_id
                / lane_id
            ),
            train_attempt_scope=scopes.train,
            eval_attempt_scope=scopes.public_dev,
            gpu_process_binding=binding,
        )

    lanes = tuple(
        GenerationLaneRuntime(
            lane_id=lane_id,
            trial_id=f"{generation_id}-{lane_id}",
            lane_intent=exploration.lane_intents[index],
            lane_intent_hash=canonical_json_hash(exploration.lane_intents[index]),
            author=(lambda exclusion, lane_id=lane_id: author(lane_id, exclusion)),
            build_round_inputs=(
                lambda authored, lane_id=lane_id: round_inputs(lane_id, authored)
            ),
            **operation_scopes[index].to_generation_lane_runtime_kwargs(),
        )
        for index, lane_id in enumerate(GENERATION_LANE_IDS)
    )
    return FixedWidthGenerationInputs(
            manifest=manifest,
            session_contract=contract,
            lanes=lanes,  # type: ignore[arg-type]
            guard_reference=guard,
            phased_adapter=phased_adapter,
            receipt_root=(
                output_dir / "generation_receipts" / generation_id
            ),
            ledger_path=ledger_path,
            parent_current_best=parent,
            initial_current_best=genesis.initial_current_best,
            acceptance_protocol=composition.task_plugin.task_spec.acceptance_protocol,
    )
