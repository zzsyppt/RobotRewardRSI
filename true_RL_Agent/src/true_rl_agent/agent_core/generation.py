"""Fixed-width generation orchestration built from existing trial and phase APIs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from true_rl_agent.agent_core.authoring_loop import CandidateAuthoringResult
from true_rl_agent.agent_core.context import sanitize_public_context
from true_rl_agent.agent_core.generation_contract import (
    GENERATION_LANE_IDS,
    GENERATION_LEDGER_RECORD_SCHEMA_VERSION,
    GENERATION_MANIFEST_V2_SCHEMA_VERSION,
    GENERATION_SESSION_CONTRACT_V3_SCHEMA_VERSION,
    GenerationManifest,
    GenerationSessionContract,
    generation_resource_plan_hash,
    generation_training_initialization_hash,
    require_generation_lane_intent,
)
from true_rl_agent.agent_core.generation_receipts import (
    GenerationReceiptJournal,
    GenerationReceiptRef,
)
from true_rl_agent.agent_core.generation_record import (
    GENERATION_FORMAL_PROJECTION_SCHEMA_VERSION,
    GENERATION_PHASE_PROJECTION_SCHEMA_VERSION,
    GENERATION_TRIAL_RECORD_SCHEMA_VERSION,
    commit_generation_record,
    validate_generation_ledger_record,
)
from true_rl_agent.agent_core.ledger import PROVIDER_AUTHORING_MODE
from true_rl_agent.agent_core.round import (
    CandidateTrainExecution,
    PreparedCandidateTrial,
    RoundInputs,
    TrialExecution,
    finalize_prepared_candidate_trial,
    prepare_candidate_trial,
    project_candidate_full_eval_result,
    run_prepared_candidate_train,
)
from true_rl_agent.agent_core.trial_receipts import TrialReceiptRef
from true_rl_agent.candidates.sibling_exclusion import (
    SingleComponentDeltaComparisonUnavailable,
    SingleComponentSemanticExclusion,
)
from true_rl_agent.evaluation.acceptance_protocol import AcceptanceProtocol
from true_rl_agent.evaluation.acceptance_replay import replay_acceptance_ledger
from true_rl_agent.evaluation.binding import (
    canonical_json_hash,
    require_sha256_checksum,
    require_sha256_digest,
)
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.evaluation.phases import EvaluationPhase, EvaluationPhaseResult
from true_rl_agent.evaluation.public_projection import project_public_runner_error
from true_rl_agent.prompt_artifacts import (
    verify_authoring_anchor_prompt,
    verify_authoring_evidence,
    verify_prompt_payload,
)
from true_rl_agent.prompting import assert_public_only_prompt_payload
from true_rl_agent.runner.base import (
    CandidateRuntimeError,
    RunnerExecutionError,
    TrainJobResult,
)
from true_rl_agent.runner.eval_binding import bind_eval_spec_to_checkpoint
from true_rl_agent.runner.job_attempts import JobAttemptScope


class _ScreenEvidence(Protocol):
    @property
    def evidence_hash(self) -> str: ...

    def to_selector_dict(self) -> dict[str, Any]: ...


class _SelectionResult(Protocol):
    selected_trial_id: str | None
    reason: str

    def to_public_dict(self) -> dict[str, Any]: ...


class _GuardReference(Protocol):
    evidence: _ScreenEvidence
    phase_receipt_ref: TrialReceiptRef

    def assert_matches_parent(self, parent: CurrentBestEvalStats) -> None: ...


class _PhasedEvaluationAdapter(Protocol):
    def selection_contract(self) -> Mapping[str, Any]: ...

    def cost_contract(self) -> Mapping[str, Any]: ...

    def project_public_dev_screen(
        self,
        *,
        trial_id: str,
        phase_result: EvaluationPhaseResult,
        screen_role: str,
        train_result: TrainJobResult,
        train_receipt_checksum: str,
        mechanism_operation_delta: Mapping[str, Any] | None,
    ) -> _ScreenEvidence: ...

    def project_public_phase_summary(
        self, *, phase_result: EvaluationPhaseResult
    ) -> Mapping[str, Any]: ...

    def select(
        self,
        screens: Sequence[_ScreenEvidence],
        *,
        guard_reference: _ScreenEvidence,
    ) -> _SelectionResult: ...

    def assemble_full_summary(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class GenerationLaneRuntime:
    """Trusted lane-local factories; provider and artifact state stay isolated."""

    lane_id: str
    trial_id: str
    author: Callable[
        [SingleComponentSemanticExclusion | None], CandidateAuthoringResult
    ]
    build_round_inputs: Callable[[CandidateAuthoringResult], RoundInputs]
    lane_intent: Mapping[str, Any]
    lane_intent_hash: str
    attempt_scope: JobAttemptScope | None = None
    sealed_attempt_scope: JobAttemptScope | None = None
    public_dev_attempt_scope: JobAttemptScope | None = None
    public_accept_attempt_scope: JobAttemptScope | None = None

    def __post_init__(self) -> None:
        if self.lane_id not in GENERATION_LANE_IDS:
            raise ValueError("generation lane runtime lane_id is invalid")
        if not isinstance(self.trial_id, str) or not self.trial_id:
            raise ValueError("generation lane runtime trial_id is required")
        if not callable(self.author) or not callable(self.build_round_inputs):
            raise TypeError("generation lane runtime factories must be callable")
        intent = require_generation_lane_intent(self.lane_intent)
        assert_public_only_prompt_payload(intent)
        if canonical_json_hash(intent) != require_sha256_digest(
            self.lane_intent_hash,
            "generation lane runtime intent hash",
        ):
            raise ValueError("generation lane runtime intent hash mismatch")
        for field_name, scope in (
            ("attempt_scope", self.attempt_scope),
            ("sealed_attempt_scope", self.sealed_attempt_scope),
            ("public_dev_attempt_scope", self.public_dev_attempt_scope),
            ("public_accept_attempt_scope", self.public_accept_attempt_scope),
        ):
            if scope is not None and not isinstance(scope, JobAttemptScope):
                raise TypeError(f"generation lane {field_name} is invalid")

    def scope_for_train(self) -> JobAttemptScope | None:
        return self.attempt_scope

    def scope_for_phase(
        self,
        phase: EvaluationPhase,
    ) -> JobAttemptScope | None:
        if phase is EvaluationPhase.PUBLIC_DEV:
            specific = self.public_dev_attempt_scope
        elif phase is EvaluationPhase.PUBLIC_ACCEPT:
            specific = self.public_accept_attempt_scope
        elif phase is EvaluationPhase.SEALED_VALIDATION:
            specific = self.sealed_attempt_scope
        else:
            raise ValueError("generation evaluation phase is invalid")
        return specific if specific is not None else self.attempt_scope

    def operation_attempt_scopes(
        self,
    ) -> tuple[
        JobAttemptScope | None,
        JobAttemptScope | None,
        JobAttemptScope | None,
        JobAttemptScope | None,
    ]:
        return (
            self.scope_for_train(),
            self.scope_for_phase(EvaluationPhase.PUBLIC_DEV),
            self.scope_for_phase(EvaluationPhase.PUBLIC_ACCEPT),
            self.scope_for_phase(EvaluationPhase.SEALED_VALIDATION),
        )


@dataclass(frozen=True)
class FixedWidthGenerationInputs:
    manifest: GenerationManifest
    session_contract: GenerationSessionContract
    lanes: tuple[GenerationLaneRuntime, GenerationLaneRuntime]
    guard_reference: _GuardReference
    phased_adapter: _PhasedEvaluationAdapter
    receipt_root: Path
    ledger_path: Path
    parent_current_best: CurrentBestEvalStats
    initial_current_best: CurrentBestEvalStats
    acceptance_protocol: AcceptanceProtocol

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, GenerationManifest):
            raise TypeError("generation inputs manifest is invalid")
        if not isinstance(self.session_contract, GenerationSessionContract):
            raise TypeError("generation inputs session contract is invalid")
        if tuple(lane.lane_id for lane in self.lanes) != GENERATION_LANE_IDS:
            raise ValueError("generation lanes must be ordered lane-a/lane-b")
        if len({lane.trial_id for lane in self.lanes}) != 2:
            raise ValueError("generation trial IDs must be unique")
        if not isinstance(self.receipt_root, Path) or not isinstance(
            self.ledger_path, Path
        ):
            raise TypeError("generation receipt and ledger roots must be Paths")
        if not isinstance(self.parent_current_best, CurrentBestEvalStats):
            raise TypeError("generation parent current-best is invalid")
        if not isinstance(self.initial_current_best, CurrentBestEvalStats):
            raise TypeError("generation session initial current-best is invalid")
        if not isinstance(self.acceptance_protocol, AcceptanceProtocol):
            raise TypeError("generation acceptance protocol is invalid")


@dataclass(frozen=True)
class GenerationExecutionResult:
    record: Mapping[str, Any]
    execution_receipt_ref: GenerationReceiptRef
    resulting_current_best: CurrentBestEvalStats
    committed: bool
    reused_execution: bool


@dataclass
class _LaneState:
    runtime: GenerationLaneRuntime
    authoring: CandidateAuthoringResult | None = None
    sibling_exclusion_hash: str | None = None
    operation_delta: Mapping[str, Any] | None = None
    prepared: PreparedCandidateTrial | None = None
    train_execution: CandidateTrainExecution | None = None
    trial_execution: TrialExecution | None = None
    phase_results: dict[EvaluationPhase, EvaluationPhaseResult] | None = None
    phase_refs: dict[EvaluationPhase, TrialReceiptRef] | None = None
    phase_job_spec_hashes: dict[EvaluationPhase, str] | None = None
    screen: _ScreenEvidence | None = None
    status: str = "not_started"
    status_reason: str = "lane_not_started"

    def __post_init__(self) -> None:
        self.phase_results = {}
        self.phase_refs = {}
        self.phase_job_spec_hashes = {}


def execute_fixed_width_generation(
    inputs: FixedWidthGenerationInputs,
) -> GenerationExecutionResult:
    """Execute A then B authoring, two execution lanes, and one atomic commit."""

    if not isinstance(inputs, FixedWidthGenerationInputs):
        raise TypeError("fixed-width generation inputs are invalid")
    _validate_manifest_inputs(inputs)
    journal = GenerationReceiptJournal(inputs.receipt_root, manifest=inputs.manifest)
    manifest_ref = journal.bind_manifest()
    loaded_execution = journal.load_execution()
    acceptance_protocol = inputs.acceptance_protocol
    if loaded_execution is not None:
        record, execution_ref = loaded_execution
        resulting = validate_generation_ledger_record(
            record,
            acceptance_protocol=acceptance_protocol,
            expected_parent=inputs.parent_current_best,
        )
        committed = commit_generation_record(
            ledger_path=inputs.ledger_path,
            record=record,
            initial_current_best=inputs.initial_current_best,
            acceptance_protocol=acceptance_protocol,
        )
        return GenerationExecutionResult(
            record=record,
            execution_receipt_ref=execution_ref,
            resulting_current_best=resulting,
            committed=committed,
            reused_execution=True,
        )

    states = [_LaneState(runtime=lane) for lane in inputs.lanes]
    exclusion: SingleComponentSemanticExclusion | None = None
    for index, state in enumerate(states):
        if index == 1 and states[0].status != "authoring_verified":
            break
        if index == 1:
            assert exclusion is not None
            state.sibling_exclusion_hash = exclusion.exclusion_hash
        state.authoring = state.runtime.author(exclusion)
        _validate_authoring_result(inputs, state, exclusion=exclusion)
        if state.authoring.stop_reason != "candidate_verified":
            state.status = "candidate_failed"
            state.status_reason = f"authoring_{state.authoring.stop_reason}"
            break
        state.status = "authoring_verified"
        state.status_reason = "candidate_verified"
        anchor_prompt = _verified_authoring_anchor_prompt(
            state.authoring,
            expected_round_id=state.runtime.trial_id,
        )
        if index == 0:
            try:
                exclusion = SingleComponentSemanticExclusion.from_verified_candidate(
                    candidate=_require_authored_candidate(state.authoring),
                    prompt_payload=anchor_prompt,
                )
                state.operation_delta = dict(exclusion.operation_delta)
            except SingleComponentDeltaComparisonUnavailable:
                state.status = "candidate_failed"
                state.status_reason = "sibling_exclusion_unavailable"
                break
        else:
            assert exclusion is not None
            try:
                equivalence = exclusion.compare_candidate(
                    candidate=_require_authored_candidate(state.authoring),
                    candidate_prompt_payload=anchor_prompt,
                )
            except SingleComponentDeltaComparisonUnavailable as exc:
                raise ValueError(
                    "verified lane B has unavailable sibling comparison"
                ) from exc
            if equivalence is not None:
                raise ValueError("verified lane B is not semantically distinct")

    if any(state.status != "authoring_verified" for state in states):
        if states[0].status == "authoring_verified":
            states[0].status = "verified_not_run"
            states[0].status_reason = "paired_authoring_incomplete"
        return _seal_and_commit(
            inputs,
            journal=journal,
            manifest_ref=manifest_ref,
            states=states,
            selection=None,
            selection_ref=None,
            acceptance_protocol=acceptance_protocol,
        )

    for state in states:
        assert state.authoring is not None
        round_inputs = state.runtime.build_round_inputs(state.authoring)
        _validate_round_inputs(inputs, state, round_inputs)
        state.prepared = prepare_candidate_trial(round_inputs)

    _validate_prepared_pair(inputs, states)

    unready = [state for state in states if not _prepared_is_runnable(state)]
    if unready:
        for state in states:
            if state in unready:
                state.status = "candidate_failed"
                state.status_reason = "candidate_preparation_failed"
                if state.prepared is not None:
                    state.trial_execution = finalize_prepared_candidate_trial(
                        state.prepared
                    )
            else:
                state.status = "verified_not_run"
                state.status_reason = "paired_authoring_incomplete"
        return _seal_and_commit(
            inputs,
            journal=journal,
            manifest_ref=manifest_ref,
            states=states,
            selection=None,
            selection_ref=None,
            acceptance_protocol=acceptance_protocol,
        )

    _parallel_apply(states, _run_train_lane)
    active = [state for state in states if state.status != "candidate_failed"]
    _parallel_apply(
        active, lambda state: _run_phase(inputs, state, EvaluationPhase.PUBLIC_DEV)
    )
    for state in active:
        if state.status != "candidate_failed":
            assert state.phase_results is not None
            phase_result = state.phase_results[EvaluationPhase.PUBLIC_DEV]
            assert state.prepared is not None and state.train_execution is not None
            train_execution = state.train_execution
            train_receipt_checksum = require_sha256_checksum(
                train_execution.train_receipt_ref.checksum,
                "generation screen train receipt checksum",
            )
            state.screen = inputs.phased_adapter.project_public_dev_screen(
                trial_id=state.runtime.trial_id,
                phase_result=phase_result,
                screen_role="candidate",
                train_result=train_execution.train_result,
                train_receipt_checksum=train_receipt_checksum,
                mechanism_operation_delta=(
                    state.prepared.authoring_mechanism_operation_delta
                ),
            )
            state.status = "screened_only"
            state.status_reason = "public_dev_complete"

    selection: _SelectionResult | None = None
    selection_ref: GenerationReceiptRef | None = None
    screens = sorted(
        (state.screen for state in states if state.screen is not None),
        key=lambda item: str(item.to_selector_dict().get("trial_id")),
    )
    if len(screens) == 2:
        selection = inputs.phased_adapter.select(
            screens,
            guard_reference=inputs.guard_reference.evidence,
        )
        selection_ref = journal.publish_selection(
            selection=selection.to_public_dict(),
            screens=[screen.to_selector_dict() for screen in screens],
            guard_reference=inputs.guard_reference.evidence.to_selector_dict(),
            guard_reference_phase_receipt_checksum=(
                inputs.guard_reference.phase_receipt_ref.checksum
            ),
        )

    selected_state = _selected_state(states, selection)
    # Public-accept confirms the public-dev winner; it is not a second selector.
    # A no-selection generation therefore pays no confirmation cost.
    if selected_state is not None and selected_state.status != "candidate_failed":
        _run_phase(inputs, selected_state, EvaluationPhase.PUBLIC_ACCEPT)
    if selected_state is not None and selected_state.status != "candidate_failed":
        _run_phase(inputs, selected_state, EvaluationPhase.SEALED_VALIDATION)
        if selected_state.status != "candidate_failed":
            selected_state.trial_execution = _finalize_formal_trial(
                inputs,
                selected_state,
                selection=selection,
                screens=screens,
            )
    if selection is not None and selected_state is None:
        for state in states:
            if state.status != "candidate_failed":
                state.status = "qualified_not_selected"
                state.status_reason = selection.reason
    elif selection is not None:
        for state in states:
            if state is not selected_state and state.status != "candidate_failed":
                state.status = "qualified_not_selected"
                state.status_reason = "sibling_not_selected"
    if selected_state is not None and selected_state.trial_execution is not None:
        if selected_state.trial_execution.decision.accepted:
            selected_state.status = "accepted"
            selected_state.status_reason = "formal_acceptance_passed"
        elif selected_state.trial_execution.public_result is not None:
            selected_state.status = "formal_rejected"
            selected_state.status_reason = (
                selected_state.trial_execution.decision.reason
            )

    return _seal_and_commit(
        inputs,
        journal=journal,
        manifest_ref=manifest_ref,
        states=states,
        selection=selection,
        selection_ref=selection_ref,
        acceptance_protocol=acceptance_protocol,
    )


def _validate_manifest_inputs(inputs: FixedWidthGenerationInputs) -> None:
    manifest = inputs.manifest
    session_contract = inputs.session_contract
    if (
        manifest.schema_version == GENERATION_MANIFEST_V2_SCHEMA_VERSION
        and session_contract.schema_version
        != GENERATION_SESSION_CONTRACT_V3_SCHEMA_VERSION
    ):
        raise ValueError(
            "bound generation manifest requires the versioned session cost contract"
        )
    if (
        manifest.generation_session_contract_hash != session_contract.contract_hash
        or manifest.generation_index > session_contract.max_generations
    ):
        raise ValueError("generation manifest differs from session contract")
    bound_hashes = session_contract.bound_contract_hashes
    if (
        bound_hashes["selector"]
        != canonical_json_hash(dict(inputs.phased_adapter.selection_contract()))
        or bound_hashes["evaluation_cost"]
        != canonical_json_hash(dict(inputs.phased_adapter.cost_contract()))
        or bound_hashes["resource_plan"]
        != generation_resource_plan_hash(manifest.lanes)
    ):
        raise ValueError("generation task/resource contract binding mismatch")
    parent = inputs.parent_current_best
    if (
        canonical_json_hash(parent.to_public_dict())
        != manifest.current_best_evidence_hash
        or parent.program_source_checksum != manifest.parent_program_source_checksum
        or parent.checkpoint_hash != manifest.parent_checkpoint_hash
        or parent.evaluation_binding_hash != manifest.evaluation_binding_hash
        or parent.eval_protocol_hash != bound_hashes["evaluation_protocol"]
        or parent.acceptance_protocol_hash != inputs.acceptance_protocol.protocol_hash
    ):
        raise ValueError(
            "generation manifest parent is not the configured current-best"
        )
    inputs.guard_reference.assert_matches_parent(parent)
    if (
        manifest.guard_reference_evidence_hash
        != inputs.guard_reference.evidence.evidence_hash
    ):
        raise ValueError("generation manifest guard evidence mismatch")
    if manifest.guard_reference_phase_receipt_hash != canonical_json_hash(
        inputs.guard_reference.phase_receipt_ref.to_dict()
    ):
        raise ValueError("generation manifest guard receipt binding mismatch")
    assignments = {item.lane_id: item for item in manifest.lanes}
    for lane in inputs.lanes:
        assignment = assignments[lane.lane_id]
        if lane.lane_intent_hash != assignment.lane_intent_hash:
            raise ValueError("generation lane intent differs from manifest")
        for scope in lane.operation_attempt_scopes():
            if scope is None:
                continue
            if (
                scope.session_genesis_hash != manifest.session_genesis_hash
                or scope.session_contract_hash
                != manifest.generation_session_contract_hash
                or scope.generation_id != manifest.generation_id
                or scope.lane_id != lane.lane_id
                or scope.resource_binding_hash != assignment.resource_binding_hash
                or (
                    "session_cost" in bound_hashes
                    and scope.budget_contract_hash != bound_hashes["session_cost"]
                )
            ):
                raise ValueError("generation lane attempt scope differs from manifest")
        if assignment.gpu_process_binding is not None and any(
            scope is None for scope in lane.operation_attempt_scopes()
        ):
            raise ValueError(
                "bound generation lane requires every operation attempt scope"
            )


def _validate_authoring_result(
    inputs: FixedWidthGenerationInputs,
    state: _LaneState,
    *,
    exclusion: SingleComponentSemanticExclusion | None,
) -> None:
    result = state.authoring
    if not isinstance(result, CandidateAuthoringResult):
        raise TypeError("generation author callback returned an invalid result")
    if (
        result.parent_program_source_checksum
        != inputs.manifest.parent_program_source_checksum
    ):
        raise ValueError("generation authoring parent differs from manifest")
    context_hash = canonical_json_hash(
        sanitize_public_context(result.round_public_context)
    )
    if context_hash != inputs.manifest.base_public_context_hash:
        raise ValueError("generation authoring base context differs from manifest")
    verifier_rules = _provider_verifier_rules(
        verify_prompt_payload(result.prompt_payload_ref)
    )
    if verifier_rules.get("generation_lane_intent") != dict(state.runtime.lane_intent):
        raise ValueError("generation authoring prompt lane intent mismatch")
    if state.runtime.lane_id == GENERATION_LANE_IDS[0]:
        if exclusion is not None:
            raise ValueError("generation lane A received sibling exclusion")
        if "single_component_semantic_exclusion" in verifier_rules:
            raise ValueError("generation lane A prompt contains sibling exclusion")
    elif exclusion is None:
        raise ValueError("generation lane B did not receive sibling exclusion")
    elif verifier_rules.get("single_component_semantic_exclusion") != dict(
        exclusion.public_projection
    ):
        raise ValueError("generation lane B prompt exclusion mismatch")


def _verified_authoring_anchor_prompt(
    result: CandidateAuthoringResult,
    *,
    expected_round_id: str,
) -> dict[str, Any]:
    evidence = verify_authoring_evidence(
        result.authoring_evidence_ref,
        expected_round_id=expected_round_id,
        expected_selected_prompt=result.prompt_payload_ref,
    )
    return verify_authoring_anchor_prompt(evidence)


def _provider_verifier_rules(prompt_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    required = prompt_payload.get("required_candidate_package")
    component_contract = (
        required.get("component_contract") if isinstance(required, Mapping) else None
    )
    rules = (
        component_contract.get("static_verifier_rules")
        if isinstance(component_contract, Mapping)
        else None
    )
    if not isinstance(rules, Mapping):
        raise ValueError("generation authoring prompt has no verifier rules")
    return rules


def _require_authored_candidate(result: CandidateAuthoringResult):
    if result.stop_reason != "candidate_verified" or result.candidate is None:
        raise ValueError("verified authoring result has no candidate")
    return result.candidate


def _validate_round_inputs(
    inputs: FixedWidthGenerationInputs,
    state: _LaneState,
    round_inputs: RoundInputs,
) -> None:
    manifest = inputs.manifest
    assignment = next(
        item for item in manifest.lanes if item.lane_id == state.runtime.lane_id
    )
    if not isinstance(round_inputs, RoundInputs):
        raise TypeError("generation lane produced invalid RoundInputs")
    if (
        round_inputs.round_id != state.runtime.trial_id
        or round_inputs.task_plugin.task_spec.task_id != inputs.session_contract.task_id
        or round_inputs.backend_adapter.backend_id != inputs.session_contract.backend_id
        or round_inputs.session_genesis_hash != manifest.session_genesis_hash
        or round_inputs.ledger_epoch_hash != manifest.ledger_epoch_hash
        or round_inputs.parent_program_source_checksum
        != manifest.parent_program_source_checksum
        or round_inputs.current_best_eval_stats.to_public_dict()
        != inputs.parent_current_best.to_public_dict()
        or round_inputs.evaluation_binding.binding_hash
        != manifest.evaluation_binding_hash
        or canonical_json_hash(sanitize_public_context(round_inputs.public_context))
        != manifest.base_public_context_hash
        or round_inputs.gpu_process_binding != assignment.gpu_process_binding
    ):
        raise ValueError("generation lane RoundInputs differ from manifest")
    if round_inputs.candidate != _require_authored_candidate(state.authoring):
        raise ValueError("generation lane RoundInputs candidate mismatch")
    if (
        round_inputs.task_plugin.task_spec.acceptance_protocol
        != inputs.acceptance_protocol
    ):
        raise ValueError("generation lane acceptance protocol differs from session")
    train_attempt_scope = state.runtime.scope_for_train()
    public_dev_attempt_scope = state.runtime.scope_for_phase(EvaluationPhase.PUBLIC_DEV)
    if round_inputs.train_attempt_scope != train_attempt_scope:
        raise ValueError("generation train attempt scope mismatch")
    if round_inputs.eval_attempt_scope != public_dev_attempt_scope:
        raise ValueError("generation eval attempt scope mismatch")
    if assignment.execution_device.startswith("cuda:") and any(
        scope is None for scope in state.runtime.operation_attempt_scopes()
    ):
        raise ValueError("generation CUDA lane requires every operation attempt scope")


def _validate_prepared_pair(
    inputs: FixedWidthGenerationInputs,
    states: Sequence[_LaneState],
) -> None:
    """Bind both materialized trials to one frozen comparison domain."""

    training_hashes: set[str] = set()
    for state in states:
        prepared = state.prepared
        if prepared is None:
            raise ValueError("generation trial preparation is missing")
        if prepared.acceptance_protocol != inputs.acceptance_protocol:
            raise ValueError("generation lane acceptance protocol mismatch")
        prepared_delta = prepared.authoring_mechanism_operation_delta
        if state.operation_delta is not None and (
            prepared_delta is None
            or dict(state.operation_delta) != dict(prepared_delta)
        ):
            raise ValueError(
                "generation prepared mechanism delta differs from authoring evidence"
            )
        if prepared_delta is not None:
            state.operation_delta = dict(prepared_delta)
        if prepared.jobs is None:
            continue
        assignment = next(
            item
            for item in inputs.manifest.lanes
            if item.lane_id == state.runtime.lane_id
        )
        train_spec = prepared.jobs.train_spec
        eval_spec = prepared.jobs.eval_spec
        train_attempt_scope = state.runtime.scope_for_train()
        public_dev_attempt_scope = state.runtime.scope_for_phase(
            EvaluationPhase.PUBLIC_DEV
        )
        if (
            train_spec.execution_device != assignment.execution_device
            or eval_spec.execution_device != assignment.execution_device
            or train_spec.gpu_process_binding != assignment.gpu_process_binding
            or eval_spec.gpu_process_binding != assignment.gpu_process_binding
            or train_spec.attempt_scope != train_attempt_scope
            or eval_spec.attempt_scope != public_dev_attempt_scope
            or eval_spec.evaluation_binding.binding_hash
            != inputs.manifest.evaluation_binding_hash
        ):
            raise ValueError("generation materialized lane resource binding mismatch")
        initialization_hash = generation_training_initialization_hash(
            policy_initialization_kind=train_spec.policy_initialization.kind,
            training_binding_hash=str(train_spec.training_binding.binding_hash),
            expected_initial_policy_state_hash=(
                train_spec.expected_initial_policy_state_hash
            ),
        )
        if initialization_hash != inputs.manifest.training_initialization_hash:
            raise ValueError("generation training initialization differs from manifest")
        training_hashes.add(str(train_spec.training_binding.binding_hash))
    if len(training_hashes) > 1:
        raise ValueError("generation sibling training bindings differ")


def _prepared_is_runnable(state: _LaneState) -> bool:
    prepared = state.prepared
    if prepared is None:
        return False
    if prepared.reused_execution is not None:
        return True
    return prepared.ready_for_training


def _parallel_apply(
    states: Sequence[_LaneState], operation: Callable[[_LaneState], None]
) -> None:
    if not states:
        return
    with ThreadPoolExecutor(
        max_workers=len(states), thread_name_prefix="generation"
    ) as pool:
        futures = {
            state.runtime.lane_id: pool.submit(operation, state) for state in states
        }
        for lane_id in GENERATION_LANE_IDS:
            future = futures.get(lane_id)
            if future is not None:
                future.result()


def _run_train_lane(state: _LaneState) -> None:
    assert state.prepared is not None
    prepared = state.prepared
    if prepared.reused_execution is not None:
        reused = prepared.reused_execution
        state.trial_execution = reused
        if reused.train_result is None:
            state.status = "candidate_failed"
            state.status_reason = reused.decision.reason
            return
        assert prepared.jobs is not None
        loaded_train = prepared.trial_journal.load_train(prepared.jobs.train_spec)
        if loaded_train is None:
            raise ValueError("reused trial execution has no train receipt")
        train_result, train_ref = loaded_train
        state.train_execution = CandidateTrainExecution(
            prepared=prepared,
            train_result=train_result,
            train_receipt_ref=train_ref,
            eval_plan=bind_eval_spec_to_checkpoint(
                prepared.jobs.eval_spec, train_result.checkpoint
            ),
        )
        state.status = "trained"
        state.status_reason = "train_receipt_reused"
        return
    try:
        state.train_execution = run_prepared_candidate_train(prepared)
    except CandidateRuntimeError as exc:
        state.trial_execution = finalize_prepared_candidate_trial(
            prepared,
            failure_reason="candidate_runtime_failed",
            runner_error=project_public_runner_error(exc),
        )
        state.status = "candidate_failed"
        state.status_reason = "candidate_runtime_failed"
        return
    except RunnerExecutionError as exc:
        if not exc.retryable_infrastructure_failure:
            raise
        state.status = "candidate_failed"
        state.status_reason = "train_incomplete"
        return
    state.status = "trained"
    state.status_reason = "train_complete"


def _run_phase(
    inputs: FixedWidthGenerationInputs,
    state: _LaneState,
    phase: EvaluationPhase,
) -> None:
    if state.status == "candidate_failed":
        return
    assert state.prepared is not None and state.train_execution is not None
    prepared = state.prepared
    attempt_scope = state.runtime.scope_for_phase(phase)
    phase_spec = prepared.inputs.backend_adapter.materialize_eval_phase(
        eval_plan=state.train_execution.eval_plan,
        phase=phase,
        attempt_scope=attempt_scope,
    )
    assignment = next(
        item for item in inputs.manifest.lanes if item.lane_id == state.runtime.lane_id
    )
    if (
        phase_spec.execution_device != assignment.execution_device
        or phase_spec.gpu_process_binding != assignment.gpu_process_binding
        or phase_spec.attempt_scope != attempt_scope
    ):
        raise ValueError("generation phase resource assignment mismatch")
    loaded = prepared.trial_journal.load_eval_phase(phase_spec)
    try:
        if loaded is None:
            if prepared.reused_execution is not None:
                # Rebuild the completed prefix from the trial journal. A sealed
                # failed trial is terminal, even when the generation was not sealed.
                if prepared.reused_execution.public_result is not None:
                    raise ValueError("completed trial execution has no phase receipt")
                state.status = "candidate_failed"
                state.status_reason = (
                    f"{prepared.reused_execution.decision.reason}_{phase.value}"
                )
                return
            result = prepared.inputs.runner.run_eval_phase(phase_spec)
            receipt_ref = prepared.trial_journal.publish_eval_phase(phase_spec, result)
        else:
            result, receipt_ref = loaded
    except CandidateRuntimeError as exc:
        state.trial_execution = finalize_prepared_candidate_trial(
            prepared,
            train_execution=state.train_execution,
            failure_reason="candidate_runtime_failed",
            runner_error=project_public_runner_error(exc),
        )
        state.status = "candidate_failed"
        state.status_reason = f"candidate_runtime_failed_{phase.value}"
        return
    except RunnerExecutionError as exc:
        if not exc.retryable_infrastructure_failure:
            raise
        state.trial_execution = finalize_prepared_candidate_trial(
            prepared,
            train_execution=state.train_execution,
            failure_reason="eval_incomplete",
            runner_error=project_public_runner_error(exc),
        )
        state.status = "candidate_failed"
        state.status_reason = f"eval_incomplete_{phase.value}"
        return
    assert (
        state.phase_results is not None
        and state.phase_refs is not None
        and state.phase_job_spec_hashes is not None
    )
    state.phase_results[phase] = result
    state.phase_refs[phase] = receipt_ref
    state.phase_job_spec_hashes[phase] = canonical_json_hash(
        phase_spec.to_public_dict()
    )


def _selected_state(
    states: Sequence[_LaneState], selection: _SelectionResult | None
) -> _LaneState | None:
    if selection is None or selection.selected_trial_id is None:
        return None
    matches = [
        state
        for state in states
        if state.runtime.trial_id == selection.selected_trial_id
    ]
    if len(matches) != 1:
        raise ValueError("generation selection names an unknown or duplicate trial")
    return matches[0]


def _finalize_formal_trial(
    inputs: FixedWidthGenerationInputs,
    state: _LaneState,
    *,
    selection: _SelectionResult,
    screens: Sequence[_ScreenEvidence],
) -> TrialExecution:
    assert state.prepared is not None and state.train_execution is not None
    if (
        state.trial_execution is not None
        and state.trial_execution.public_result is not None
    ):
        return state.trial_execution
    assert state.phase_results is not None and state.phase_refs is not None
    assembly = inputs.phased_adapter.assemble_full_summary(
        eval_plan=state.train_execution.eval_plan,
        trial_id=state.runtime.trial_id,
        selection_result=selection,
        selection_screens=screens,
        guard_reference=inputs.guard_reference.evidence,
        phase_results=[state.phase_results[phase] for phase in EvaluationPhase],
        phase_receipt_refs={
            phase: state.phase_refs[phase].to_dict() for phase in EvaluationPhase
        },
    )
    raw_result = state.prepared.inputs.runner.project_assembled_eval_summary(
        state.train_execution.eval_plan,
        assembly,
    )
    public_result = project_candidate_full_eval_result(
        state.train_execution, raw_result
    )
    return finalize_prepared_candidate_trial(
        state.prepared,
        train_execution=state.train_execution,
        public_result=public_result,
    )


def _seal_and_commit(
    inputs: FixedWidthGenerationInputs,
    *,
    journal: GenerationReceiptJournal,
    manifest_ref: GenerationReceiptRef,
    states: Sequence[_LaneState],
    selection: _SelectionResult | None,
    selection_ref: GenerationReceiptRef | None,
    acceptance_protocol: AcceptanceProtocol,
) -> GenerationExecutionResult:
    record, resulting = _build_generation_record(
        inputs,
        manifest_ref=manifest_ref,
        states=states,
        selection=selection,
        selection_ref=selection_ref,
        acceptance_protocol=acceptance_protocol,
    )
    validate_generation_ledger_record(
        record,
        acceptance_protocol=acceptance_protocol,
        expected_parent=inputs.parent_current_best,
    )
    execution_ref = journal.publish_execution(record)
    loaded = journal.load_execution()
    if loaded is None or loaded[0] != record or loaded[1] != execution_ref:
        raise ValueError("generation execution receipt changed after publication")
    committed = commit_generation_record(
        ledger_path=inputs.ledger_path,
        record=record,
        initial_current_best=inputs.initial_current_best,
        acceptance_protocol=acceptance_protocol,
    )
    return GenerationExecutionResult(
        record=record,
        execution_receipt_ref=execution_ref,
        resulting_current_best=resulting,
        committed=committed,
        reused_execution=False,
    )


def _build_generation_record(
    inputs: FixedWidthGenerationInputs,
    *,
    manifest_ref: GenerationReceiptRef,
    states: Sequence[_LaneState],
    selection: _SelectionResult | None,
    selection_ref: GenerationReceiptRef | None,
    acceptance_protocol: AcceptanceProtocol,
) -> tuple[dict[str, Any], CurrentBestEvalStats]:
    manifest = inputs.manifest
    private_artifacts = [
        _private_generation_ref("generation_manifest_receipt", manifest_ref),
        _private_trial_ref(
            "generation_guard_reference_phase_receipt",
            inputs.guard_reference.phase_receipt_ref,
        ),
    ]
    if selection_ref is not None:
        private_artifacts.append(
            _private_generation_ref("generation_selection_receipt", selection_ref)
        )
    trials = []
    formal = None
    formal_round = None
    selected_id = selection.selected_trial_id if selection is not None else None
    for state in states:
        trial, artifacts = _project_trial(inputs, state)
        private_artifacts.extend(artifacts)
        if state.trial_execution is not None:
            execution = state.trial_execution
            execution_hash = canonical_json_hash(execution.record)
            trial["execution_record_hash"] = execution_hash
            trial["execution_receipt_checksum"] = execution.receipt_ref.checksum
            private_artifacts.append(
                _private_trial_ref(
                    f"{state.runtime.lane_id}_trial_execution_receipt",
                    execution.receipt_ref,
                )
            )
            if (
                state.runtime.trial_id == selected_id
                and execution.public_result is not None
            ):
                formal_round = execution.record
                trial["formal_record_hash"] = execution_hash
                formal = {
                    "schema_version": GENERATION_FORMAL_PROJECTION_SCHEMA_VERSION,
                    "selected_lane_id": state.runtime.lane_id,
                    "selected_trial_id": state.runtime.trial_id,
                    "trial_execution_receipt_checksum": execution.receipt_ref.checksum,
                    "round_record_hash": execution_hash,
                    "sealed_phase_receipt_checksum": state.phase_refs[
                        EvaluationPhase.SEALED_VALIDATION
                    ].checksum,
                    "decision_outcome": execution.decision.outcome.value,
                    "accepted": execution.decision.accepted,
                }
        trials.append(trial)

    if formal_round is not None:
        raw_formal_artifacts = formal_round.get("private_artifacts")
        matches = (
            [
                item
                for item in raw_formal_artifacts
                if isinstance(item, Mapping)
                and item.get("kind") == "candidate_artifact"
            ]
            if isinstance(raw_formal_artifacts, list)
            else []
        )
        if len(matches) != 1:
            raise ValueError(
                "generation formal winner has no unique candidate artifact"
            )
        private_artifacts.append(
            {
                **dict(matches[0]),
                "reason": "generation_selected_current_best_source",
            }
        )

    resulting = inputs.parent_current_best
    if formal_round is not None:
        replay = replay_acceptance_ledger(
            [formal_round],
            initial_current_best=inputs.parent_current_best,
            expected_session_genesis_hash=manifest.session_genesis_hash,
            acceptance_protocol=acceptance_protocol,
            expected_ledger_epoch_hash=manifest.ledger_epoch_hash,
            expected_authoring_mode=PROVIDER_AUTHORING_MODE,
        )
        replay.raise_for_continuation()
        resulting = replay.current_best
    outcome = _generation_outcome(trials, formal_round)
    record = {
        "schema_version": GENERATION_LEDGER_RECORD_SCHEMA_VERSION,
        "record_kind": "generation",
        "authoring_mode": PROVIDER_AUTHORING_MODE,
        "generation_id": manifest.generation_id,
        "generation_index": manifest.generation_index,
        "session_genesis_hash": manifest.session_genesis_hash,
        "ledger_epoch_hash": manifest.ledger_epoch_hash,
        "generation_session_contract_hash": (manifest.generation_session_contract_hash),
        "manifest": manifest.to_public_dict(),
        "manifest_receipt_checksum": manifest_ref.checksum,
        "parent_current_best": inputs.parent_current_best.to_public_dict(),
        "parent_current_best_hash": canonical_json_hash(
            inputs.parent_current_best.to_public_dict()
        ),
        "trials": trials,
        "selection": selection.to_public_dict() if selection is not None else None,
        "selection_receipt_checksum": (
            selection_ref.checksum if selection_ref is not None else None
        ),
        "formal": formal,
        "generation_outcome": outcome,
        "resulting_current_best": resulting.to_public_dict(),
        "resulting_current_best_hash": canonical_json_hash(resulting.to_public_dict()),
        "private_artifacts": private_artifacts,
    }
    return record, resulting


def _project_trial(
    inputs: FixedWidthGenerationInputs, state: _LaneState
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    authoring = state.authoring
    prepared = state.prepared
    candidate = None
    verifier = None
    surface = None
    delta = dict(state.operation_delta) if state.operation_delta is not None else None
    contract_hash = None
    artifacts: list[dict[str, Any]] = []
    authoring_projection = None
    if authoring is not None:
        candidate = (
            prepared.record_candidate.to_public_dict()
            if prepared is not None
            else (
                authoring.candidate.to_public_dict()
                if authoring.candidate is not None
                else None
            )
        )
        authoring_projection = {
            "base_public_context_hash": inputs.manifest.base_public_context_hash,
            "lane_intent_hash": state.runtime.lane_intent_hash,
            "prompt_payload_checksum": authoring.prompt_payload_ref.checksum,
            "authoring_evidence_checksum": authoring.authoring_evidence_ref.checksum,
            "sibling_exclusion_hash": state.sibling_exclusion_hash,
        }
        artifacts.extend(
            [
                {
                    "kind": f"{state.runtime.lane_id}_prompt_payload",
                    "uri": authoring.prompt_payload_ref.path,
                    "checksum": authoring.prompt_payload_ref.checksum,
                    "size_bytes": authoring.prompt_payload_ref.size_bytes,
                    "reason": "generation_lane_authoring",
                },
                {
                    "kind": f"{state.runtime.lane_id}_authoring_evidence",
                    "uri": authoring.authoring_evidence_ref.path,
                    "checksum": authoring.authoring_evidence_ref.checksum,
                    "size_bytes": authoring.authoring_evidence_ref.size_bytes,
                    "reason": "generation_lane_authoring",
                },
            ]
        )
    if prepared is not None:
        verifier = prepared.verification.to_public_dict()
        surface = prepared.surface.to_public_dict()
        if prepared.authoring_mechanism_operation_delta is not None:
            delta = dict(prepared.authoring_mechanism_operation_delta)
        contract_hash = prepared.trial_journal.contract_hash
        artifacts.append(
            {
                "kind": f"{state.runtime.lane_id}_candidate_artifact",
                "uri": prepared.record_candidate.path,
                "reason": "generation_lane_candidate_source",
            }
        )
    if state.train_execution is not None:
        artifacts.append(
            _private_trial_ref(
                f"{state.runtime.lane_id}_train_receipt",
                state.train_execution.train_receipt_ref,
            )
        )
    train = _project_train(state)
    public_dev = _project_phase(inputs, state, EvaluationPhase.PUBLIC_DEV)
    public_accept = _project_phase(inputs, state, EvaluationPhase.PUBLIC_ACCEPT)
    trial = {
        "schema_version": GENERATION_TRIAL_RECORD_SCHEMA_VERSION,
        "lane_id": state.runtime.lane_id,
        "trial_id": state.runtime.trial_id,
        "trial_status": state.status,
        "status_reason": state.status_reason,
        "authoring": authoring_projection,
        "candidate": candidate,
        "verifier": verifier,
        "candidate_surface": surface,
        "mechanism_operation_delta": delta,
        "trial_contract_hash": contract_hash,
        "train": train,
        "public_dev": public_dev,
        "public_accept": public_accept,
        "execution_record_hash": None,
        "execution_receipt_checksum": None,
        "formal_record_hash": None,
    }
    if state.phase_refs:
        for phase, ref in state.phase_refs.items():
            artifacts.append(
                _private_trial_ref(
                    f"{state.runtime.lane_id}_{phase.value}_phase_receipt", ref
                )
            )
    return trial, artifacts


def _project_train(state: _LaneState) -> dict[str, Any] | None:
    execution = state.train_execution
    if execution is None:
        return None
    spec = execution.prepared.jobs.train_spec
    result = execution.train_result
    binding = result.gpu_process_binding
    return {
        "job_spec_hash": canonical_json_hash(spec.to_public_dict()),
        "train_result_hash": canonical_json_hash(result.to_public_dict()),
        "train_receipt_checksum": execution.train_receipt_ref.checksum,
        "checkpoint_hash": result.checkpoint.checksum,
        "execution_device": result.execution_device,
        "resource_binding_hash": (
            binding.resource_binding_hash
            if binding is not None
            else (
                spec.attempt_scope.resource_binding_hash
                if spec.attempt_scope is not None
                else None
            )
        ),
    }


def _project_phase(
    inputs: FixedWidthGenerationInputs,
    state: _LaneState,
    phase: EvaluationPhase,
) -> dict[str, Any] | None:
    if not state.phase_results or phase not in state.phase_results:
        return None
    assert state.phase_refs is not None and state.phase_job_spec_hashes is not None
    result = state.phase_results[phase]
    ref = state.phase_refs[phase]
    return {
        "schema_version": GENERATION_PHASE_PROJECTION_SCHEMA_VERSION,
        "phase": phase.value,
        "job_spec_hash": state.phase_job_spec_hashes[phase],
        "evaluation_plan_hash": result.evaluation_plan_hash,
        "phase_result_hash": canonical_json_hash(result.to_private_dict()),
        "phase_receipt_checksum": ref.checksum,
        "checkpoint_hash": result.checkpoint_hash,
        "execution_device": result.execution_device,
        "resource_binding_hash": result.resource_binding_hash,
        "task_evidence_hash": result.task_evidence_hash,
        "public_summary": dict(
            inputs.phased_adapter.project_public_phase_summary(phase_result=result)
        ),
    }


def _generation_outcome(
    trials: Sequence[Mapping[str, Any]], formal_round: Mapping[str, Any] | None
) -> str:
    statuses = [trial["trial_status"] for trial in trials]
    if statuses in (
        ["candidate_failed", "not_started"],
        ["verified_not_run", "candidate_failed"],
        ["candidate_failed", "verified_not_run"],
    ) and all(trial["train"] is None for trial in trials):
        return "paired_authoring_incomplete"
    if formal_round is None:
        if any(status == "candidate_failed" for status in statuses):
            return "candidate_execution_incomplete"
        return "no_selection"
    return (
        "accepted"
        if formal_round["decision"]["accepted"] is True
        else "formal_rejected"
    )


def _private_trial_ref(kind: str, ref: TrialReceiptRef) -> dict[str, Any]:
    return {
        "kind": kind,
        "uri": ref.path,
        "checksum": ref.checksum,
        "size_bytes": ref.size_bytes,
    }


def _private_generation_ref(kind: str, ref: GenerationReceiptRef) -> dict[str, Any]:
    return {
        "kind": kind,
        "uri": ref.path,
        "checksum": ref.checksum,
        "size_bytes": ref.size_bytes,
    }


__all__ = [
    "FixedWidthGenerationInputs",
    "GenerationExecutionResult",
    "GenerationLaneRuntime",
    "execute_fixed_width_generation",
]
