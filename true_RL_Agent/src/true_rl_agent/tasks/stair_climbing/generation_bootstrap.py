"""Stair-owned parent public-dev evidence for generation screening."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from true_rl_agent.agent_core.generation_record import (
    load_bound_generation_trial_phase_receipt,
    validate_generation_ledger_record,
)
from true_rl_agent.agent_core.ledger import primary_ledger_epoch_hash
from true_rl_agent.agent_core.session_genesis import VerifiedSessionGenesis
from true_rl_agent.agent_core.trial_receipts import TrialReceiptJournal, TrialReceiptRef
from true_rl_agent.backends.base import BackendAdapter
from true_rl_agent.evaluation.acceptance_protocol import AcceptanceProtocol
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.evaluation.phases import EvaluationPhase, EvaluationPhaseResult
from true_rl_agent.runner.base import Runner
from true_rl_agent.runner.job_attempts import JobAttemptScope
from true_rl_agent.tasks.stair_climbing.phased_adapter import (
    StairPhasedEvaluationAdapter,
)
from true_rl_agent.tasks.stair_climbing.selection import (
    StairPublicDevScreenEvidence,
)


@dataclass(frozen=True)
class StairGenerationGuardReference:
    """One validated parent scorecard and its immutable phase receipt."""

    evidence: StairPublicDevScreenEvidence
    phase_result: EvaluationPhaseResult
    phase_receipt_ref: TrialReceiptRef

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, StairPublicDevScreenEvidence):
            raise TypeError("Stair guard reference evidence is invalid")
        if (
            not isinstance(self.phase_result, EvaluationPhaseResult)
            or self.phase_result.phase is not EvaluationPhase.PUBLIC_DEV
        ):
            raise TypeError("Stair guard reference requires a public-dev phase")
        if (
            not isinstance(self.phase_receipt_ref, TrialReceiptRef)
            or self.phase_receipt_ref.kind != "eval_phase"
        ):
            raise TypeError("Stair guard reference receipt is invalid")
        if self.evidence.phase_result_hash != canonical_json_hash(
            self.phase_result.to_private_dict()
        ):
            raise ValueError("Stair guard reference evidence/result mismatch")

    def assert_matches_parent(self, parent: CurrentBestEvalStats) -> None:
        if not isinstance(parent, CurrentBestEvalStats):
            raise TypeError("Stair guard reference parent is invalid")
        expected_subject_kind = parent.anchor_kind
        if expected_subject_kind == "candidate":
            expected_subject_kind = "candidate"
        if (
            self.evidence.subject.kind != expected_subject_kind
            or self.evidence.subject.subject_id != parent.anchor_id
            or self.evidence.checkpoint_hash != parent.checkpoint_hash
            or self.evidence.training_binding_hash != parent.training_binding_hash
            or self.evidence.capability_profile_hash != parent.capability_profile_hash
            or self.evidence.evaluation_binding_hash != parent.evaluation_binding_hash
        ):
            raise ValueError("Stair guard reference is not the frozen current-best")


def run_genesis_parent_public_dev(
    *,
    genesis: VerifiedSessionGenesis,
    adapter: StairPhasedEvaluationAdapter,
    backend: BackendAdapter,
    runner: Runner,
    receipt_root: Path,
    generation_session_contract_hash: str,
    attempt_scope: JobAttemptScope | None = None,
) -> StairGenerationGuardReference:
    """Run or reload the single session-level Genesis public-dev phase."""

    if not isinstance(genesis, VerifiedSessionGenesis):
        raise TypeError("generation bootstrap requires verified Session Genesis")
    eval_plan = genesis.reference_eval_plan()
    if eval_plan is None:
        raise ValueError(
            "first Stair generation requires a reference-program Genesis anchor"
        )
    if (
        adapter.config.task_protocol_hash
        != genesis.evaluation_binding.task_protocol_hash
    ):
        raise ValueError("generation bootstrap Stair task binding mismatch")
    contract = {
        "schema_version": "true_rl_agent.trial_execution_contract.v1",
        "round_id": "generation-parent-public-dev-bootstrap",
        "session_genesis_hash": genesis.session_genesis_hash,
        "ledger_epoch_hash": primary_ledger_epoch_hash(genesis.session_genesis_hash),
        "generation_session_contract_hash": generation_session_contract_hash,
        "purpose": "generation_parent_public_dev_guard_reference",
        "evaluation_plan_hash": canonical_json_hash(eval_plan.to_public_dict()),
        "parent_current_best": genesis.initial_current_best.to_public_dict(),
    }
    journal = TrialReceiptJournal(receipt_root, contract=contract)
    journal.bind_contract()
    phase_spec = backend.materialize_eval_phase(
        eval_plan=eval_plan,
        phase=EvaluationPhase.PUBLIC_DEV,
        attempt_scope=attempt_scope,
    )
    loaded = journal.load_eval_phase(phase_spec)
    if loaded is None:
        result = runner.run_eval_phase(phase_spec)
        receipt_ref = journal.publish_eval_phase(phase_spec, result)
    else:
        result, receipt_ref = loaded
    return build_stair_generation_guard_reference(
        adapter=adapter,
        parent=genesis.initial_current_best,
        trial_id="genesis-parent",
        phase_result=result,
        phase_receipt_ref=receipt_ref,
    )


def build_stair_generation_guard_reference(
    *,
    adapter: StairPhasedEvaluationAdapter,
    parent: CurrentBestEvalStats,
    trial_id: str,
    phase_result: EvaluationPhaseResult,
    phase_receipt_ref: TrialReceiptRef,
) -> StairGenerationGuardReference:
    """Build every generation guard through the sole Stair screen producer."""

    evidence = adapter.project_public_dev_screen(
        trial_id=trial_id,
        phase_result=phase_result,
        screen_role="guard_reference",
        train_result=None,
        train_receipt_checksum=None,
        mechanism_operation_delta=None,
    )
    guard = StairGenerationGuardReference(
        evidence=evidence,
        phase_result=phase_result,
        phase_receipt_ref=phase_receipt_ref,
    )
    guard.assert_matches_parent(parent)
    return guard


def load_stair_next_generation_guard_reference(
    *,
    generation_record: Mapping[str, Any],
    previous_parent: CurrentBestEvalStats,
    acceptance_protocol: AcceptanceProtocol,
    adapter: StairPhasedEvaluationAdapter,
) -> StairGenerationGuardReference:
    """Recover an accepted winner's public-dev receipt for the next generation."""

    next_parent = validate_generation_ledger_record(
        generation_record,
        acceptance_protocol=acceptance_protocol,
        expected_parent=previous_parent,
    )
    if generation_record.get("generation_outcome") != "accepted":
        raise ValueError(
            "only an accepted generation creates a new Stair guard reference"
        )
    selection = generation_record.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("accepted generation has no selection")
    selected_trial_id = selection.get("selected_trial_id")
    if not isinstance(selected_trial_id, str) or not selected_trial_id:
        raise ValueError("accepted generation has no selected trial")
    loaded = load_bound_generation_trial_phase_receipt(
        generation_record,
        trial_id=selected_trial_id,
        phase=EvaluationPhase.PUBLIC_DEV,
    )
    return build_stair_generation_guard_reference(
        adapter=adapter,
        parent=next_parent,
        trial_id=selected_trial_id,
        phase_result=loaded.result,
        phase_receipt_ref=loaded.receipt_ref,
    )


def replay_stair_generation_guard_chain(
    *,
    generation_records: Sequence[Mapping[str, Any]],
    initial_parent: CurrentBestEvalStats,
    initial_guard_reference: StairGenerationGuardReference,
    acceptance_protocol: AcceptanceProtocol,
    adapter: StairPhasedEvaluationAdapter,
    expected_generation_session_contract_hash: str | None = None,
) -> tuple[CurrentBestEvalStats, StairGenerationGuardReference]:
    """Replay the authoritative head and its Stair public-dev guard in lockstep."""

    if not isinstance(generation_records, Sequence) or isinstance(
        generation_records, str | bytes
    ):
        raise TypeError("Stair generation guard replay records must be a sequence")
    initial_guard_reference.assert_matches_parent(initial_parent)
    current_parent = initial_parent
    current_guard = initial_guard_reference
    for record in generation_records:
        if not isinstance(record, Mapping):
            raise TypeError("Stair generation guard replay record must be an object")
        next_parent = validate_generation_ledger_record(
            record,
            acceptance_protocol=acceptance_protocol,
            expected_parent=current_parent,
            expected_generation_session_contract_hash=(
                expected_generation_session_contract_hash
            ),
        )
        if record.get("generation_outcome") == "accepted":
            current_guard = load_stair_next_generation_guard_reference(
                generation_record=record,
                previous_parent=current_parent,
                acceptance_protocol=acceptance_protocol,
                adapter=adapter,
            )
            current_guard.assert_matches_parent(next_parent)
        elif next_parent != current_parent:
            raise ValueError("non-accepted generation changed the replay current-best")
        current_parent = next_parent
    current_guard.assert_matches_parent(current_parent)
    return current_parent, current_guard


__all__ = [
    "StairGenerationGuardReference",
    "build_stair_generation_guard_reference",
    "load_stair_next_generation_guard_reference",
    "replay_stair_generation_guard_chain",
    "run_genesis_parent_public_dev",
]
