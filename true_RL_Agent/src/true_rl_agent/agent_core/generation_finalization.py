"""Task-neutral target lineage for terminal paired evaluation.

The task binds the final cohort and interprets its scores. This module selects
only the verified reference and accepted terminal checkpoint.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

from true_rl_agent.agent_core.generation_lifecycle import GenerationTerminalBinding
from true_rl_agent.agent_core.generation_record import load_bound_generation_trial_train_receipt
from true_rl_agent.agent_core.session_genesis import VerifiedSessionGenesis
from true_rl_agent.evaluation.binding import canonical_json_hash, require_sha256_digest
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.runner.specs import EvalJobSpec, EvalPhaseJobSpec
from true_rl_agent.agent_core.generation_foundation import GenerationSessionFoundation
from true_rl_agent.agent_core.generation_attempt_scopes import generation_job_attempt_scope
from true_rl_agent.agent_core.generation_operations import read_operation_slot
from true_rl_agent.agent_core.generation_accounting import remaining_sealed_retry_entitlement
from true_rl_agent.agent_core.ledger import primary_ledger_epoch_hash
from true_rl_agent.agent_core.trial_receipts import (
    TRIAL_EXECUTION_CONTRACT_SCHEMA_VERSION, TrialReceiptJournal, TrialReceiptRef,
    load_trial_eval_phase_receipt,
)
from true_rl_agent.evaluation.phases import EvaluationPhase, EvaluationPhaseResult
from true_rl_agent.runner.job_attempts import JobAttemptJournal, JobAttemptScope, JobAttemptReceiptRef
from pathlib import Path
from true_rl_agent.evaluation.subject import validate_experiment_subject_id
from true_rl_agent.fs_contract import create_once_bytes
from true_rl_agent.json_contract import canonical_json_bytes, load_strict_json_object
from true_rl_agent.evaluation.binding import EvaluationBindingV1
from true_rl_agent.runner.checkpoint import file_sha256


FINAL_PLAN_SCHEMA_VERSION = "true_rl_agent.generation_final_confirmation_plan.v1"
FINAL_INFORMATION_FLOW = {
    "acceptance_authority": False,
    "feedback_allowed": False,
    "public_history_projection": False,
    "result_role": "diagnostic_final_comparison_only",
}


@dataclass(frozen=True)
class GenerationFinalPlan:
    """Execution facts plus an opaque, task-owned cohort commitment.

    Phase, seeds per job and total work are distinct fields. Task-specific
    geometry counts or seed namespaces are never renamed into these fields.
    """

    session_id: str
    task_id: str
    task_protocol_hash: str
    evaluation_protocol_hash: str
    seed_cohort_hash: str
    phase: EvaluationPhase
    seed_count_per_job: int
    required_closed_generations: int
    task_commitment: Mapping[str, Any]

    def __post_init__(self) -> None:
        validate_experiment_subject_id(self.session_id)
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("final plan requires a task identity")
        for name in ("task_protocol_hash", "evaluation_protocol_hash", "seed_cohort_hash"):
            require_sha256_digest(getattr(self, name), name)
        for name in ("seed_count_per_job", "required_closed_generations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"final plan {name} must be positive")
        if not isinstance(self.phase, EvaluationPhase):
            raise TypeError("final plan phase must be an EvaluationPhase")
        object.__setattr__(self, "task_commitment", MappingProxyType(deepcopy(dict(self.task_commitment))))

    @property
    def total_eval_seed_units(self) -> int:
        return 2 * self.seed_count_per_job

    @property
    def plan_hash(self) -> str:
        return canonical_json_hash(self._hash_payload())

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": FINAL_PLAN_SCHEMA_VERSION,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "task_protocol_hash": self.task_protocol_hash,
            "evaluation_protocol_hash": self.evaluation_protocol_hash,
            "seed_cohort_hash": self.seed_cohort_hash,
            "required_closed_generations": self.required_closed_generations,
            "execution": {
                "phase": self.phase.value,
                "job_roles": ["genesis", "final"],
                "seed_count_per_job": self.seed_count_per_job,
                "total_eval_seed_units": self.total_eval_seed_units,
            },
            "task_commitment": deepcopy(dict(self.task_commitment)),
            "information_flow": dict(FINAL_INFORMATION_FLOW),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._hash_payload()
        return {**payload, "plan_hash": canonical_json_hash(payload)}

    def genesis_commitment(self) -> dict[str, Any]:
        return {
            "schema_version": FINAL_PLAN_SCHEMA_VERSION,
            "plan_hash": self.plan_hash,
            "seed_cohort_hash": self.seed_cohort_hash,
            "required_closed_generations": self.required_closed_generations,
            "execution": self._hash_payload()["execution"],
            **FINAL_INFORMATION_FLOW,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GenerationFinalPlan":
        execution = value["execution"]
        plan = cls(
            session_id=value["session_id"], task_id=value["task_id"],
            task_protocol_hash=value["task_protocol_hash"],
            evaluation_protocol_hash=value["evaluation_protocol_hash"],
            seed_cohort_hash=value["seed_cohort_hash"],
            phase=EvaluationPhase(execution["phase"]),
            seed_count_per_job=execution["seed_count_per_job"],
            required_closed_generations=value["required_closed_generations"],
            task_commitment=value["task_commitment"],
        )
        if plan.to_dict() != dict(value):
            raise ValueError("final plan differs from its frozen projection")
        return plan


def publish_final_plan(output_dir: Path, plan: GenerationFinalPlan, *, require_existing: bool) -> Path:
    path = output_dir / "final_confirmation" / "plan.json"
    if path.exists():
        stored = GenerationFinalPlan.from_mapping(load_strict_json_object(
            path, require_canonical=True, trailing_newline=True,
        ))
        if stored != plan:
            raise ValueError("existing final plan belongs to another study")
    elif require_existing:
        raise FileNotFoundError("resumed session is missing its final plan")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        create_once_bytes(path, canonical_json_bytes(plan.to_dict(), trailing_newline=True))
    return path


@dataclass(frozen=True)
class FinalEvaluationTargets:
    reference: EvalJobSpec
    terminal: EvalJobSpec
    current_best: CurrentBestEvalStats


def select_final_evaluation_targets(
    *, genesis: VerifiedSessionGenesis, ledger: Sequence[Mapping[str, Any]],
    terminal: GenerationTerminalBinding,
) -> FinalEvaluationTargets:
    if not ledger:
        raise ValueError("final evaluation requires a non-empty ledger")
    current_best = CurrentBestEvalStats.from_mapping(
        ledger[-1].get("resulting_current_best")
    )
    current_best_hash = canonical_json_hash(current_best.to_public_dict())
    if current_best_hash != terminal.current_head_hash:
        raise ValueError("final current-best differs from terminal head")
    genesis_plan = genesis.reference_eval_plan()
    if genesis_plan is None:
        raise ValueError("finalization requires the Genesis reference eval plan")
    require_final_eval_target_matches_current_best(
        genesis_plan,
        current_best=genesis.initial_current_best,
        context="Genesis",
    )
    if current_best.anchor_kind == "reference":
        if current_best != genesis.initial_current_best:
            raise ValueError("terminal reference head differs from Genesis")
        final_target = genesis_plan
    elif current_best.anchor_kind == "candidate":
        accepted = [
            record
            for record in ledger
            if record.get("generation_outcome") == "accepted"
            and record.get("resulting_current_best_hash") == current_best_hash
        ]
        if len(accepted) != 1:
            raise ValueError("final candidate head is not uniquely introduced")
        formal = accepted[0].get("formal")
        if not isinstance(formal, Mapping) or formal.get("accepted") is not True:
            raise ValueError("final candidate lacks accepted formal evidence")
        trial_id = formal.get("selected_trial_id")
        if (
            not isinstance(trial_id, str)
            or trial_id != current_best.accepted_round_id
        ):
            raise ValueError("final candidate trial identity mismatch")
        loaded_train = load_bound_generation_trial_train_receipt(
            accepted[0],
            trial_id=trial_id,
        )
        train_result = loaded_train.result
        if train_result.initial_policy_state_pairing is not None:
            train_result.initial_policy_state_pairing.require_match()
            pairing_hash = train_result.initial_policy_state_pairing.pairing_hash
        else:
            pairing_hash = None
        final_target = replace(
            genesis_plan,
            subject=train_result.subject,
            subject_manifest_hash=train_result.subject_manifest_hash,
            program_source_checksum=train_result.program_source_checksum,
            training_binding_hash=train_result.training_binding_hash,
            capability_profile_hash=train_result.capability_profile_hash,
            checkpoint=train_result.checkpoint,
            checkpoint_output_plan=None,
            initial_policy_state_pairing_hash=pairing_hash,
        )
        require_final_eval_target_matches_current_best(
            final_target,
            current_best=current_best,
            context="final candidate",
        )
    else:
        raise ValueError("terminal head is not an evaluable program anchor")

    return FinalEvaluationTargets(genesis_plan, final_target, current_best)


def require_final_eval_target_matches_current_best(
    target: EvalJobSpec,
    *,
    current_best: CurrentBestEvalStats,
    context: str,
) -> None:
    checkpoint = target.execution_checkpoint()
    expected = {
        "anchor_kind": target.subject.kind,
        "anchor_id": target.subject.subject_id,
        "checkpoint_hash": checkpoint.checksum,
        "subject_manifest_hash": target.subject_manifest_hash,
        "program_source_checksum": target.program_source_checksum,
        "training_binding_hash": target.training_binding_hash,
        "capability_profile_hash": target.capability_profile_hash,
    }
    actual = {
        "anchor_kind": current_best.anchor_kind,
        "anchor_id": current_best.anchor_id,
        "checkpoint_hash": current_best.checkpoint_hash,
        "subject_manifest_hash": current_best.subject_manifest_hash,
        "program_source_checksum": current_best.program_source_checksum,
        "training_binding_hash": current_best.training_binding_hash,
        "capability_profile_hash": current_best.capability_profile_hash,
    }
    if actual != expected:
        raise ValueError(f"{context} eval target lineage mismatch")


@dataclass(frozen=True)
class FinalPhaseReceipt:
    spec: EvalPhaseJobSpec
    result: EvaluationPhaseResult
    receipt_ref: TrialReceiptRef
    attempt_result: JobAttemptReceiptRef


def final_attempt_journal(
    foundation: GenerationSessionFoundation,
    *,
    logical_job_id: str,
) -> JobAttemptJournal | None:
    root = foundation.config.output_dir / "job_attempts"
    if not root.exists():
        return None
    journals: dict[Path, JobAttemptJournal] = {}
    for path in sorted(root.glob("*/attempt_*.reservation.json")):
        journal, reservation = JobAttemptJournal.from_reservation_path(path)
        if journal.logical_job_id != logical_job_id:
            continue
        if (
            not isinstance(journal.scope, JobAttemptScope)
            or journal.scope.generation_id != "generation-final-confirmation"
            or journal.scope.lane_id != "lane-a"
            or journal.scope.budget_contract_hash
            != foundation.cost_contract.contract_hash
            or journal.scope.session_contract_hash
            != foundation.session_contract.contract_hash
            or journal.resource_binding_hash
            != foundation.gpu_bindings[0].resource_binding_hash
        ):
            raise ValueError("final attempt journal binding mismatch")
        read_operation_slot(
            session_id=foundation.config.session_id,
            scope=journal.scope,
            logical_job_id=journal.logical_job_id,
            job_kind=journal.job_kind,
            attempt_index=reservation.payload["attempt_index"],
            cost_contract=foundation.cost_contract,
        )
        journals[journal.root] = journal
    if len(journals) > 1:
        raise ValueError("final job has more than one attempt journal")
    return next(iter(journals.values()), None)


def final_attempt_scope(
    foundation: GenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    logical_job_id: str,
) -> JobAttemptScope:
    existing = final_attempt_journal(
        foundation,
        logical_job_id=logical_job_id,
    )
    if existing is not None:
        assert isinstance(existing.scope, JobAttemptScope)
        return existing.scope
    return generation_job_attempt_scope(
        foundation,
        genesis=genesis,
        generation_id="generation-final-confirmation",
        lane_id="lane-a",
        invocation_segment_id="formal-finalization",
        binding=foundation.gpu_bindings[0],
        max_infrastructure_retries=remaining_sealed_retry_entitlement(foundation),
    )


def run_final_evaluation_phase(
    foundation: GenerationSessionFoundation,
    *, genesis: VerifiedSessionGenesis, eval_plan: EvalJobSpec,
    role: str, phase: EvaluationPhase, plan_hash: str,
    reservation_hash: str, expected_job_id: str, expected_spec_hash: str,
    require_existing: bool = False,
) -> FinalPhaseReceipt:
    """Execute or reuse one reserved phase, including attempt and receipt replay."""
    if role not in {"genesis", "final"}:
        raise ValueError("final phase role is invalid")
    scope = final_attempt_scope(
        foundation,
        genesis=genesis,
        logical_job_id=expected_job_id,
    )
    spec = foundation.composition.backend_adapter.materialize_eval_phase(
        eval_plan=eval_plan,
        phase=phase,
        attempt_scope=scope,
    )
    if (
        spec.job_id != expected_job_id
        or canonical_json_hash(spec.to_public_dict()) != expected_spec_hash
    ):
        raise ValueError("final phase spec differs from Reservation")
    contract = {
        "schema_version": TRIAL_EXECUTION_CONTRACT_SCHEMA_VERSION,
        "session_genesis_hash": genesis.session_genesis_hash,
        "ledger_epoch_hash": primary_ledger_epoch_hash(
            genesis.session_genesis_hash
        ),
        "round_id": spec.job_id,
        "generation_session_contract_hash": foundation.session_contract.contract_hash,
        "final_confirmation_plan_hash": plan_hash,
        "final_confirmation_reservation_hash": reservation_hash,
        "final_confirmation_role": role,
    }
    journal = TrialReceiptJournal(
        foundation.config.output_dir
        / "final_confirmation"
        / "phase_receipts"
        / role,
        contract=contract,
    )
    if require_existing and not (journal.root / "contract.json").is_file():
        raise FileNotFoundError("terminal audit is missing a final phase contract")
    journal.bind_contract()
    loaded = journal.load_eval_phase(spec)
    if loaded is None:
        if require_existing:
            raise FileNotFoundError("terminal audit is missing a final phase receipt")
        phase_result = foundation.runner.run_eval_phase(spec)
        receipt_ref = journal.publish_eval_phase(spec, phase_result)
    else:
        phase_result, receipt_ref = loaded
    reloaded = load_trial_eval_phase_receipt(receipt_ref)
    if reloaded.result != phase_result:
        raise ValueError("final phase receipt replay mismatch")
    attempt_journal = final_attempt_journal(
        foundation,
        logical_job_id=spec.job_id,
    )
    if attempt_journal is None:
        raise ValueError("final phase has no attempt journal")
    attempt_result = attempt_journal.load_reusable_output_result()
    if (
        attempt_result is None
        or attempt_result.payload.get("outcome") != "completed"
        or attempt_result.payload.get("output_hash")
        != canonical_json_hash(phase_result.to_private_dict())
    ):
        raise ValueError("final phase attempt result binding mismatch")
    return FinalPhaseReceipt(spec, phase_result, receipt_ref, attempt_result)


def bind_final_evaluation_job(foundation, *, target, role):
    """Bind execution identity here; the task supplies its cohort CLI overlay."""
    plan = foundation.final_confirmation_plan
    checkpoint = target.execution_checkpoint()
    if file_sha256(Path(checkpoint.metadata["local_path"])) != checkpoint.checksum:
        raise ValueError("final checkpoint differs from the terminal evidence")
    binding = EvaluationBindingV1(eval_protocol_hash=plan.evaluation_protocol_hash,
        acceptance_protocol_hash=target.evaluation_binding.acceptance_protocol_hash,
        task_protocol_hash=plan.task_protocol_hash, seed_cohort_hash=plan.seed_cohort_hash)
    job_id = f"{foundation.config.session_id}-final-confirmation-{role}-eval"
    summary_path = foundation.config.output_dir / "runner" / "final_confirmation" / role / "eval" / "summary.json"
    bound = foundation.composition.backend_adapter.rematerialize_eval_plan(
        eval_plan=replace(target, job_id=job_id, evaluation_binding=binding,
            execution_device=foundation.gpu_bindings[0].process_execution_device,
            gpu_process_binding=foundation.gpu_bindings[0], attempt_scope=None),
        round_id=job_id, summary_path=summary_path.as_posix())
    return foundation.composition.task_plugin.bind_final_eval_plan(bound, plan=plan,
        plan_path=foundation.config.output_dir / "final_confirmation" / "plan.json")
