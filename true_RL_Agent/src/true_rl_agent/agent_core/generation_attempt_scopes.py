"""Recover generation operation scopes from the real attempt journal."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from true_rl_agent.agent_core.generation_foundation import GenerationSessionFoundation
    from true_rl_agent.agent_core.session_genesis import VerifiedSessionGenesis
    from true_rl_agent.runner.specs import GpuProcessBinding

from true_rl_agent.runner.job_attempts import JobAttemptJournal, JobAttemptScope

_OPERATIONS = (
    "train",
    "public_dev",
    "public_accept",
    "sealed_validation",
)


@dataclass(frozen=True)
class GenerationLaneOperationScopes:
    """Attempt authority for the four fixed lane operations."""

    train: JobAttemptScope
    public_dev: JobAttemptScope
    public_accept: JobAttemptScope
    sealed_validation: JobAttemptScope

    def __post_init__(self) -> None:
        for operation in _OPERATIONS:
            if not isinstance(self.for_operation(operation), JobAttemptScope):
                raise TypeError(f"{operation} attempt scope is invalid")

    def for_operation(self, operation: str) -> JobAttemptScope:
        if operation not in _OPERATIONS:
            raise ValueError("generation operation is invalid")
        return getattr(self, operation)

    def to_generation_lane_runtime_kwargs(self) -> dict[str, JobAttemptScope]:
        """Project directly into the task-neutral GenerationLaneRuntime API."""

        return {
            "attempt_scope": self.train,
            "public_dev_attempt_scope": self.public_dev,
            "public_accept_attempt_scope": self.public_accept,
            "sealed_attempt_scope": self.sealed_validation,
        }


def recover_generation_lane_operation_scopes(
    *,
    attempt_journal_root: Path,
    generation_id: str,
    lane_id: str,
    current_invocation_segment_id: str,
    current_scopes: GenerationLaneOperationScopes,
) -> GenerationLaneOperationScopes:
    """Keep started jobs on their original segment and new jobs on this segment."""

    if not isinstance(attempt_journal_root, Path):
        raise TypeError("attempt journal root must be a Path")
    if not isinstance(current_scopes, GenerationLaneOperationScopes):
        raise TypeError("current operation scopes are invalid")
    root = attempt_journal_root.resolve(strict=False)
    _validate_current_scopes(
        root=root,
        generation_id=generation_id,
        lane_id=lane_id,
        current_invocation_segment_id=current_invocation_segment_id,
        scopes=current_scopes,
    )
    if not root.exists():
        return current_scopes
    if root.is_symlink() or not root.is_dir():
        raise ValueError("attempt journal root must be a regular directory")

    expected_jobs = _expected_operation_jobs(
        generation_id=generation_id,
        lane_id=lane_id,
    )
    operation_by_job_id = {
        logical_job_id: operation
        for operation, (logical_job_id, _job_kind) in expected_jobs.items()
    }
    journals: dict[str, dict[Path, JobAttemptJournal]] = {
        operation: {} for operation in _OPERATIONS
    }
    indices: dict[tuple[str, Path], set[int]] = {}
    for reservation_path in sorted(root.glob("*/attempt_*.reservation.json")):
        journal, reservation = JobAttemptJournal.from_reservation_path(reservation_path)
        operation = operation_by_job_id.get(journal.logical_job_id)
        if operation is None:
            continue
        expected_job_id, expected_job_kind = expected_jobs[operation]
        if (
            journal.logical_job_id != expected_job_id
            or journal.job_kind != expected_job_kind
        ):
            raise ValueError("generation operation job identity mismatch")
        if not isinstance(journal.scope, JobAttemptScope):
            raise TypeError("generation operation used a pre-Genesis scope")
        _validate_recovered_scope(
            operation=operation,
            recovered=journal.scope,
            current=current_scopes.for_operation(operation),
        )
        journal_root = journal.root.resolve(strict=True)
        journals[operation].setdefault(journal_root, journal)
        indices.setdefault((operation, journal_root), set()).add(
            reservation.payload["attempt_index"]
        )

    resolved: dict[str, JobAttemptScope] = {}
    for operation in _OPERATIONS:
        operation_journals = journals[operation]
        if not operation_journals:
            resolved[operation] = current_scopes.for_operation(operation)
            continue
        if len(operation_journals) != 1:
            raise ValueError(
                f"{operation} operation has multiple attempt journals"
            )
        journal_root, journal = next(iter(operation_journals.items()))
        _validate_attempt_chain(
            journal,
            indices=indices[(operation, journal_root)],
        )
        assert isinstance(journal.scope, JobAttemptScope)
        resolved[operation] = journal.scope

    return GenerationLaneOperationScopes(**resolved)


def recover_parent_bootstrap_attempt_scope(
    *,
    attempt_journal_root: Path,
    session_id: str,
    current_invocation_segment_id: str,
    current_scope: JobAttemptScope,
) -> JobAttemptScope:
    """Keep a started parent bootstrap on its segment; bind new work to this one."""

    if not isinstance(attempt_journal_root, Path):
        raise TypeError("attempt journal root must be a Path")
    root = attempt_journal_root.resolve(strict=False)
    if (
        not isinstance(session_id, str)
        or not session_id
        or not isinstance(current_invocation_segment_id, str)
        or not current_invocation_segment_id
        or not isinstance(current_scope, JobAttemptScope)
    ):
        raise ValueError("current parent-bootstrap scope is invalid")
    if (
        current_scope.journal_root.resolve(strict=False) != root
        or current_scope.generation_id != "generation-parent-public-dev-bootstrap"
        or current_scope.lane_id != "lane-a"
        or current_scope.invocation_segment_id != current_invocation_segment_id
    ):
        raise ValueError("current parent-bootstrap scope identity mismatch")
    if not root.exists():
        return current_scope
    if root.is_symlink() or not root.is_dir():
        raise ValueError("attempt journal root must be a regular directory")

    expected_job_id = f"{session_id}-eval-public-dev"
    journals: dict[Path, JobAttemptJournal] = {}
    indices: dict[Path, set[int]] = {}
    for reservation_path in sorted(root.glob("*/attempt_*.reservation.json")):
        journal, reservation = JobAttemptJournal.from_reservation_path(reservation_path)
        if journal.logical_job_id != expected_job_id:
            continue
        if journal.job_kind != "eval_phase" or not isinstance(
            journal.scope, JobAttemptScope
        ):
            raise ValueError("parent-bootstrap attempt identity mismatch")
        _validate_recovered_scope(
            operation="parent_bootstrap",
            recovered=journal.scope,
            current=current_scope,
        )
        journal_root = journal.root.resolve(strict=True)
        journals.setdefault(journal_root, journal)
        indices.setdefault(journal_root, set()).add(
            reservation.payload["attempt_index"]
        )
    if not journals:
        return current_scope
    if len(journals) != 1:
        raise ValueError("parent-bootstrap has multiple attempt journals")
    journal_root, journal = next(iter(journals.items()))
    _validate_attempt_chain(journal, indices=indices[journal_root])
    assert isinstance(journal.scope, JobAttemptScope)
    return journal.scope


def _validate_current_scopes(
    *,
    root: Path,
    generation_id: str,
    lane_id: str,
    current_invocation_segment_id: str,
    scopes: GenerationLaneOperationScopes,
) -> None:
    if not isinstance(generation_id, str) or not generation_id:
        raise ValueError("generation_id is required")
    if lane_id not in {"lane-a", "lane-b"}:
        raise ValueError("generation lane_id is invalid")
    if (
        not isinstance(current_invocation_segment_id, str)
        or not current_invocation_segment_id
    ):
        raise ValueError("current invocation segment ID is required")
    anchor = scopes.train
    common_fields = (
        "session_genesis_hash",
        "session_contract_hash",
        "generation_id",
        "lane_id",
        "invocation_segment_id",
        "budget_contract_hash",
        "resource_binding_hash",
    )
    for operation in _OPERATIONS:
        scope = scopes.for_operation(operation)
        if scope.journal_root.resolve(strict=False) != root:
            raise ValueError("current attempt scope journal root mismatch")
        if any(
            getattr(scope, field_name) != getattr(anchor, field_name)
            for field_name in common_fields
        ):
            raise ValueError("current operation scope identity mismatch")
    if (
        anchor.generation_id != generation_id
        or anchor.lane_id != lane_id
        or anchor.invocation_segment_id != current_invocation_segment_id
    ):
        raise ValueError("current operation scope lane identity mismatch")


def _validate_recovered_scope(
    *,
    operation: str,
    recovered: JobAttemptScope,
    current: JobAttemptScope,
) -> None:
    for field_name in (
        "session_genesis_hash",
        "session_contract_hash",
        "generation_id",
        "lane_id",
        "budget_contract_hash",
        "resource_binding_hash",
        "max_infrastructure_retries",
    ):
        if getattr(recovered, field_name) != getattr(current, field_name):
            raise ValueError(f"{operation} recovered scope {field_name} mismatch")
    if recovered.journal_root.resolve(strict=False) != current.journal_root.resolve(
        strict=False
    ):
        raise ValueError(f"{operation} recovered scope journal root mismatch")


def _validate_attempt_chain(
    journal: JobAttemptJournal,
    *,
    indices: set[int],
) -> None:
    if not indices:
        raise ValueError("operation attempt chain is empty")
    ordered = sorted(indices)
    if ordered != list(range(ordered[-1] + 1)):
        raise ValueError("operation attempt reservation chain is not contiguous")
    for attempt_index in ordered:
        reservation = journal.load_reservation(attempt_index)
        result_path = reservation.path.with_name(
            reservation.path.name.replace(".reservation.", ".result.")
        )
        if result_path.exists():
            journal.load_result(attempt_index)
        elif attempt_index != ordered[-1]:
            raise ValueError("operation attempt chain has an open predecessor")


def _expected_operation_jobs(
    *,
    generation_id: str,
    lane_id: str,
) -> dict[str, tuple[str, str]]:
    trial_id = f"{generation_id}-{lane_id}"
    return {
        "train": (f"{trial_id}-train", "train"),
        "public_dev": (f"{trial_id}-eval-public-dev", "eval_phase"),
        "public_accept": (f"{trial_id}-eval-public-accept", "eval_phase"),
        "sealed_validation": (
            f"{trial_id}-eval-sealed-validation",
            "eval_phase",
        ),
    }


__all__ = [
    "GenerationLaneOperationScopes",
    "recover_generation_lane_operation_scopes",
    "recover_parent_bootstrap_attempt_scope",
]


def generation_job_attempt_scope(
    foundation: GenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    generation_id: str,
    lane_id: str,
    invocation_segment_id: str,
    binding: GpuProcessBinding,
    max_infrastructure_retries: int = 0,
) -> JobAttemptScope:
    return JobAttemptScope(
        journal_root=foundation.config.output_dir / "job_attempts",
        session_genesis_hash=genesis.session_genesis_hash,
        session_contract_hash=foundation.session_contract.contract_hash,
        generation_id=generation_id,
        lane_id=lane_id,
        invocation_segment_id=invocation_segment_id,
        budget_contract_hash=foundation.cost_contract.contract_hash,
        attempt_budget_units=max_infrastructure_retries + 1,
        max_infrastructure_retries=max_infrastructure_retries,
        resource_binding_hash=binding.resource_binding_hash,
    )

