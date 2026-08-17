"""Write-once Stair protocol for a never-fed final paired confirmation."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from true_rl_agent.backends.base import (
    BACKEND_EXECUTION_ENVIRONMENT_VERIFICATION_SCHEMA_VERSION,
)
from true_rl_agent.backends.isaac_lab_execution_environment import (
    ISAAC_LAB_RSL_RL_BACKEND_ID,
)
from true_rl_agent.evaluation.binding import (
    canonical_json_hash,
    require_sha256_checksum,
    require_sha256_digest,
)
from true_rl_agent.evaluation.subject import validate_experiment_subject_id
from true_rl_agent.fs_contract import create_once_bytes, read_bounded_regular_file
from true_rl_agent.json_contract import (
    canonical_json_bytes,
    strict_json_object_loads,
)
from true_rl_agent.tasks.stair_climbing.config import (
    STAIR_TASK_ID,
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.generalization_contract import (
    STAIR_GENERALIZATION_SUMMARY_SCHEMA_VERSION,
    validate_stair_generalization_summary,
)
from true_rl_agent.tasks.stair_climbing.isaaclab_execution import (
    StairEvaluationExecutionResult,
)
from true_rl_agent.tasks.stair_climbing.seed_policy import (
    STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE,
)
from true_rl_agent.tasks.stair_climbing.training_distribution import (
    STAIR_EXECUTION_SOURCE_PREFLIGHT_SCHEMA_VERSION,
    stair_training_distribution_upstream_source_evidence,
)

STAIR_FINAL_CONFIRMATION_PLAN_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.final_confirmation_plan.v1"
)
STAIR_FINAL_CONFIRMATION_RESERVATION_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.final_confirmation_reservation.v1"
)
STAIR_FINAL_CONFIRMATION_RESULT_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.final_confirmation_result.v1"
)
STAIR_FINAL_CONFIRMATION_DIRNAME = "final_confirmation"
STAIR_FINAL_CONFIRMATION_PLAN_FILENAME = "plan.json"
STAIR_FINAL_CONFIRMATION_RESERVATION_FILENAME = "reservation.json"
STAIR_FINAL_CONFIRMATION_RESULT_FILENAME = "result.json"
STAIR_FINAL_CONFIRMATION_REQUIRED_ROUNDS = 15
STAIR_FINAL_CONFIRMATION_SPLIT = "sealed_validation"
MAX_STAIR_FINAL_CONFIRMATION_PLAN_BYTES = 64 * 1024
MAX_STAIR_FINAL_CONFIRMATION_RESERVATION_BYTES = 64 * 1024
# Two canonical v4 7-seed x 35-cell summaries currently encode to about 116 KiB.
# Keep bounded headroom for task-contract growth without permitting unbounded reads.
MAX_STAIR_FINAL_CONFIRMATION_RESULT_BYTES = 256 * 1024

_COMPARISON = {
    "genesis_role": "reference_program_genesis_checkpoint",
    "final_role": "formal_final_current_best_after_round_15",
    "pairing": "same_private_cohort_by_run_index",
    "execution_order": ["genesis", "final"],
    "delta_direction": "final_minus_genesis",
}
_INFORMATION_FLOW = {
    "acceptance_authority": False,
    "feedback_allowed": False,
    "public_history_projection": False,
    "raw_seed_values_persisted": False,
    "result_role": "diagnostic_final_generalization_confirmation_only",
}
_STATISTICS = {
    "unit": "one_equal_height_tread_cell_macro_score_per_seed",
    "reported": [
        "canonical_genesis_generalization_summary",
        "canonical_final_generalization_summary",
        "paired_seed_deltas",
        "paired_delta_mean",
        "paired_delta_standard_error",
    ],
    "decision_rule": "none_diagnostic_report_only",
}


@dataclass(frozen=True)
class StairFinalConfirmationPlan:
    """Public-safe commitment frozen before Session Genesis exists."""

    session_id: str
    task_protocol_hash: str
    eval_protocol_hash: str
    evaluation_profile_hash: str
    seed_cohort_hash: str
    seed_count: int
    cell_count: int
    episodes_per_seed_per_cell: int
    plan_hash: str | None = None

    def __post_init__(self) -> None:
        validate_experiment_subject_id(self.session_id)
        for field_name in (
            "task_protocol_hash",
            "eval_protocol_hash",
            "evaluation_profile_hash",
            "seed_cohort_hash",
        ):
            require_sha256_digest(
                getattr(self, field_name),
                f"Stair final confirmation {field_name}",
            )
        for field_name in (
            "seed_count",
            "cell_count",
            "episodes_per_seed_per_cell",
        ):
            _positive_int(
                getattr(self, field_name),
                f"Stair final confirmation {field_name}",
            )
        expected = canonical_json_hash(self._hash_payload())
        if self.plan_hash is not None:
            require_sha256_digest(
                self.plan_hash,
                "Stair final confirmation plan hash",
            )
            if self.plan_hash != expected:
                raise ValueError("Stair final confirmation plan hash mismatch")
        object.__setattr__(self, "plan_hash", expected)

    @classmethod
    def from_mapping(cls, value: Any) -> "StairFinalConfirmationPlan":
        mapping = _exact_mapping(
            value,
            {
                "schema_version",
                "session_id",
                "task_id",
                "task_protocol_hash",
                "eval_protocol_hash",
                "evaluation_profile_hash",
                "confirmation_namespace",
                "cohort",
                "comparison",
                "statistics",
                "required_closed_rounds",
                "single_use",
                "information_flow",
                "plan_hash",
            },
            "Stair final confirmation plan",
        )
        if (
            mapping.get("schema_version")
            != STAIR_FINAL_CONFIRMATION_PLAN_SCHEMA_VERSION
            or mapping.get("task_id") != STAIR_TASK_ID
            or mapping.get("confirmation_namespace")
            != STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE
            or mapping.get("required_closed_rounds")
            != STAIR_FINAL_CONFIRMATION_REQUIRED_ROUNDS
            or mapping.get("single_use") is not True
        ):
            raise ValueError("Stair final confirmation plan identity mismatch")
        if dict(_mapping(mapping.get("comparison"), "comparison")) != _COMPARISON:
            raise ValueError("Stair final confirmation comparison contract mismatch")
        if dict(_mapping(mapping.get("statistics"), "statistics")) != _STATISTICS:
            raise ValueError("Stair final confirmation statistics contract mismatch")
        if (
            dict(_mapping(mapping.get("information_flow"), "information_flow"))
            != _INFORMATION_FLOW
        ):
            raise ValueError("Stair final confirmation information-flow mismatch")
        cohort = _exact_mapping(
            mapping.get("cohort"),
            {
                "source_shape",
                "seed_cohort_hash",
                "seed_count",
                "cell_count",
                "episodes_per_seed_per_cell",
            },
            "Stair final confirmation cohort",
        )
        if cohort.get("source_shape") != STAIR_FINAL_CONFIRMATION_SPLIT:
            raise ValueError("Stair final confirmation cohort shape mismatch")
        return cls(
            session_id=_text(mapping.get("session_id"), "session_id"),
            task_protocol_hash=require_sha256_digest(
                mapping.get("task_protocol_hash"), "task_protocol_hash"
            ),
            eval_protocol_hash=require_sha256_digest(
                mapping.get("eval_protocol_hash"), "eval_protocol_hash"
            ),
            evaluation_profile_hash=require_sha256_digest(
                mapping.get("evaluation_profile_hash"),
                "evaluation_profile_hash",
            ),
            seed_cohort_hash=require_sha256_digest(
                cohort.get("seed_cohort_hash"), "seed_cohort_hash"
            ),
            seed_count=_positive_int(cohort.get("seed_count"), "seed_count"),
            cell_count=_positive_int(cohort.get("cell_count"), "cell_count"),
            episodes_per_seed_per_cell=_positive_int(
                cohort.get("episodes_per_seed_per_cell"),
                "episodes_per_seed_per_cell",
            ),
            plan_hash=require_sha256_digest(mapping.get("plan_hash"), "plan_hash"),
        )

    def to_dict(self) -> dict[str, Any]:
        assert self.plan_hash is not None
        return {**self._hash_payload(), "plan_hash": self.plan_hash}

    def genesis_commitment(self) -> dict[str, Any]:
        """Return the only plan projection allowed into Genesis task defaults."""

        assert self.plan_hash is not None
        return {
            "schema_version": STAIR_FINAL_CONFIRMATION_PLAN_SCHEMA_VERSION,
            "plan_hash": self.plan_hash,
            "confirmation_namespace": STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE,
            "seed_cohort_hash": self.seed_cohort_hash,
            "seed_count": self.seed_count,
            "required_closed_rounds": STAIR_FINAL_CONFIRMATION_REQUIRED_ROUNDS,
            "single_use": True,
            **_INFORMATION_FLOW,
        }

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": STAIR_FINAL_CONFIRMATION_PLAN_SCHEMA_VERSION,
            "session_id": self.session_id,
            "task_id": STAIR_TASK_ID,
            "task_protocol_hash": self.task_protocol_hash,
            "eval_protocol_hash": self.eval_protocol_hash,
            "evaluation_profile_hash": self.evaluation_profile_hash,
            "confirmation_namespace": STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE,
            "cohort": {
                "source_shape": STAIR_FINAL_CONFIRMATION_SPLIT,
                "seed_cohort_hash": self.seed_cohort_hash,
                "seed_count": self.seed_count,
                "cell_count": self.cell_count,
                "episodes_per_seed_per_cell": self.episodes_per_seed_per_cell,
            },
            "comparison": {
                **_COMPARISON,
                "execution_order": list(_COMPARISON["execution_order"]),
            },
            "statistics": {
                **_STATISTICS,
                "reported": list(_STATISTICS["reported"]),
            },
            "required_closed_rounds": STAIR_FINAL_CONFIRMATION_REQUIRED_ROUNDS,
            "single_use": True,
            "information_flow": dict(_INFORMATION_FLOW),
        }


def build_stair_final_confirmation_plan(
    *,
    session_id: str,
    config: StairClimbingTaskConfig,
) -> StairFinalConfirmationPlan:
    if not isinstance(config, StairClimbingTaskConfig):
        raise TypeError("Stair final confirmation requires StairClimbingTaskConfig")
    resolver = config.seed_resolver()
    profile = config.evaluation_profile
    return StairFinalConfirmationPlan(
        session_id=session_id,
        task_protocol_hash=config.task_protocol_hash,
        eval_protocol_hash=config.evaluation_protocol().protocol_hash,
        evaluation_profile_hash=profile.profile_hash,
        seed_cohort_hash=resolver.namespace_cohort_hash(
            STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE
        ),
        seed_count=config.sealed_validation_seed_count,
        cell_count=len(profile.cells),
        episodes_per_seed_per_cell=profile.episodes_per_seed_per_cell,
    )


def create_stair_final_confirmation_plan(
    *,
    output_dir: Path,
    session_id: str,
    config: StairClimbingTaskConfig,
) -> tuple[StairFinalConfirmationPlan, Path]:
    output_dir = output_dir.resolve()
    _assert_pre_genesis_output_state(output_dir)
    plan = build_stair_final_confirmation_plan(
        session_id=session_id,
        config=config,
    )
    path = stair_final_confirmation_plan_path(output_dir)
    _atomic_create_json(
        path,
        plan.to_dict(),
        max_bytes=MAX_STAIR_FINAL_CONFIRMATION_PLAN_BYTES,
        artifact_name="Stair final confirmation plan",
    )
    loaded = load_stair_final_confirmation_plan(
        path,
        expected_session_id=session_id,
        expected_config=config,
    )
    return loaded, path


def load_stair_final_confirmation_plan(
    path: Path,
    *,
    expected_session_id: str | None = None,
    expected_config: StairClimbingTaskConfig | None = None,
) -> StairFinalConfirmationPlan:
    plan = StairFinalConfirmationPlan.from_mapping(
        _load_bounded_canonical_json_object(
            path,
            context="Stair final confirmation plan",
            max_bytes=MAX_STAIR_FINAL_CONFIRMATION_PLAN_BYTES,
        )
    )
    if expected_session_id is not None and plan.session_id != expected_session_id:
        raise ValueError("Stair final confirmation plan session mismatch")
    if expected_config is not None:
        expected = build_stair_final_confirmation_plan(
            session_id=plan.session_id,
            config=expected_config,
        )
        if plan.to_dict() != expected.to_dict():
            raise ValueError("Stair final confirmation plan differs from task config")
    return plan


def create_stair_final_confirmation_reservation(
    *,
    output_dir: Path,
    plan: StairFinalConfirmationPlan,
    session_genesis_hash: str,
    ledger_checksum: str,
    genesis_checkpoint_hash: str,
    final_checkpoint_hash: str,
    source_manifest_hash: str,
    backend_execution_environment_manifest_hash: str,
    execution_device: str,
    final_anchor: Mapping[str, Any],
) -> tuple["StairFinalConfirmationReservation", Path]:
    path = stair_final_confirmation_reservation_path(output_dir.resolve())
    result_path = stair_final_confirmation_result_path(output_dir.resolve())
    if path.exists() or result_path.exists():
        raise FileExistsError("Stair final confirmation is single-use and already consumed")
    reservation = StairFinalConfirmationReservation(
        plan_hash=require_sha256_digest(plan.plan_hash, "plan_hash"),
        session_id=plan.session_id,
        session_genesis_hash=session_genesis_hash,
        ledger_checksum=ledger_checksum,
        genesis_checkpoint_hash=genesis_checkpoint_hash,
        final_checkpoint_hash=final_checkpoint_hash,
        source_manifest_hash=source_manifest_hash,
        backend_execution_environment_manifest_hash=(
            backend_execution_environment_manifest_hash
        ),
        execution_device=execution_device,
        final_anchor=_json_mapping(final_anchor, "final_anchor"),
    )
    _atomic_create_json(
        path,
        reservation.to_dict(),
        max_bytes=MAX_STAIR_FINAL_CONFIRMATION_RESERVATION_BYTES,
        artifact_name="Stair final confirmation reservation",
    )
    loaded = load_stair_final_confirmation_reservation(path, expected_plan=plan)
    if loaded != reservation:
        raise RuntimeError("Stair final confirmation reservation write verification failed")
    return loaded, path


@dataclass(frozen=True)
class StairFinalConfirmationReservation:
    plan_hash: str
    session_id: str
    session_genesis_hash: str
    ledger_checksum: str
    genesis_checkpoint_hash: str
    final_checkpoint_hash: str
    source_manifest_hash: str
    backend_execution_environment_manifest_hash: str
    execution_device: str
    final_anchor: Mapping[str, Any]
    reservation_hash: str | None = None

    def __post_init__(self) -> None:
        require_sha256_digest(self.plan_hash, "reservation plan_hash")
        validate_experiment_subject_id(self.session_id)
        require_sha256_digest(
            self.session_genesis_hash, "reservation session_genesis_hash"
        )
        for field_name in (
            "ledger_checksum",
            "genesis_checkpoint_hash",
            "final_checkpoint_hash",
        ):
            require_sha256_checksum(
                getattr(self, field_name), f"reservation {field_name}"
            )
        for field_name in (
            "source_manifest_hash",
            "backend_execution_environment_manifest_hash",
        ):
            require_sha256_digest(
                getattr(self, field_name), f"reservation {field_name}"
            )
        if not isinstance(self.execution_device, str) or not self.execution_device:
            raise ValueError("reservation execution_device must be non-empty")
        object.__setattr__(
            self,
            "final_anchor",
            _json_mapping(self.final_anchor, "reservation final_anchor"),
        )
        expected = canonical_json_hash(self._hash_payload())
        if self.reservation_hash is not None:
            require_sha256_digest(
                self.reservation_hash, "reservation reservation_hash"
            )
            if self.reservation_hash != expected:
                raise ValueError("Stair final confirmation reservation hash mismatch")
        object.__setattr__(self, "reservation_hash", expected)

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        expected_plan: StairFinalConfirmationPlan,
    ) -> "StairFinalConfirmationReservation":
        mapping = _exact_mapping(
            value,
            {
                "schema_version",
                "plan_hash",
                "session_id",
                "session_genesis_hash",
                "ledger_checksum",
                "genesis_checkpoint_hash",
                "final_checkpoint_hash",
                "source_manifest_hash",
                "backend_execution_environment_manifest_hash",
                "execution_device",
                "final_anchor",
                "single_use",
                "reservation_hash",
            },
            "Stair final confirmation reservation",
        )
        if (
            mapping.get("schema_version")
            != STAIR_FINAL_CONFIRMATION_RESERVATION_SCHEMA_VERSION
            or mapping.get("single_use") is not True
        ):
            raise ValueError("Stair final confirmation reservation identity mismatch")
        reservation = cls(
            plan_hash=require_sha256_digest(mapping.get("plan_hash"), "plan_hash"),
            session_id=_text(mapping.get("session_id"), "session_id"),
            session_genesis_hash=require_sha256_digest(
                mapping.get("session_genesis_hash"), "session_genesis_hash"
            ),
            ledger_checksum=require_sha256_checksum(
                mapping.get("ledger_checksum"), "ledger_checksum"
            ),
            genesis_checkpoint_hash=require_sha256_checksum(
                mapping.get("genesis_checkpoint_hash"), "genesis_checkpoint_hash"
            ),
            final_checkpoint_hash=require_sha256_checksum(
                mapping.get("final_checkpoint_hash"), "final_checkpoint_hash"
            ),
            source_manifest_hash=require_sha256_digest(
                mapping.get("source_manifest_hash"), "source_manifest_hash"
            ),
            backend_execution_environment_manifest_hash=require_sha256_digest(
                mapping.get("backend_execution_environment_manifest_hash"),
                "backend_execution_environment_manifest_hash",
            ),
            execution_device=_text(
                mapping.get("execution_device"), "execution_device"
            ),
            final_anchor=_json_mapping(mapping.get("final_anchor"), "final_anchor"),
            reservation_hash=require_sha256_digest(
                mapping.get("reservation_hash"), "reservation_hash"
            ),
        )
        if (
            reservation.plan_hash != expected_plan.plan_hash
            or reservation.session_id != expected_plan.session_id
        ):
            raise ValueError("Stair final confirmation reservation plan binding mismatch")
        return reservation

    def to_dict(self) -> dict[str, Any]:
        assert self.reservation_hash is not None
        return {**self._hash_payload(), "reservation_hash": self.reservation_hash}

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": STAIR_FINAL_CONFIRMATION_RESERVATION_SCHEMA_VERSION,
            "plan_hash": self.plan_hash,
            "session_id": self.session_id,
            "session_genesis_hash": self.session_genesis_hash,
            "ledger_checksum": self.ledger_checksum,
            "genesis_checkpoint_hash": self.genesis_checkpoint_hash,
            "final_checkpoint_hash": self.final_checkpoint_hash,
            "source_manifest_hash": self.source_manifest_hash,
            "backend_execution_environment_manifest_hash": (
                self.backend_execution_environment_manifest_hash
            ),
            "execution_device": self.execution_device,
            "final_anchor": dict(self.final_anchor),
            "single_use": True,
        }


def load_stair_final_confirmation_reservation(
    path: Path,
    *,
    expected_plan: StairFinalConfirmationPlan,
) -> StairFinalConfirmationReservation:
    return StairFinalConfirmationReservation.from_mapping(
        _load_bounded_canonical_json_object(
            path,
            context="Stair final confirmation reservation",
            max_bytes=MAX_STAIR_FINAL_CONFIRMATION_RESERVATION_BYTES,
        ),
        expected_plan=expected_plan,
    )


def build_stair_final_confirmation_result(
    *,
    plan: StairFinalConfirmationPlan,
    reservation: StairFinalConfirmationReservation,
    config: StairClimbingTaskConfig,
    genesis_execution: StairEvaluationExecutionResult,
    final_execution: StairEvaluationExecutionResult,
) -> "StairFinalConfirmationResult":
    _require_plan_reservation_binding(plan, reservation)
    _require_plan_config_binding(plan, config)
    genesis_identity, genesis, genesis_summary_hash = (
        _validated_confirmation_execution(
            execution=genesis_execution,
            plan=plan,
            reservation=reservation,
            config=config,
            role="genesis",
            expected_checkpoint_hash=reservation.genesis_checkpoint_hash,
        )
    )
    final_identity, final, final_summary_hash = _validated_confirmation_execution(
        execution=final_execution,
        plan=plan,
        reservation=reservation,
        config=config,
        role="final",
        expected_checkpoint_hash=reservation.final_checkpoint_hash,
    )
    genesis_scores = _summary_seed_scores(genesis, plan.seed_count, "genesis")
    final_scores = _summary_seed_scores(final, plan.seed_count, "final")
    deltas = [
        right - left
        for left, right in zip(genesis_scores, final_scores, strict=True)
    ]
    delta_std = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    return StairFinalConfirmationResult(
        plan_hash=require_sha256_digest(plan.plan_hash, "plan_hash"),
        reservation_hash=require_sha256_digest(
            reservation.reservation_hash, "reservation_hash"
        ),
        session_id=plan.session_id,
        seed_cohort_hash=plan.seed_cohort_hash,
        genesis_execution_identity=genesis_identity,
        final_execution_identity=final_identity,
        genesis_summary_hash=genesis_summary_hash,
        final_summary_hash=final_summary_hash,
        genesis_summary=genesis,
        final_summary=final,
        paired_seed_deltas=tuple(deltas),
        paired_mean=statistics.fmean(deltas),
        paired_standard_error=delta_std / math.sqrt(len(deltas)),
    )


@dataclass(frozen=True)
class StairFinalConfirmationResult:
    plan_hash: str
    reservation_hash: str
    session_id: str
    seed_cohort_hash: str
    genesis_execution_identity: Mapping[str, Any]
    final_execution_identity: Mapping[str, Any]
    genesis_summary_hash: str
    final_summary_hash: str
    genesis_summary: Mapping[str, Any]
    final_summary: Mapping[str, Any]
    paired_seed_deltas: tuple[float, ...]
    paired_mean: float
    paired_standard_error: float
    result_hash: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("plan_hash", "reservation_hash", "seed_cohort_hash"):
            require_sha256_digest(getattr(self, field_name), f"result {field_name}")
        validate_experiment_subject_id(self.session_id)
        for field_name in ("genesis_summary_hash", "final_summary_hash"):
            require_sha256_digest(
                getattr(self, field_name), f"result {field_name}"
            )
        for field_name in (
            "genesis_execution_identity",
            "final_execution_identity",
        ):
            object.__setattr__(
                self,
                field_name,
                _json_mapping(getattr(self, field_name), f"result {field_name}"),
            )
        object.__setattr__(
            self,
            "genesis_summary",
            _json_mapping(self.genesis_summary, "result genesis summary"),
        )
        object.__setattr__(
            self,
            "final_summary",
            _json_mapping(self.final_summary, "result final summary"),
        )
        deltas = tuple(
            _finite_number(value, "result paired seed delta")
            for value in self.paired_seed_deltas
        )
        if not deltas:
            raise ValueError("result paired seed deltas must be non-empty")
        object.__setattr__(self, "paired_seed_deltas", deltas)
        object.__setattr__(
            self, "paired_mean", _finite_number(self.paired_mean, "paired mean")
        )
        object.__setattr__(
            self,
            "paired_standard_error",
            _nonnegative_number(
                self.paired_standard_error, "paired standard_error"
            ),
        )
        expected = canonical_json_hash(self._hash_payload())
        if self.result_hash is not None:
            require_sha256_digest(self.result_hash, "result result_hash")
            if self.result_hash != expected:
                raise ValueError("Stair final confirmation result hash mismatch")
        object.__setattr__(self, "result_hash", expected)

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        plan: StairFinalConfirmationPlan,
        reservation: StairFinalConfirmationReservation,
        config: StairClimbingTaskConfig,
    ) -> "StairFinalConfirmationResult":
        mapping = _exact_mapping(
            value,
            {
                "schema_version",
                "plan_hash",
                "reservation_hash",
                "session_id",
                "confirmation_namespace",
                "seed_cohort_hash",
                "genesis",
                "final",
                "paired",
                "single_use",
                "information_flow",
                "result_hash",
            },
            "Stair final confirmation result",
        )
        if (
            mapping.get("schema_version")
            != STAIR_FINAL_CONFIRMATION_RESULT_SCHEMA_VERSION
            or mapping.get("confirmation_namespace")
            != STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE
            or mapping.get("single_use") is not True
            or dict(_mapping(mapping.get("information_flow"), "information_flow"))
            != _INFORMATION_FLOW
        ):
            raise ValueError("Stair final confirmation result identity mismatch")
        genesis = _exact_mapping(
            mapping.get("genesis"),
            {
                "execution_identity",
                "generalization_summary_hash",
                "generalization_summary",
            },
            "Stair final confirmation result genesis",
        )
        final = _exact_mapping(
            mapping.get("final"),
            {
                "execution_identity",
                "generalization_summary_hash",
                "generalization_summary",
            },
            "Stair final confirmation result final",
        )
        paired = _exact_mapping(
            mapping.get("paired"),
            {"seed_deltas", "mean", "standard_error"},
            "Stair final confirmation result paired",
        )
        result = cls(
            plan_hash=require_sha256_digest(mapping.get("plan_hash"), "plan_hash"),
            reservation_hash=require_sha256_digest(
                mapping.get("reservation_hash"), "reservation_hash"
            ),
            session_id=_text(mapping.get("session_id"), "session_id"),
            seed_cohort_hash=require_sha256_digest(
                mapping.get("seed_cohort_hash"), "seed_cohort_hash"
            ),
            genesis_execution_identity=_json_mapping(
                genesis.get("execution_identity"),
                "genesis execution_identity",
            ),
            final_execution_identity=_json_mapping(
                final.get("execution_identity"),
                "final execution_identity",
            ),
            genesis_summary_hash=require_sha256_digest(
                genesis.get("generalization_summary_hash"),
                "genesis generalization_summary_hash",
            ),
            final_summary_hash=require_sha256_digest(
                final.get("generalization_summary_hash"),
                "final generalization_summary_hash",
            ),
            genesis_summary=_json_mapping(
                genesis.get("generalization_summary"), "genesis summary"
            ),
            final_summary=_json_mapping(
                final.get("generalization_summary"), "final summary"
            ),
            paired_seed_deltas=tuple(
                _number_sequence(paired.get("seed_deltas"), "paired seed_deltas")
            ),
            paired_mean=_finite_number(paired.get("mean"), "paired mean"),
            paired_standard_error=_nonnegative_number(
                paired.get("standard_error"), "paired standard_error"
            ),
            result_hash=require_sha256_digest(
                mapping.get("result_hash"), "result_hash"
            ),
        )
        _require_plan_reservation_binding(plan, reservation)
        _require_plan_config_binding(plan, config)
        if (
            result.plan_hash != plan.plan_hash
            or result.reservation_hash != reservation.reservation_hash
            or result.session_id != plan.session_id
            or result.seed_cohort_hash != plan.seed_cohort_hash
        ):
            raise ValueError("Stair final confirmation result binding mismatch")
        canonical_genesis = _validate_stored_confirmation_role(
            execution_identity=result.genesis_execution_identity,
            summary=result.genesis_summary,
            summary_hash=result.genesis_summary_hash,
            plan=plan,
            reservation=reservation,
            config=config,
            role="genesis",
            expected_checkpoint_hash=reservation.genesis_checkpoint_hash,
        )
        canonical_final = _validate_stored_confirmation_role(
            execution_identity=result.final_execution_identity,
            summary=result.final_summary,
            summary_hash=result.final_summary_hash,
            plan=plan,
            reservation=reservation,
            config=config,
            role="final",
            expected_checkpoint_hash=reservation.final_checkpoint_hash,
        )
        genesis_scores = _summary_seed_scores(
            canonical_genesis, plan.seed_count, "genesis"
        )
        final_scores = _summary_seed_scores(canonical_final, plan.seed_count, "final")
        expected_deltas = tuple(
            right - left
            for left, right in zip(genesis_scores, final_scores, strict=True)
        )
        expected_std = statistics.stdev(expected_deltas) if len(expected_deltas) > 1 else 0.0
        if (
            result.paired_seed_deltas != expected_deltas
            or not math.isclose(
                result.paired_mean,
                statistics.fmean(expected_deltas),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                result.paired_standard_error,
                expected_std / math.sqrt(len(expected_deltas)),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError("Stair final confirmation paired statistics mismatch")
        return result

    def to_dict(self) -> dict[str, Any]:
        assert self.result_hash is not None
        return {**self._hash_payload(), "result_hash": self.result_hash}

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": STAIR_FINAL_CONFIRMATION_RESULT_SCHEMA_VERSION,
            "plan_hash": self.plan_hash,
            "reservation_hash": self.reservation_hash,
            "session_id": self.session_id,
            "confirmation_namespace": STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE,
            "seed_cohort_hash": self.seed_cohort_hash,
            "genesis": {
                "execution_identity": dict(self.genesis_execution_identity),
                "generalization_summary_hash": self.genesis_summary_hash,
                "generalization_summary": dict(self.genesis_summary),
            },
            "final": {
                "execution_identity": dict(self.final_execution_identity),
                "generalization_summary_hash": self.final_summary_hash,
                "generalization_summary": dict(self.final_summary),
            },
            "paired": {
                "seed_deltas": list(self.paired_seed_deltas),
                "mean": self.paired_mean,
                "standard_error": self.paired_standard_error,
            },
            "single_use": True,
            "information_flow": dict(_INFORMATION_FLOW),
        }


def _validated_confirmation_execution(
    *,
    execution: StairEvaluationExecutionResult,
    plan: StairFinalConfirmationPlan,
    reservation: StairFinalConfirmationReservation,
    config: StairClimbingTaskConfig,
    role: str,
    expected_checkpoint_hash: str,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    if not isinstance(execution, StairEvaluationExecutionResult):
        raise TypeError(
            f"Stair final confirmation {role} requires an evaluation execution result"
        )
    if set(execution.summaries) != {STAIR_FINAL_CONFIRMATION_SPLIT}:
        raise ValueError(
            f"Stair final confirmation {role} execution split mismatch"
        )
    split_summary = _mapping(
        execution.public_summaries()[STAIR_FINAL_CONFIRMATION_SPLIT],
        f"Stair final confirmation {role} split summary",
    )
    summary = _canonical_confirmation_summary(
        config,
        _mapping(
            split_summary.get("generalization"),
            f"Stair final confirmation {role} generalization summary",
        ),
    )
    summary_hash = canonical_json_hash(summary)
    identity = execution.execution_identity()
    _validate_stored_confirmation_role(
        execution_identity=identity,
        summary=summary,
        summary_hash=summary_hash,
        plan=plan,
        reservation=reservation,
        config=config,
        role=role,
        expected_checkpoint_hash=expected_checkpoint_hash,
    )
    return identity, summary, summary_hash


def _validate_stored_confirmation_role(
    *,
    execution_identity: Mapping[str, Any],
    summary: Mapping[str, Any],
    summary_hash: str,
    plan: StairFinalConfirmationPlan,
    reservation: StairFinalConfirmationReservation,
    config: StairClimbingTaskConfig,
    role: str,
    expected_checkpoint_hash: str,
) -> dict[str, Any]:
    identity = _exact_mapping(
        execution_identity,
        {
            "schema_version",
            "checkpoint_hash",
            "seed_namespace",
            "seed_cohort_hash",
            "splits",
            "execution_device",
            "environment_source_preflight",
            "backend_execution_environment_verification",
        },
        f"Stair final confirmation {role} execution identity",
    )
    if (
        identity.get("schema_version")
        != "true_rl_agent.task.stair_climbing.eval_execution_identity.v1"
        or identity.get("checkpoint_hash") != expected_checkpoint_hash
        or identity.get("seed_namespace")
        != STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE
        or identity.get("seed_cohort_hash") != plan.seed_cohort_hash
        or identity.get("splits") != [STAIR_FINAL_CONFIRMATION_SPLIT]
        or identity.get("execution_device") != reservation.execution_device
    ):
        raise ValueError(
            f"Stair final confirmation {role} execution identity mismatch"
        )
    require_sha256_checksum(
        identity.get("checkpoint_hash"),
        f"Stair final confirmation {role} checkpoint hash",
    )
    require_sha256_digest(
        identity.get("seed_cohort_hash"),
        f"Stair final confirmation {role} seed cohort hash",
    )

    source = _exact_mapping(
        identity.get("environment_source_preflight"),
        {
            "schema_version",
            "status",
            "task_protocol_hash",
            "source_manifest_hash",
        },
        f"Stair final confirmation {role} source preflight",
    )
    contract = config.protocol_v2.runner_owned_training_distribution
    if contract is None:
        raise ValueError("Stair final confirmation requires a source contract")
    expected_source_hash = require_sha256_digest(
        stair_training_distribution_upstream_source_evidence(contract).get(
            "source_manifest_hash"
        ),
        "Stair final confirmation expected source manifest hash",
    )
    if (
        reservation.source_manifest_hash != expected_source_hash
        or source.get("schema_version")
        != STAIR_EXECUTION_SOURCE_PREFLIGHT_SCHEMA_VERSION
        or source.get("status") != "pass"
        or source.get("task_protocol_hash") != config.task_protocol_hash
        or source.get("source_manifest_hash") != reservation.source_manifest_hash
    ):
        raise ValueError(
            f"Stair final confirmation {role} source preflight mismatch"
        )
    require_sha256_digest(
        source.get("task_protocol_hash"),
        f"Stair final confirmation {role} source task protocol hash",
    )
    require_sha256_digest(
        source.get("source_manifest_hash"),
        f"Stair final confirmation {role} source manifest hash",
    )

    backend = _exact_mapping(
        identity.get("backend_execution_environment_verification"),
        {
            "schema_version",
            "backend_id",
            "status",
            "expected_manifest_hash",
            "actual_manifest_hash",
        },
        f"Stair final confirmation {role} backend verification",
    )
    if (
        backend.get("schema_version")
        != BACKEND_EXECUTION_ENVIRONMENT_VERIFICATION_SCHEMA_VERSION
        or backend.get("backend_id") != ISAAC_LAB_RSL_RL_BACKEND_ID
        or backend.get("status") != "pass"
        or backend.get("expected_manifest_hash")
        != reservation.backend_execution_environment_manifest_hash
        or backend.get("actual_manifest_hash")
        != reservation.backend_execution_environment_manifest_hash
    ):
        raise ValueError(
            f"Stair final confirmation {role} backend verification mismatch"
        )
    require_sha256_digest(
        backend.get("expected_manifest_hash"),
        f"Stair final confirmation {role} backend manifest hash",
    )

    canonical = _canonical_confirmation_summary(config, summary)
    if dict(summary) != canonical:
        raise ValueError(
            f"Stair final confirmation {role} summary is not canonical"
        )
    expected_summary_hash = canonical_json_hash(canonical)
    require_sha256_digest(
        summary_hash,
        f"Stair final confirmation {role} summary hash",
    )
    if summary_hash != expected_summary_hash:
        raise ValueError(
            f"Stair final confirmation {role} summary hash mismatch"
        )
    return canonical


def write_stair_final_confirmation_result(
    *,
    output_dir: Path,
    result: StairFinalConfirmationResult,
    plan: StairFinalConfirmationPlan,
    reservation: StairFinalConfirmationReservation,
    config: StairClimbingTaskConfig,
) -> Path:
    path = stair_final_confirmation_result_path(output_dir.resolve())
    verified = StairFinalConfirmationResult.from_mapping(
        result.to_dict(),
        plan=plan,
        reservation=reservation,
        config=config,
    )
    _atomic_create_json(
        path,
        verified.to_dict(),
        max_bytes=MAX_STAIR_FINAL_CONFIRMATION_RESULT_BYTES,
        artifact_name="Stair final confirmation result",
    )
    loaded = load_stair_final_confirmation_result(
        path,
        plan=plan,
        reservation=reservation,
        config=config,
    )
    if loaded != verified:
        raise RuntimeError("Stair final confirmation result write verification failed")
    return path


def load_stair_final_confirmation_result(
    path: Path,
    *,
    plan: StairFinalConfirmationPlan,
    reservation: StairFinalConfirmationReservation,
    config: StairClimbingTaskConfig,
) -> StairFinalConfirmationResult:
    return StairFinalConfirmationResult.from_mapping(
        _load_bounded_canonical_json_object(
            path,
            context="Stair final confirmation result",
            max_bytes=MAX_STAIR_FINAL_CONFIRMATION_RESULT_BYTES,
        ),
        plan=plan,
        reservation=reservation,
        config=config,
    )


def stair_final_confirmation_plan_path(output_dir: Path) -> Path:
    return output_dir / STAIR_FINAL_CONFIRMATION_DIRNAME / STAIR_FINAL_CONFIRMATION_PLAN_FILENAME


def stair_final_confirmation_reservation_path(output_dir: Path) -> Path:
    return (
        output_dir
        / STAIR_FINAL_CONFIRMATION_DIRNAME
        / STAIR_FINAL_CONFIRMATION_RESERVATION_FILENAME
    )


def stair_final_confirmation_result_path(output_dir: Path) -> Path:
    return output_dir / STAIR_FINAL_CONFIRMATION_DIRNAME / STAIR_FINAL_CONFIRMATION_RESULT_FILENAME


def _assert_pre_genesis_output_state(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ValueError("Stair final confirmation output_dir must be a regular directory")
    forbidden = (
        output_dir / "session_genesis.json",
        output_dir / "ledger.jsonl",
        output_dir / "runner",
        output_dir / "live_invocations",
        output_dir / "authoring",
    )
    existing = [path.name for path in forbidden if path.exists()]
    if existing:
        raise ValueError(
            "Stair final confirmation plan must be created before Genesis/state: "
            + ", ".join(existing)
        )
    if stair_final_confirmation_plan_path(output_dir).exists():
        raise FileExistsError("Stair final confirmation plan is create-once")


def _atomic_create_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    max_bytes: int,
    artifact_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("Stair final confirmation artifact directory is invalid")
    encoded = canonical_json_bytes(value, trailing_newline=True)
    if len(encoded) > max_bytes:
        raise ValueError(f"{artifact_name} exceeds {max_bytes} bytes")
    create_once_bytes(path, encoded)


def _load_bounded_canonical_json_object(
    path: Path,
    *,
    context: str,
    max_bytes: int,
) -> dict[str, Any]:
    payload = read_bounded_regular_file(
        Path(path),
        max_bytes=max_bytes,
        label=context,
    )
    parsed = strict_json_object_loads(payload, context=context)
    if payload != canonical_json_bytes(parsed, trailing_newline=True):
        raise ValueError(f"{context} is not canonical JSON")
    return parsed


def _require_plan_reservation_binding(
    plan: StairFinalConfirmationPlan,
    reservation: StairFinalConfirmationReservation,
) -> None:
    if not isinstance(reservation, StairFinalConfirmationReservation):
        raise TypeError("Stair final confirmation requires a verified reservation")
    if (
        reservation.plan_hash != plan.plan_hash
        or reservation.session_id != plan.session_id
    ):
        raise ValueError("Stair final confirmation reservation plan binding mismatch")


def _require_plan_config_binding(
    plan: StairFinalConfirmationPlan,
    config: StairClimbingTaskConfig,
) -> None:
    if not isinstance(config, StairClimbingTaskConfig):
        raise TypeError("Stair final confirmation requires StairClimbingTaskConfig")
    expected = build_stair_final_confirmation_plan(
        session_id=plan.session_id,
        config=config,
    )
    if expected != plan:
        raise ValueError("Stair final confirmation plan differs from task config")


def _canonical_confirmation_summary(
    config: StairClimbingTaskConfig,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    if summary.get("schema_version") != STAIR_GENERALIZATION_SUMMARY_SCHEMA_VERSION:
        raise ValueError(
            "Stair final confirmation requires the current generalization summary schema"
        )
    return validate_stair_generalization_summary(
        profile=config.evaluation_profile,
        semantics=config.evaluation_semantics,
        evaluation_protocol=config.evaluation_protocol(),
        split=STAIR_FINAL_CONFIRMATION_SPLIT,
        summary=summary,
    )


def _summary_seed_scores(
    summary: Mapping[str, Any],
    expected_count: int,
    role: str,
) -> list[float]:
    raw_scores = summary.get("seed_macro_scores")
    if not isinstance(raw_scores, list) or len(raw_scores) != expected_count:
        raise ValueError(f"Stair final confirmation {role} seed scores mismatch")
    scores = [_finite_number(value, f"{role} seed score") for value in raw_scores]
    if any(not 0.0 <= value <= 1.0 for value in scores):
        raise ValueError(f"Stair final confirmation {role} seed score is out of range")
    return scores


def _number_sequence(value: Any, field_name: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array")
    return [_finite_number(item, field_name) for item in value]


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be a finite number")
    return parsed


def _nonnegative_number(value: Any, field_name: str) -> float:
    parsed = _finite_number(value, field_name)
    if parsed < 0.0:
        raise ValueError(f"{field_name} must be non-negative")
    return parsed


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _json_mapping(value: Any, field_name: str) -> dict[str, Any]:
    mapping = _mapping(value, field_name)
    normalized = strict_json_object_loads(
        canonical_json_bytes(dict(mapping)),
        context=field_name,
    )
    return normalized


def _exact_mapping(
    value: Any,
    expected_fields: set[str],
    field_name: str,
) -> Mapping[str, Any]:
    mapping = _mapping(value, field_name)
    if set(mapping) != expected_fields:
        raise ValueError(f"{field_name} fields mismatch")
    return mapping


__all__ = [
    "STAIR_FINAL_CONFIRMATION_DIRNAME",
    "MAX_STAIR_FINAL_CONFIRMATION_PLAN_BYTES",
    "MAX_STAIR_FINAL_CONFIRMATION_RESERVATION_BYTES",
    "MAX_STAIR_FINAL_CONFIRMATION_RESULT_BYTES",
    "STAIR_FINAL_CONFIRMATION_PLAN_FILENAME",
    "STAIR_FINAL_CONFIRMATION_PLAN_SCHEMA_VERSION",
    "STAIR_FINAL_CONFIRMATION_REQUIRED_ROUNDS",
    "STAIR_FINAL_CONFIRMATION_RESERVATION_FILENAME",
    "STAIR_FINAL_CONFIRMATION_RESULT_FILENAME",
    "STAIR_FINAL_CONFIRMATION_SPLIT",
    "StairFinalConfirmationPlan",
    "StairFinalConfirmationReservation",
    "StairFinalConfirmationResult",
    "build_stair_final_confirmation_plan",
    "build_stair_final_confirmation_result",
    "create_stair_final_confirmation_plan",
    "create_stair_final_confirmation_reservation",
    "load_stair_final_confirmation_plan",
    "load_stair_final_confirmation_reservation",
    "load_stair_final_confirmation_result",
    "stair_final_confirmation_plan_path",
    "stair_final_confirmation_reservation_path",
    "stair_final_confirmation_result_path",
    "write_stair_final_confirmation_result",
]
