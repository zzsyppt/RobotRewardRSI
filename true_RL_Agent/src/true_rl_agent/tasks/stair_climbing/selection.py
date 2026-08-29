"""Task-owned public-dev screening for two Stair sibling trials."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from true_rl_agent.evaluation.binding import (
    canonical_json_hash,
    require_sha256_checksum,
    require_sha256_digest,
)
from true_rl_agent.evaluation.phases import EvaluationPhase, EvaluationPhaseResult
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.runner.base import TrainJobResult
from true_rl_agent.tasks.stair_climbing.config import (
    STAIR_TASK_ID,
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.evaluation_contract import (
    validate_stair_split_summary,
)
from true_rl_agent.tasks.stair_climbing.generalization_contract import (
    validate_stair_generalization_summary,
)
from true_rl_agent.tasks.stair_climbing.phase_contract import (
    STAIR_PHASE_EVIDENCE_SCHEMA_VERSION,
)
from true_rl_agent.tasks.stair_climbing.public_projection import (
    build_stair_candidate_runtime_effect_evidence,
    require_stair_candidate_runtime_effect_evidence,
)
from true_rl_agent.tasks.stair_climbing.seed_policy import (
    STAIR_FORMAL_EVAL_NAMESPACE,
)
from true_rl_agent.tasks.stair_climbing.unified_scoring import (
    build_unified_score_projection,
    validate_objective_family_mask,
    validate_unified_stair_summary,
)

STAIR_SCREEN_EVIDENCE_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.public_dev_screen_evidence.v4"
)
_STAIR_LEGACY_SCREEN_EVIDENCE_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.public_dev_screen_evidence.v3"
)
STAIR_SELECTION_CONTRACT_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.selection_contract.v5"
)
STAIR_SELECTION_RESULT_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.selection_result.v5"
)


@dataclass(frozen=True)
class StairPublicDevScreenEvidence:
    """Validated public-dev evidence with no acceptance authority."""

    trial_id: str
    screen_role: str
    train_receipt_checksum: str | None
    runtime_effect_evidence: Mapping[str, Any] | None
    phase_result_hash: str
    evaluation_plan_hash: str
    task_id: str
    subject: ExperimentSubject
    checkpoint_hash: str
    eval_protocol_hash: str
    evaluation_binding_hash: str
    training_binding_hash: str
    capability_profile_hash: str
    execution_environment_hash: str
    execution_device: str
    resource_binding_hash: str | None
    anchor_score_stats: Mapping[str, Any]
    anchor_seed_macro_scores: tuple[float, ...]
    generalization_score_stats: Mapping[str, Any]
    generalization_seed_macro_scores: tuple[float, ...]
    objective_family_mask_hash: str | None = None
    schema_version: str = STAIR_SCREEN_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in ("trial_id", "task_id", "execution_device"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Stair screen {field_name} is required")
        if not isinstance(self.subject, ExperimentSubject):
            raise TypeError("Stair screen subject is invalid")
        if self.schema_version not in {
            STAIR_SCREEN_EVIDENCE_SCHEMA_VERSION,
            _STAIR_LEGACY_SCREEN_EVIDENCE_SCHEMA_VERSION,
        }:
            raise ValueError("Stair screen evidence schema is unsupported")
        if (
            self.schema_version == _STAIR_LEGACY_SCREEN_EVIDENCE_SCHEMA_VERSION
            and self.objective_family_mask_hash is not None
        ):
            raise ValueError("legacy Stair screen cannot carry a family mask")
        if self.screen_role == "candidate":
            require_sha256_checksum(
                self.train_receipt_checksum,
                "Stair candidate screen train receipt checksum",
            )
            runtime_effect = require_stair_candidate_runtime_effect_evidence(
                self.runtime_effect_evidence
            )
        elif self.screen_role == "guard_reference":
            if (
                self.train_receipt_checksum is not None
                or self.runtime_effect_evidence is not None
            ):
                raise ValueError(
                    "Stair guard screen cannot carry candidate train evidence"
                )
            runtime_effect = None
        else:
            raise ValueError("Stair screen role is invalid")
        require_sha256_checksum(self.checkpoint_hash, "Stair screen checkpoint_hash")
        for field_name in (
            "phase_result_hash",
            "evaluation_plan_hash",
            "eval_protocol_hash",
            "evaluation_binding_hash",
            "training_binding_hash",
            "capability_profile_hash",
            "execution_environment_hash",
        ):
            require_sha256_digest(getattr(self, field_name), f"Stair screen {field_name}")
        if self.resource_binding_hash is not None:
            require_sha256_digest(
                self.resource_binding_hash,
                "Stair screen resource_binding_hash",
            )
        if self.objective_family_mask_hash is not None:
            require_sha256_digest(
                self.objective_family_mask_hash,
                "Stair screen objective family mask hash",
            )
        for field_name in ("anchor_score_stats", "generalization_score_stats"):
            if not isinstance(getattr(self, field_name), Mapping):
                raise TypeError(f"Stair screen {field_name} must be an object")
        for field_name in (
            "anchor_seed_macro_scores",
            "generalization_seed_macro_scores",
        ):
            scores = getattr(self, field_name)
            if not scores:
                raise ValueError(f"Stair screen {field_name} cannot be empty")
            for score in scores:
                _finite_score(score, f"Stair screen {field_name}")
        object.__setattr__(self, "runtime_effect_evidence", runtime_effect)

    @property
    def evidence_hash(self) -> str:
        return canonical_json_hash(self.to_selector_dict())

    @property
    def comparison_domain(self) -> tuple[str, ...]:
        return (
            self.task_id,
            self.eval_protocol_hash,
            self.evaluation_binding_hash,
            self.training_binding_hash,
            self.capability_profile_hash,
            self.execution_environment_hash,
            self.objective_family_mask_hash or "legacy_unmasked_objective",
        )

    def to_selector_dict(self) -> dict[str, Any]:
        """Serialize selector evidence; this is not a prompt-memory projection."""

        payload = {
            "schema_version": self.schema_version,
            "trial_id": self.trial_id,
            "screen_role": self.screen_role,
            "train_receipt_checksum": self.train_receipt_checksum,
            "runtime_effect_evidence": (
                dict(self.runtime_effect_evidence)
                if self.runtime_effect_evidence is not None
                else None
            ),
            "phase_result_hash": self.phase_result_hash,
            "evaluation_plan_hash": self.evaluation_plan_hash,
            "task_id": self.task_id,
            "subject": self.subject.to_public_dict(),
            "checkpoint_hash": self.checkpoint_hash,
            "eval_protocol_hash": self.eval_protocol_hash,
            "evaluation_binding_hash": self.evaluation_binding_hash,
            "training_binding_hash": self.training_binding_hash,
            "capability_profile_hash": self.capability_profile_hash,
            "execution_environment_hash": self.execution_environment_hash,
            "execution_device": self.execution_device,
            "resource_binding_hash": self.resource_binding_hash,
            "anchor_score_stats": dict(self.anchor_score_stats),
            "anchor_seed_macro_scores": list(self.anchor_seed_macro_scores),
            "generalization_score_stats": dict(self.generalization_score_stats),
            "generalization_seed_macro_scores": list(
                self.generalization_seed_macro_scores
            ),
            "acceptance_authority": False,
        }
        if self.schema_version == STAIR_SCREEN_EVIDENCE_SCHEMA_VERSION:
            payload["objective_family_mask_hash"] = self.objective_family_mask_hash
        return payload

    @classmethod
    def from_mapping(cls, value: Any) -> "StairPublicDevScreenEvidence":
        base_fields = {
            "schema_version",
            "trial_id",
            "screen_role",
            "train_receipt_checksum",
            "runtime_effect_evidence",
            "phase_result_hash",
            "evaluation_plan_hash",
            "task_id",
            "subject",
            "checkpoint_hash",
            "eval_protocol_hash",
            "evaluation_binding_hash",
            "training_binding_hash",
            "capability_profile_hash",
            "execution_environment_hash",
            "execution_device",
            "resource_binding_hash",
            "anchor_score_stats",
            "anchor_seed_macro_scores",
            "generalization_score_stats",
            "generalization_seed_macro_scores",
            "acceptance_authority",
        }
        if not isinstance(value, Mapping):
            raise ValueError("Stair screen evidence fields mismatch")
        schema_version = value.get("schema_version")
        expected_fields = set(base_fields)
        if schema_version == STAIR_SCREEN_EVIDENCE_SCHEMA_VERSION:
            expected_fields.add("objective_family_mask_hash")
        elif schema_version != _STAIR_LEGACY_SCREEN_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("Stair screen evidence contract mismatch")
        if set(value) != expected_fields:
            raise ValueError("Stair screen evidence fields mismatch")
        if (
            value.get("acceptance_authority") is not False
        ):
            raise ValueError("Stair screen evidence contract mismatch")
        anchor_scores = value.get("anchor_seed_macro_scores")
        generalization_scores = value.get("generalization_seed_macro_scores")
        if not isinstance(anchor_scores, list) or not isinstance(
            generalization_scores, list
        ):
            raise ValueError("Stair screen seed scores must be lists")
        anchor_stats = value.get("anchor_score_stats")
        generalization_stats = value.get("generalization_score_stats")
        if not isinstance(anchor_stats, Mapping) or not isinstance(
            generalization_stats, Mapping
        ):
            raise ValueError("Stair screen score stats must be objects")
        return cls(
            trial_id=value.get("trial_id"),
            screen_role=value.get("screen_role"),
            train_receipt_checksum=value.get("train_receipt_checksum"),
            runtime_effect_evidence=value.get("runtime_effect_evidence"),
            phase_result_hash=value.get("phase_result_hash"),
            evaluation_plan_hash=value.get("evaluation_plan_hash"),
            task_id=value.get("task_id"),
            subject=ExperimentSubject.from_mapping(value.get("subject")),
            checkpoint_hash=value.get("checkpoint_hash"),
            eval_protocol_hash=value.get("eval_protocol_hash"),
            evaluation_binding_hash=value.get("evaluation_binding_hash"),
            training_binding_hash=value.get("training_binding_hash"),
            capability_profile_hash=value.get("capability_profile_hash"),
            execution_environment_hash=value.get("execution_environment_hash"),
            execution_device=value.get("execution_device"),
            resource_binding_hash=value.get("resource_binding_hash"),
            objective_family_mask_hash=value.get("objective_family_mask_hash"),
            anchor_score_stats=dict(anchor_stats),
            anchor_seed_macro_scores=tuple(anchor_scores),
            generalization_score_stats=dict(generalization_stats),
            generalization_seed_macro_scores=tuple(generalization_scores),
            schema_version=str(schema_version),
        )


@dataclass(frozen=True)
class StairSelectionResult:
    """Deterministic screen decision; never an acceptance decision."""

    contract_hash: str
    screen_evidence_hashes: tuple[str, str]
    guard_reference_evidence_hash: str
    selected_trial_id: str | None
    reason: str
    ranking_confidence: str
    paired_mean_delta: float | None
    paired_standard_error: float | None
    uncertainty_threshold: float | None
    guard_regression_by_trial: Mapping[str, float]
    selection_eligible_trial_ids: tuple[str, ...]
    paired_statistics_status: str = "applicable"
    schema_version: str = STAIR_SELECTION_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_sha256_digest(self.contract_hash, "Stair selection contract_hash")
        if (
            not isinstance(self.screen_evidence_hashes, tuple)
            or len(self.screen_evidence_hashes) != 2
            or len(set(self.screen_evidence_hashes)) != 2
        ):
            raise ValueError("Stair selection requires two unique evidence hashes")
        for value in self.screen_evidence_hashes:
            require_sha256_digest(value, "Stair selection evidence hash")
        require_sha256_digest(
            self.guard_reference_evidence_hash,
            "Stair selection guard reference evidence hash",
        )
        if self.reason not in {
            "public_dev_objective_rank",
            "public_dev_guard_filter",
            "public_dev_runtime_activation_filter",
            "public_dev_guard_tiebreak",
            "inconclusive_exact_tie",
            "no_guard_eligible_candidate",
            "no_runtime_activation_eligible_candidate",
        }:
            raise ValueError("Stair selection reason is invalid")
        if self.ranking_confidence not in {
            "outside_estimated_noise_band",
            "within_estimated_noise_band",
            "deterministic_paired_gap",
            "guard_tiebreak",
            "not_applicable",
        }:
            raise ValueError("Stair selection ranking_confidence is invalid")
        if self.selected_trial_id is None:
            if self.reason not in {
                "inconclusive_exact_tie",
                "no_guard_eligible_candidate",
                "no_runtime_activation_eligible_candidate",
            } or self.ranking_confidence != "not_applicable":
                raise ValueError("Stair no-selection reason mismatch")
        elif (
            not isinstance(self.selected_trial_id, str)
            or not self.selected_trial_id
            or self.reason
            not in {
                "public_dev_objective_rank",
                "public_dev_guard_filter",
                "public_dev_runtime_activation_filter",
                "public_dev_guard_tiebreak",
            }
        ):
            raise ValueError("Stair selected trial identity mismatch")
        if self.schema_version != STAIR_SELECTION_RESULT_SCHEMA_VERSION:
            raise ValueError("Stair selection result schema is unsupported")
        expected_confidences = {
            "public_dev_objective_rank": {
                "outside_estimated_noise_band",
                "within_estimated_noise_band",
                "deterministic_paired_gap",
            },
            "public_dev_guard_filter": {"not_applicable"},
            "public_dev_runtime_activation_filter": {"not_applicable"},
            "public_dev_guard_tiebreak": {"guard_tiebreak"},
            "inconclusive_exact_tie": {"not_applicable"},
            "no_guard_eligible_candidate": {"not_applicable"},
            "no_runtime_activation_eligible_candidate": {"not_applicable"},
        }
        if self.ranking_confidence not in expected_confidences[self.reason]:
            raise ValueError("Stair selection reason/confidence mismatch")
        if not isinstance(self.guard_regression_by_trial, Mapping):
            raise TypeError("Stair guard regression evidence must be an object")
        if len(self.guard_regression_by_trial) != 2:
            raise ValueError("Stair selection requires two guard regressions")
        for trial_id, regression in self.guard_regression_by_trial.items():
            if not isinstance(trial_id, str) or not trial_id:
                raise ValueError("Stair guard regression trial ID is invalid")
            _finite_score(regression, "Stair guard regression")
        if (
            not isinstance(self.selection_eligible_trial_ids, tuple)
            or len(set(self.selection_eligible_trial_ids))
            != len(self.selection_eligible_trial_ids)
            or any(
                not isinstance(trial_id, str) or not trial_id
                for trial_id in self.selection_eligible_trial_ids
            )
            or not set(self.selection_eligible_trial_ids).issubset(
                self.guard_regression_by_trial
            )
        ):
            raise ValueError("Stair selection-eligible trial IDs are invalid")
        if (
            self.selected_trial_id is not None
            and self.selected_trial_id not in self.selection_eligible_trial_ids
        ):
            raise ValueError("Stair selected trial failed the selection precheck")
        if len(self.selection_eligible_trial_ids) == 2:
            if self.paired_statistics_status != "applicable":
                raise ValueError("Stair paired statistics must be applicable")
            self._validate_present_paired_statistics()
        elif (
            self.paired_statistics_status != "not_applicable"
            or self.paired_mean_delta is not None
            or self.paired_standard_error is not None
            or self.uncertainty_threshold is not None
        ):
            raise ValueError(
                "Stair paired statistics require two selection-eligible trials"
            )
        object.__setattr__(
            self,
            "guard_regression_by_trial",
            dict(self.guard_regression_by_trial),
        )

    def _validate_present_paired_statistics(self) -> None:
        _finite_score(self.paired_mean_delta, "Stair paired mean delta")
        for name in ("paired_standard_error", "uncertainty_threshold"):
            value = _finite_score(getattr(self, name), f"Stair {name}")
            if value < 0.0:
                raise ValueError(f"Stair {name} must be non-negative")

    @property
    def selection_hash(self) -> str:
        return canonical_json_hash(self.to_public_dict())

    def to_public_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "contract_hash": self.contract_hash,
            "screen_evidence_hashes": list(self.screen_evidence_hashes),
            "guard_reference_evidence_hash": self.guard_reference_evidence_hash,
            "selected_trial_id": self.selected_trial_id,
            "reason": self.reason,
            "ranking_confidence": self.ranking_confidence,
            "guard_regression_by_trial": dict(self.guard_regression_by_trial),
            "selection_eligible_trial_ids": list(
                self.selection_eligible_trial_ids
            ),
            "evidence_tier": "public_dev_screen_only",
            "acceptance_authority": False,
        }
        paired_statistics: dict[str, Any] = {
            "status": self.paired_statistics_status,
        }
        if self.paired_statistics_status == "applicable":
            paired_statistics.update(
                {
                    "mean_delta": self.paired_mean_delta,
                    "standard_error": self.paired_standard_error,
                    "uncertainty_threshold": self.uncertainty_threshold,
                }
            )
        payload["paired_statistics"] = paired_statistics
        return payload

    @classmethod
    def from_mapping(cls, value: Any) -> "StairSelectionResult":
        common_fields = {
            "contract_hash",
            "screen_evidence_hashes",
            "guard_reference_evidence_hash",
            "selected_trial_id",
            "reason",
            "ranking_confidence",
            "guard_regression_by_trial",
            "selection_eligible_trial_ids",
            "evidence_tier",
            "acceptance_authority",
            "schema_version",
            "paired_statistics",
        }
        if not isinstance(value, Mapping):
            raise ValueError("Stair selection result fields mismatch")
        schema_version = value.get("schema_version")
        if schema_version != STAIR_SELECTION_RESULT_SCHEMA_VERSION or set(
            value
        ) != common_fields:
            raise ValueError("Stair selection result fields mismatch")
        hashes = value.get("screen_evidence_hashes")
        if not isinstance(hashes, list) or len(hashes) != 2:
            raise ValueError("Stair selection evidence hashes are invalid")
        guard_regressions = value.get("guard_regression_by_trial")
        eligible = value.get("selection_eligible_trial_ids")
        if not isinstance(guard_regressions, Mapping) or not isinstance(eligible, list):
            raise ValueError("Stair selection guard evidence is invalid")
        if any(
            not isinstance(trial_id, str) or not trial_id
            for trial_id in guard_regressions
        ):
            raise ValueError("Stair selection guard trial IDs are invalid")
        if (
            value.get("evidence_tier") != "public_dev_screen_only"
            or value.get("acceptance_authority") is not False
        ):
            raise ValueError("Stair selection result contract mismatch")
        paired = value.get("paired_statistics")
        if not isinstance(paired, Mapping):
            raise ValueError("Stair paired statistics are invalid")
        paired_statistics_status = paired.get("status")
        if paired_statistics_status == "applicable":
            if set(paired) != {
                "status",
                "mean_delta",
                "standard_error",
                "uncertainty_threshold",
            }:
                raise ValueError("Stair paired statistics fields mismatch")
            paired_mean_delta = _finite_score(
                paired.get("mean_delta"), "paired mean_delta"
            )
            paired_standard_error = _finite_score(
                paired.get("standard_error"), "paired standard_error"
            )
            uncertainty_threshold = _finite_score(
                paired.get("uncertainty_threshold"),
                "paired uncertainty_threshold",
            )
        elif paired_statistics_status == "not_applicable":
            if set(paired) != {"status"}:
                raise ValueError("Stair inapplicable paired statistics mismatch")
            paired_mean_delta = None
            paired_standard_error = None
            uncertainty_threshold = None
        else:
            raise ValueError("Stair paired statistics status is invalid")
        return cls(
            contract_hash=require_sha256_digest(
                value.get("contract_hash"), "contract_hash"
            ),
            screen_evidence_hashes=(
                require_sha256_digest(hashes[0], "screen evidence hash"),
                require_sha256_digest(hashes[1], "screen evidence hash"),
            ),
            guard_reference_evidence_hash=require_sha256_digest(
                value.get("guard_reference_evidence_hash"),
                "guard_reference_evidence_hash",
            ),
            selected_trial_id=value.get("selected_trial_id"),
            reason=value.get("reason"),
            ranking_confidence=value.get("ranking_confidence"),
            paired_mean_delta=paired_mean_delta,
            paired_standard_error=paired_standard_error,
            uncertainty_threshold=uncertainty_threshold,
            guard_regression_by_trial={
                trial_id: _finite_score(regression, "guard_regression")
                for trial_id, regression in guard_regressions.items()
            },
            selection_eligible_trial_ids=tuple(eligible),
            paired_statistics_status=paired_statistics_status,
            schema_version=schema_version,
        )


def build_stair_public_dev_screen_evidence(
    *,
    config: StairClimbingTaskConfig,
    trial_id: str,
    phase_result: EvaluationPhaseResult,
    screen_role: str,
    train_result: TrainJobResult | None = None,
    train_receipt_checksum: str | None = None,
    mechanism_operation_delta: Mapping[str, Any] | None = None,
    objective_family_mask: Mapping[str, Any] | None = None,
) -> StairPublicDevScreenEvidence:
    """Project one validated public-dev phase into selector-safe evidence."""

    if phase_result.phase is not EvaluationPhase.PUBLIC_DEV:
        raise ValueError("Stair selector accepts public-dev phase results only")
    if screen_role == "candidate":
        if not isinstance(train_result, TrainJobResult):
            raise TypeError("Stair candidate screen requires a train result")
        require_sha256_checksum(
            train_receipt_checksum,
            "Stair candidate screen train receipt checksum",
        )
        if mechanism_operation_delta is None:
            raise ValueError(
                "Stair candidate screen requires its verifier-owned operation delta"
            )
        expected_train_identity = {
            "subject": phase_result.subject,
            "subject_manifest_hash": phase_result.subject_manifest_hash,
            "program_source_checksum": phase_result.program_source_checksum,
            "training_binding_hash": phase_result.training_binding_hash,
            "capability_profile_hash": phase_result.capability_profile_hash,
            "checkpoint_hash": phase_result.checkpoint_hash,
            "execution_device": phase_result.execution_device,
        }
        observed_train_identity = {
            "subject": train_result.subject,
            "subject_manifest_hash": train_result.subject_manifest_hash,
            "program_source_checksum": train_result.program_source_checksum,
            "training_binding_hash": train_result.training_binding_hash,
            "capability_profile_hash": train_result.capability_profile_hash,
            "checkpoint_hash": train_result.checkpoint.checksum,
            "execution_device": train_result.execution_device,
        }
        if observed_train_identity != expected_train_identity:
            raise ValueError("Stair candidate screen train/eval lineage mismatch")
        runtime_effect_evidence = build_stair_candidate_runtime_effect_evidence(
            train_result.summary,
            mechanism_operation_delta=mechanism_operation_delta,
        )
    elif screen_role == "guard_reference":
        if (
            train_result is not None
            or train_receipt_checksum is not None
            or mechanism_operation_delta is not None
        ):
            raise ValueError("Stair guard screen received candidate train evidence")
        runtime_effect_evidence = None
    else:
        raise ValueError("Stair screen role is invalid")
    raw_evidence = phase_result.task_evidence
    if (
        set(raw_evidence)
        != {"schema_version", "phase", "execution_identity", "summary"}
        or raw_evidence.get("schema_version")
        != STAIR_PHASE_EVIDENCE_SCHEMA_VERSION
        or raw_evidence.get("phase") != EvaluationPhase.PUBLIC_DEV.value
    ):
        raise ValueError("Stair public-dev phase evidence contract mismatch")
    execution_identity = raw_evidence.get("execution_identity")
    if not isinstance(execution_identity, Mapping) or (
        execution_identity.get("checkpoint_hash") != phase_result.checkpoint_hash
        or execution_identity.get("seed_namespace")
        != STAIR_FORMAL_EVAL_NAMESPACE
        or execution_identity.get("seed_cohort_hash")
        != config.seed_resolver().namespace_cohort_hash(
            STAIR_FORMAL_EVAL_NAMESPACE
        )
        or execution_identity.get("splits")
        != [EvaluationPhase.PUBLIC_DEV.value]
        or execution_identity.get("execution_device")
        != phase_result.execution_device
        or execution_identity.get("environment_source_preflight")
        != dict(phase_result.environment_source_preflight)
        or execution_identity.get("backend_execution_environment_verification")
        != dict(phase_result.backend_execution_environment_verification)
    ):
        raise ValueError("Stair public-dev execution identity mismatch")
    raw_summary = raw_evidence.get("summary")
    if config.is_unified_protocol:
        if objective_family_mask is None:
            raise ValueError("unified Stair screen requires the reference family mask")
        objective_family_mask = validate_objective_family_mask(
            suite=config.ordered_evaluation_suite,
            mask=objective_family_mask,
        )
        canonical = validate_unified_stair_summary(
            suite=config.ordered_evaluation_suite,
            semantics=config.evaluation_semantics,
            split=EvaluationPhase.PUBLIC_DEV.value,
            summary=raw_summary,
        )
        projection = build_unified_score_projection(
            canonical,
            objective_family_mask=objective_family_mask,
        )
        generalization = projection["j_primary"]
        guard_vectors = [
            projection[guard.name]["seed_macro_scores"]
            for guard in config.acceptance_protocol.named_guards
        ]
        anchor_scores = tuple(
            min(float(values[index]) for values in guard_vectors)
            for index in range(config.public_seed_count)
        )
        anchor = {
            "score_stats": _screen_score_stats(
                anchor_scores,
                eval_protocol_hash=config.evaluation_protocol().protocol_hash,
            ),
            "seed_macro_scores": list(anchor_scores),
        }
    else:
        if objective_family_mask is not None:
            raise ValueError("legacy Stair screen cannot carry a family mask")
        if not isinstance(raw_summary, Mapping) or set(raw_summary) != {
            "anchor",
            "generalization",
        }:
            raise ValueError("Stair public-dev summary fields mismatch")
        anchor = validate_stair_split_summary(
            config.protocol_v1,
            semantics=config.evaluation_semantics,
            evaluation_protocol=config.evaluation_protocol(),
            split=EvaluationPhase.PUBLIC_DEV.value,
            summary=raw_summary["anchor"],
        )
        generalization = validate_stair_generalization_summary(
            profile=config.evaluation_profile,
            semantics=config.evaluation_semantics,
            evaluation_protocol=config.evaluation_protocol(),
            split=EvaluationPhase.PUBLIC_DEV.value,
            summary=raw_summary["generalization"],
        )
    anchor_scores = tuple(float(value) for value in anchor["seed_macro_scores"])
    scores = tuple(float(value) for value in generalization["seed_macro_scores"])
    if (
        len(anchor_scores) != config.public_seed_count
        or len(scores) != config.public_seed_count
    ):
        raise ValueError("Stair screen public-dev seed count mismatch")
    return StairPublicDevScreenEvidence(
        trial_id=trial_id,
        screen_role=screen_role,
        train_receipt_checksum=train_receipt_checksum,
        runtime_effect_evidence=runtime_effect_evidence,
        phase_result_hash=canonical_json_hash(phase_result.to_private_dict()),
        evaluation_plan_hash=phase_result.evaluation_plan_hash,
        task_id=phase_result.task_id,
        subject=phase_result.subject,
        checkpoint_hash=phase_result.checkpoint_hash,
        eval_protocol_hash=phase_result.eval_protocol_hash,
        evaluation_binding_hash=phase_result.evaluation_binding_hash,
        training_binding_hash=phase_result.training_binding_hash,
        capability_profile_hash=phase_result.capability_profile_hash,
        execution_environment_hash=canonical_json_hash(
            {
                "environment_source_preflight": dict(
                    phase_result.environment_source_preflight
                ),
                "backend_execution_environment_verification": dict(
                    phase_result.backend_execution_environment_verification
                ),
            }
        ),
        execution_device=phase_result.execution_device,
        resource_binding_hash=phase_result.resource_binding_hash,
        objective_family_mask_hash=(
            str(objective_family_mask["objective_family_mask_hash"])
            if objective_family_mask is not None
            else None
        ),
        anchor_score_stats=dict(anchor["score_stats"]),
        anchor_seed_macro_scores=anchor_scores,
        generalization_score_stats=dict(generalization["score_stats"]),
        generalization_seed_macro_scores=scores,
    )


def stair_selection_contract(config: StairClimbingTaskConfig) -> dict[str, Any]:
    multiplier = config.acceptance_protocol.paired_standard_error_multiplier
    guard_max_regression = (
        min(guard.max_regression for guard in config.acceptance_protocol.named_guards)
        if config.acceptance_protocol.uses_named_guards
        else config.acceptance_protocol.guard_max_regression
    )
    if multiplier is None:
        raise ValueError("Stair selector requires a paired-noise acceptance protocol")
    if guard_max_regression is None:
        raise ValueError("Stair selector requires a guarded acceptance protocol")
    return {
        "schema_version": STAIR_SELECTION_CONTRACT_SCHEMA_VERSION,
        "task_id": STAIR_TASK_ID,
        "task_protocol_hash": config.task_protocol_hash,
        "acceptance_protocol_hash": config.acceptance_protocol.protocol_hash,
        "rule": (
            "named_guard_min_and_runtime_eligible_public_dev_rank_v1"
            if config.is_unified_protocol
            else "guard_and_runtime_eligible_public_dev_generalization_rank_v4"
        ),
        "primary_score_source": (
            "public_dev.unified_score.j_primary.seed_macro_scores"
            if config.is_unified_protocol
            else "public_dev.generalization.seed_macro_scores"
        ),
        "guard_score_source": (
            "public_dev.minimum_of_named_guard_seed_scores"
            if config.is_unified_protocol
            else "public_dev.anchor.seed_macro_scores"
        ),
        "guard_rule": "parent_relative_hard_regression_cap_only",
        "runtime_effect_rule": (
            "candidate_reward_operation_runtime_effect.status_must_pass"
        ),
        "guard_max_regression": float(guard_max_regression),
        "paired_standard_error_multiplier": float(multiplier),
        "screen_seed_count": config.public_seed_count,
        "candidate_count": 2,
        "outcomes": ["selected", "no_selection"],
        "reason_codes": [
            "public_dev_objective_rank",
            "public_dev_guard_filter",
            "public_dev_runtime_activation_filter",
            "public_dev_guard_tiebreak",
            "inconclusive_exact_tie",
            "no_guard_eligible_candidate",
            "no_runtime_activation_eligible_candidate",
        ],
        "ranking_policy": (
            "always_rank_selection_eligible_candidates; paired noise is evidence only"
        ),
        "acceptance_headroom_role": "not_evaluated_by_selector",
        "allowed_execution_differences": [
            "trial_id",
            "phase_result_hash",
            "evaluation_plan_hash",
            "execution_device",
            "resource_binding_hash",
            "subject",
            "checkpoint_hash",
            "train_receipt_checksum",
            "runtime_effect_evidence",
        ],
        "paired_statistics_policy": (
            "emit only when both sibling trials are selection eligible; otherwise "
            "status is not_applicable and numeric statistics are absent"
        ),
        "acceptance_authority": False,
    }


def _screen_score_stats(
    values: Sequence[float],
    *,
    eval_protocol_hash: str,
) -> dict[str, Any]:
    canonical = [float(value) for value in values]
    mean = statistics.fmean(canonical)
    std = statistics.stdev(canonical) if len(canonical) > 1 else 0.0
    return {
        "count": len(canonical),
        "mean": mean,
        "std": std,
        "standard_error": std / (len(canonical) ** 0.5),
        "values": canonical,
        "eval_protocol_hash": eval_protocol_hash,
    }


def select_stair_public_dev_winner(
    *,
    config: StairClimbingTaskConfig,
    screens: Sequence[StairPublicDevScreenEvidence],
    guard_reference: StairPublicDevScreenEvidence,
) -> StairSelectionResult:
    return _select_stair_public_dev_winner(
        config=config,
        screens=screens,
        guard_reference=guard_reference,
    )


def replay_stair_public_dev_winner(
    *,
    config: StairClimbingTaskConfig,
    screens: Sequence[StairPublicDevScreenEvidence],
    guard_reference: StairPublicDevScreenEvidence,
    result_schema_version: str,
) -> StairSelectionResult:
    """Replay the current selector against its sealed public inputs."""

    if result_schema_version != STAIR_SELECTION_RESULT_SCHEMA_VERSION:
        raise ValueError("Stair selection replay schema is unsupported")
    return _select_stair_public_dev_winner(
        config=config,
        screens=screens,
        guard_reference=guard_reference,
    )


def _select_stair_public_dev_winner(
    *,
    config: StairClimbingTaskConfig,
    screens: Sequence[StairPublicDevScreenEvidence],
    guard_reference: StairPublicDevScreenEvidence,
) -> StairSelectionResult:
    """Allocate formal evaluation to the best fully eligible public-dev trial."""

    if len(screens) != 2 or any(
        not isinstance(item, StairPublicDevScreenEvidence) for item in screens
    ):
        raise ValueError("Stair selection requires exactly two screen evidences")
    ordered = tuple(sorted(screens, key=lambda item: item.trial_id))
    if any(item.screen_role != "candidate" for item in ordered):
        raise ValueError("Stair selection screens must identify candidates")
    if ordered[0].trial_id == ordered[1].trial_id:
        raise ValueError("Stair selection trial IDs must be unique")
    if ordered[0].comparison_domain != ordered[1].comparison_domain:
        raise ValueError("Stair screen evidences are not comparable")
    if not isinstance(guard_reference, StairPublicDevScreenEvidence):
        raise TypeError("Stair selection requires a public-dev guard reference")
    if guard_reference.screen_role != "guard_reference":
        raise ValueError("Stair selection guard must identify the frozen parent")
    if guard_reference.comparison_domain != ordered[0].comparison_domain:
        raise ValueError("Stair guard reference is not comparable")
    if (
        guard_reference.evidence_hash
        in {item.evidence_hash for item in ordered}
        or guard_reference.subject in {item.subject for item in ordered}
    ):
        raise ValueError("Stair guard reference must identify the frozen parent")
    for evidence in (*ordered, guard_reference):
        if (
            len(evidence.anchor_seed_macro_scores) != config.public_seed_count
            or len(evidence.generalization_seed_macro_scores)
            != config.public_seed_count
        ):
            raise ValueError("Stair screen paired seed evidence is incomplete")
    left_scores = ordered[0].generalization_seed_macro_scores
    right_scores = ordered[1].generalization_seed_macro_scores
    if len(left_scores) != len(right_scores) or len(left_scores) != config.public_seed_count:
        raise ValueError("Stair screen paired seed evidence is incomplete")

    paired = tuple(left - right for left, right in zip(left_scores, right_scores))
    mean_delta = statistics.fmean(paired)
    standard_error = (
        statistics.stdev(paired) / math.sqrt(len(paired))
        if len(paired) > 1
        else 0.0
    )
    contract = stair_selection_contract(config)
    threshold = float(contract["paired_standard_error_multiplier"]) * standard_error
    magnitude = abs(mean_delta)
    outside_noise = magnitude > threshold and not math.isclose(
        magnitude, threshold, rel_tol=1.0e-12, abs_tol=1.0e-12
    )
    reference_guard = statistics.fmean(guard_reference.anchor_seed_macro_scores)
    hard_guard_cap = float(contract["guard_max_regression"])
    guard_regressions = {
        item.trial_id: reference_guard
        - statistics.fmean(item.anchor_seed_macro_scores)
        for item in ordered
    }
    guard_eligible = tuple(
        item.trial_id
        for item in ordered
        if guard_regressions[item.trial_id] <= hard_guard_cap
    )
    runtime_eligible = tuple(
        item.trial_id
        for item in ordered
        if item.runtime_effect_evidence is not None
        and item.runtime_effect_evidence["status"] == "pass"
    )
    eligible = tuple(
        item.trial_id
        for item in ordered
        if item.trial_id in guard_eligible and item.trial_id in runtime_eligible
    )
    selected = None
    reason = (
        "no_guard_eligible_candidate"
        if not guard_eligible
        else "no_runtime_activation_eligible_candidate"
    )
    confidence = "not_applicable"
    if len(eligible) == 1:
        selected = eligible[0]
        other = next(item.trial_id for item in ordered if item.trial_id != selected)
        reason = (
            "public_dev_guard_filter"
            if other not in guard_eligible
            else "public_dev_runtime_activation_filter"
        )
    elif len(eligible) == 2 and not math.isclose(
        mean_delta, 0.0, rel_tol=1.0e-12, abs_tol=1.0e-12
    ):
        selected = ordered[0].trial_id if mean_delta > 0.0 else ordered[1].trial_id
        reason = "public_dev_objective_rank"
        confidence = (
            "deterministic_paired_gap"
            if math.isclose(standard_error, 0.0, rel_tol=0.0, abs_tol=1.0e-15)
            else (
                "outside_estimated_noise_band"
                if outside_noise
                else "within_estimated_noise_band"
            )
        )
    elif len(eligible) == 2:
        guard_delta = statistics.fmean(ordered[0].anchor_seed_macro_scores) - (
            statistics.fmean(ordered[1].anchor_seed_macro_scores)
        )
        if not math.isclose(
            guard_delta, 0.0, rel_tol=1.0e-12, abs_tol=1.0e-12
        ):
            selected = (
                ordered[0].trial_id if guard_delta > 0.0 else ordered[1].trial_id
            )
            reason = "public_dev_guard_tiebreak"
            confidence = "guard_tiebreak"
        else:
            reason = "inconclusive_exact_tie"
    return StairSelectionResult(
        contract_hash=canonical_json_hash(contract),
        screen_evidence_hashes=tuple(item.evidence_hash for item in ordered),
        guard_reference_evidence_hash=guard_reference.evidence_hash,
        selected_trial_id=selected,
        reason=reason,
        ranking_confidence=confidence,
        paired_mean_delta=(mean_delta if len(eligible) == 2 else None),
        paired_standard_error=(standard_error if len(eligible) == 2 else None),
        uncertainty_threshold=(threshold if len(eligible) == 2 else None),
        guard_regression_by_trial=guard_regressions,
        selection_eligible_trial_ids=eligible,
        paired_statistics_status=(
            "applicable" if len(eligible) == 2 else "not_applicable"
        ),
        schema_version=STAIR_SELECTION_RESULT_SCHEMA_VERSION,
    )


def _finite_score(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field_name} must be finite")
    return float(value)


__all__ = [
    "STAIR_SCREEN_EVIDENCE_SCHEMA_VERSION",
    "STAIR_SELECTION_CONTRACT_SCHEMA_VERSION",
    "STAIR_SELECTION_RESULT_SCHEMA_VERSION",
    "StairPublicDevScreenEvidence",
    "StairSelectionResult",
    "build_stair_public_dev_screen_evidence",
    "replay_stair_public_dev_winner",
    "select_stair_public_dev_winner",
    "stair_selection_contract",
]
