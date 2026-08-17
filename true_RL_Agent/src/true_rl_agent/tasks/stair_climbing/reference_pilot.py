"""Task-owned R0 pilot capability evidence and strict validation."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from true_rl_agent.evaluation.binding import EvaluationBindingV1, canonical_json_hash
from true_rl_agent.evaluation.protocol import EvaluationProtocol
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.episode_evidence import StairEpisodeOutcome
from true_rl_agent.tasks.stair_climbing.evaluation_contract import (
    build_stair_split_summary,
    validate_stair_split_summary,
)
from true_rl_agent.tasks.stair_climbing.phase0_contract import StairProtocolV1
from true_rl_agent.tasks.stair_climbing.protocol_v2 import StairProtocolV2Draft
from true_rl_agent.tasks.stair_climbing.seed_policy import (
    STAIR_FORMAL_EVAL_NAMESPACE,
    STAIR_FORMAL_TRAIN_NAMESPACE,
    STAIR_PILOT_EVAL_NAMESPACE,
    STAIR_PILOT_TRAIN_NAMESPACE,
    STAIR_TRAIN_SEED_BASE,
    StairPrivateTrainingSeedResolver,
)
from true_rl_agent.training_binding import (
    TrainingBinding,
    fixed_training_seed_resolver_hash,
    initialization_seed_schedule_hash,
)

STAIR_REFERENCE_CAPABILITY_EVIDENCE_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.reference_capability_evidence.v1"
)
STAIR_PILOT_TRAIN_BASE_SEED = STAIR_TRAIN_SEED_BASE

_EVIDENCE_FIELDS = {
    "schema_version",
    "draft_protocol_hash",
    "base_protocol_hash",
    "split",
    "split_summary",
    "seed_measurements",
    "gate",
    "evidence_hash",
}
_SEED_MEASUREMENT_FIELDS = {
    "seed_index",
    "low_step_bin_ids",
    "success_rate_by_bin",
    "mean_max_progress_m_by_bin",
    "low_step_macro_success_rate",
    "low_step_macro_max_progress_m",
}
_GATE_FIELDS = {
    "schema_version",
    "status",
    "predicate",
    "minimum_macro_mean_success_rate",
    "minimum_macro_mean_max_progress_m",
    "observed_macro_mean_success_rate",
    "observed_macro_mean_max_progress_m",
    "success_floor_met",
    "progress_floor_met",
    "threshold_relaxation_allowed",
    "reason",
}


def build_stair_pilot_config(
    protocol_v2_path: Path,
) -> StairClimbingTaskConfig:
    """Resolve the single fixed-budget pilot and its private training seed."""

    base = StairClimbingTaskConfig.defaults(protocol_v2_path)
    denominator = base.train_num_envs * base.train_steps_per_iteration
    if base.budget_steps % denominator:
        raise ValueError("stair pilot budget is not an integral iteration count")
    resolver = StairPrivateTrainingSeedResolver(
        base_seed=STAIR_PILOT_TRAIN_BASE_SEED,
        task_protocol_hash=base.protocol_v2.protocol_hash,
    )
    seed_count = base.protocol_v2.pilot_training_seed_count
    if seed_count != 1:
        raise ValueError("stair fixed-budget confirmation requires one pilot seed")
    resolver.assert_namespaces_disjoint(pilot_seed_count=seed_count)
    return replace(
        base,
        train_seed=resolver.derive_seed(
            namespace=STAIR_PILOT_TRAIN_NAMESPACE,
            training_seed_index=0,
        ),
    )


def build_stair_pilot_training_binding(
    config: StairClimbingTaskConfig,
) -> TrainingBinding:
    _assert_pilot_config_seed(config)
    agent = config.training_agent_configuration()
    return TrainingBinding(
        initialization_mode=PolicyInitialization.random().kind,
        initialization_checkpoint_hash=None,
        policy_architecture_hash=str(agent.policy_architecture_hash),
        agent_configuration_hash=str(agent.agent_configuration_hash),
        initialization_seed_schedule_hash=initialization_seed_schedule_hash(
            config.train_seed
        ),
        optimizer_training_profile_hash=str(
            agent.optimizer_training_profile_hash
        ),
        budget_steps=config.budget_steps,
        private_training_seed_resolver_hash=fixed_training_seed_resolver_hash(
            config.train_seed
        ),
        capability_profile_hash=config.protocol_v2.capability_profile.profile_hash,
    )


def build_stair_pilot_evaluation_binding(
    config: StairClimbingTaskConfig,
) -> EvaluationBindingV1:
    _assert_pilot_config_seed(config)
    resolver = config.seed_resolver()
    return EvaluationBindingV1(
        eval_protocol_hash=config.evaluation_protocol().protocol_hash,
        acceptance_protocol_hash=(
            config.acceptance_protocol.protocol_hash
        ),
        task_protocol_hash=config.protocol_v2.protocol_hash,
        seed_cohort_hash=resolver.namespace_cohort_hash(
            STAIR_PILOT_EVAL_NAMESPACE
        ),
    )


def assert_stair_pilot_formal_bindings_disjoint(
    config: StairClimbingTaskConfig,
) -> None:
    _assert_pilot_config_seed(config)
    resolver = StairPrivateTrainingSeedResolver(
        base_seed=STAIR_PILOT_TRAIN_BASE_SEED,
        task_protocol_hash=config.protocol_v2.protocol_hash,
    )
    seed_count = config.protocol_v2.pilot_training_seed_count
    resolver.assert_namespaces_disjoint(pilot_seed_count=seed_count)
    formal_seed = resolver.derive_seed(namespace=STAIR_FORMAL_TRAIN_NAMESPACE)
    formal_config = replace(config, train_seed=formal_seed)
    agent = formal_config.training_agent_configuration()
    formal_training = TrainingBinding(
        initialization_mode="random",
        initialization_checkpoint_hash=None,
        policy_architecture_hash=str(agent.policy_architecture_hash),
        agent_configuration_hash=str(agent.agent_configuration_hash),
        initialization_seed_schedule_hash=initialization_seed_schedule_hash(
            formal_seed
        ),
        optimizer_training_profile_hash=str(
            agent.optimizer_training_profile_hash
        ),
        budget_steps=formal_config.budget_steps,
        private_training_seed_resolver_hash=fixed_training_seed_resolver_hash(
            formal_seed
        ),
        capability_profile_hash=(
            formal_config.protocol_v2.capability_profile.profile_hash
        ),
    )
    for training_seed_index in range(seed_count):
        pilot_config = replace(
            config,
            train_seed=resolver.derive_seed(
                namespace=STAIR_PILOT_TRAIN_NAMESPACE,
                training_seed_index=training_seed_index,
            ),
        )
        pilot_training = build_stair_pilot_training_binding(pilot_config)
        if pilot_training.binding_hash == formal_training.binding_hash:
            raise ValueError("stair pilot and formal TrainingBinding overlap")

    eval_resolver = config.seed_resolver()
    pilot_eval = build_stair_pilot_evaluation_binding(config)
    formal_eval = EvaluationBindingV1(
        eval_protocol_hash=config.evaluation_protocol().protocol_hash,
        acceptance_protocol_hash=(
            config.acceptance_protocol.protocol_hash
        ),
        task_protocol_hash=config.protocol_v2.protocol_hash,
        seed_cohort_hash=eval_resolver.namespace_cohort_hash(
            STAIR_FORMAL_EVAL_NAMESPACE
        ),
    )
    if pilot_eval.binding_hash == formal_eval.binding_hash:
        raise ValueError("stair pilot and formal EvaluationBinding overlap")


def stair_pilot_training_seed_domain_hash(
    config: StairClimbingTaskConfig,
) -> str:
    _assert_pilot_config_seed(config)
    resolver = StairPrivateTrainingSeedResolver(
        base_seed=STAIR_PILOT_TRAIN_BASE_SEED,
        task_protocol_hash=config.protocol_v2.protocol_hash,
    )
    return resolver.namespace_cohort_hash(
        namespace=STAIR_PILOT_TRAIN_NAMESPACE,
        seed_count=config.protocol_v2.pilot_training_seed_count,
    )


def stair_pilot_training_seed_index(config: StairClimbingTaskConfig) -> int:
    if not isinstance(config, StairClimbingTaskConfig):
        raise TypeError("stair pilot config must be StairClimbingTaskConfig")
    resolver = StairPrivateTrainingSeedResolver(
        base_seed=STAIR_PILOT_TRAIN_BASE_SEED,
        task_protocol_hash=config.protocol_v2.protocol_hash,
    )
    seed_count = config.protocol_v2.pilot_training_seed_count
    resolver.assert_namespaces_disjoint(pilot_seed_count=seed_count)
    matches = [
        index
        for index in range(seed_count)
        if config.train_seed
        == resolver.derive_seed(
            namespace=STAIR_PILOT_TRAIN_NAMESPACE,
            training_seed_index=index,
        )
    ]
    if len(matches) != 1:
        raise ValueError("stair pilot config train_seed is outside the pilot cohort")
    return matches[0]


def _assert_pilot_config_seed(config: StairClimbingTaskConfig) -> None:
    stair_pilot_training_seed_index(config)


def build_stair_reference_capability_evidence(
    protocol: StairProtocolV1,
    *,
    protocol_v2: StairProtocolV2Draft,
    evaluation_protocol: EvaluationProtocol,
    per_seed_outcomes: Sequence[Sequence[StairEpisodeOutcome]],
) -> dict[str, Any]:
    """Build the frozen low-step gate from closed recorder outcomes."""

    _require_contracts(protocol, protocol_v2, evaluation_protocol)
    gate_contract = _gate_contract(protocol_v2)
    split = str(gate_contract["source_split"])
    outcomes = [tuple(items) for items in per_seed_outcomes]
    split_summary = build_stair_split_summary(
        protocol,
        semantics=None,
        evaluation_protocol=evaluation_protocol,
        split=split,
        per_seed_outcomes=outcomes,
    )
    low_bins = tuple(str(item) for item in gate_contract["low_step_bin_ids"])
    measurements = [
        _seed_measurement(
            seed_index=index,
            outcomes=seed_outcomes,
            low_bins=low_bins,
        )
        for index, seed_outcomes in enumerate(outcomes, start=1)
    ]
    gate = _build_gate(gate_contract, measurements)
    evidence = {
        "schema_version": STAIR_REFERENCE_CAPABILITY_EVIDENCE_SCHEMA_VERSION,
        "draft_protocol_hash": protocol_v2.protocol_hash,
        "base_protocol_hash": protocol.protocol_hash,
        "split": split,
        "split_summary": split_summary,
        "seed_measurements": measurements,
        "gate": gate,
    }
    evidence["evidence_hash"] = canonical_json_hash(evidence)
    return validate_stair_reference_capability_evidence(
        protocol,
        protocol_v2=protocol_v2,
        evaluation_protocol=evaluation_protocol,
        evidence=evidence,
    )


def validate_stair_reference_capability_evidence(
    protocol: StairProtocolV1,
    *,
    protocol_v2: StairProtocolV2Draft,
    evaluation_protocol: EvaluationProtocol,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute every gate aggregate from one strict pilot evidence object."""

    _require_contracts(protocol, protocol_v2, evaluation_protocol)
    if not isinstance(evidence, Mapping) or set(evidence) != _EVIDENCE_FIELDS:
        raise ValueError("stair reference capability evidence fields mismatch")
    if evidence.get("schema_version") != (
        STAIR_REFERENCE_CAPABILITY_EVIDENCE_SCHEMA_VERSION
    ):
        raise ValueError("stair reference capability evidence schema mismatch")
    if evidence.get("draft_protocol_hash") != protocol_v2.protocol_hash:
        raise ValueError("stair reference capability draft hash mismatch")
    if evidence.get("base_protocol_hash") != protocol.protocol_hash:
        raise ValueError("stair reference capability base hash mismatch")

    gate_contract = _gate_contract(protocol_v2)
    split = str(gate_contract["source_split"])
    if evidence.get("split") != split:
        raise ValueError("stair reference capability split mismatch")
    canonical_summary = validate_stair_split_summary(
        protocol,
        semantics=None,
        evaluation_protocol=evaluation_protocol,
        split=split,
        summary=_mapping(evidence.get("split_summary"), "split_summary"),
    )
    if canonical_summary["seed_count"] != gate_contract["seed_count"]:
        raise ValueError("stair reference capability seed count mismatch")

    low_bins = tuple(str(item) for item in gate_contract["low_step_bin_ids"])
    raw_measurements = evidence.get("seed_measurements")
    if not isinstance(raw_measurements, list) or len(raw_measurements) != int(
        gate_contract["seed_count"]
    ):
        raise ValueError("stair reference capability measurements are incomplete")
    measurements = [
        _validate_seed_measurement(
            raw,
            expected_index=index,
            low_bins=low_bins,
            split_seed_summary=canonical_summary["seed_summaries"][index - 1],
        )
        for index, raw in enumerate(raw_measurements, start=1)
    ]
    _validate_progress_projection(
        measurements,
        canonical_summary=canonical_summary,
        low_bins=low_bins,
    )

    expected_gate = _build_gate(gate_contract, measurements)
    if evidence.get("gate") != expected_gate:
        raise ValueError("stair reference capability gate was not recomputed")
    declared_hash = evidence.get("evidence_hash")
    if not isinstance(declared_hash, str) or declared_hash != canonical_json_hash(
        {key: value for key, value in evidence.items() if key != "evidence_hash"}
    ):
        raise ValueError("stair reference capability evidence hash mismatch")
    return {
        "schema_version": STAIR_REFERENCE_CAPABILITY_EVIDENCE_SCHEMA_VERSION,
        "draft_protocol_hash": protocol_v2.protocol_hash,
        "base_protocol_hash": protocol.protocol_hash,
        "split": split,
        "split_summary": canonical_summary,
        "seed_measurements": measurements,
        "gate": expected_gate,
        "evidence_hash": declared_hash,
    }


def _seed_measurement(
    *,
    seed_index: int,
    outcomes: Sequence[StairEpisodeOutcome],
    low_bins: tuple[str, ...],
) -> dict[str, Any]:
    success_rates: dict[str, float] = {}
    progress_means: dict[str, float] = {}
    for bin_id in low_bins:
        selected = [item for item in outcomes if item.bin_id == bin_id]
        if not selected:
            raise ValueError(
                f"stair reference capability seed {seed_index} misses {bin_id}"
            )
        success_rates[bin_id] = statistics.fmean(
            float(item.success) for item in selected
        )
        progress_means[bin_id] = statistics.fmean(
            float(item.max_progress_m) for item in selected
        )
    return {
        "seed_index": seed_index,
        "low_step_bin_ids": list(low_bins),
        "success_rate_by_bin": success_rates,
        "mean_max_progress_m_by_bin": progress_means,
        "low_step_macro_success_rate": statistics.fmean(success_rates.values()),
        "low_step_macro_max_progress_m": statistics.fmean(
            progress_means.values()
        ),
    }


def _validate_seed_measurement(
    value: Any,
    *,
    expected_index: int,
    low_bins: tuple[str, ...],
    split_seed_summary: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _mapping(value, "seed_measurement")
    if set(raw) != _SEED_MEASUREMENT_FIELDS:
        raise ValueError("stair reference seed measurement fields mismatch")
    if raw.get("seed_index") != expected_index:
        raise ValueError("stair reference seed measurement index mismatch")
    if raw.get("low_step_bin_ids") != list(low_bins):
        raise ValueError("stair reference seed measurement bin order mismatch")

    success = _finite_bin_mapping(
        raw.get("success_rate_by_bin"),
        low_bins=low_bins,
        field_name="stair reference success rate",
        minimum=0.0,
        maximum=1.0,
    )
    declared_success = _mapping(
        split_seed_summary.get("bin_success_rates"),
        "split seed success rates",
    )
    if any(
        not math.isclose(
            success[bin_id],
            _finite(declared_success.get(bin_id), "split seed success rate"),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        for bin_id in low_bins
    ):
        raise ValueError("stair reference success measurement differs from split")

    progress = _finite_bin_mapping(
        raw.get("mean_max_progress_m_by_bin"),
        low_bins=low_bins,
        field_name="stair reference progress",
        minimum=0.0,
        maximum=None,
    )
    success_macro = statistics.fmean(success.values())
    progress_macro = statistics.fmean(progress.values())
    if not math.isclose(
        _finite(raw.get("low_step_macro_success_rate"), "success macro"),
        success_macro,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("stair reference success macro mismatch")
    if not math.isclose(
        _finite(raw.get("low_step_macro_max_progress_m"), "progress macro"),
        progress_macro,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("stair reference progress macro mismatch")
    return {
        "seed_index": expected_index,
        "low_step_bin_ids": list(low_bins),
        "success_rate_by_bin": success,
        "mean_max_progress_m_by_bin": progress,
        "low_step_macro_success_rate": success_macro,
        "low_step_macro_max_progress_m": progress_macro,
    }


def _validate_progress_projection(
    measurements: Sequence[Mapping[str, Any]],
    *,
    canonical_summary: Mapping[str, Any],
    low_bins: tuple[str, ...],
) -> None:
    progress_summary = _mapping(
        canonical_summary.get("progress_summary"),
        "split progress summary",
    )
    by_bin = _mapping(
        progress_summary.get("mean_max_progress_m_by_bin"),
        "split progress by bin",
    )
    for bin_id in low_bins:
        projected = statistics.fmean(
            _finite(
                _mapping(
                    item.get("mean_max_progress_m_by_bin"),
                    "seed progress by bin",
                ).get(bin_id),
                "seed progress",
            )
            for item in measurements
        )
        if not math.isclose(
            projected,
            _finite(by_bin.get(bin_id), "split progress by bin"),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError("stair reference progress measurement differs from split")


def _build_gate(
    contract: Mapping[str, Any],
    measurements: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    observed_success = statistics.fmean(
        _finite(item.get("low_step_macro_success_rate"), "success macro")
        for item in measurements
    )
    observed_progress = statistics.fmean(
        _finite(item.get("low_step_macro_max_progress_m"), "progress macro")
        for item in measurements
    )
    success_floor = _finite(
        contract["minimum_macro_mean_success_rate"],
        "success floor",
    )
    progress_floor = _finite(
        contract["minimum_macro_mean_max_progress_m"],
        "progress floor",
    )
    success_met = observed_success >= success_floor
    progress_met = observed_progress >= progress_floor
    passed = success_met and progress_met
    return {
        "schema_version": str(contract["schema_version"]),
        "status": "pass" if passed else "fail",
        "predicate": str(contract["predicate"]),
        "minimum_macro_mean_success_rate": success_floor,
        "minimum_macro_mean_max_progress_m": progress_floor,
        "observed_macro_mean_success_rate": observed_success,
        "observed_macro_mean_max_progress_m": observed_progress,
        "success_floor_met": success_met,
        "progress_floor_met": progress_met,
        "threshold_relaxation_allowed": False,
        "reason": (
            "frozen_reference_capability_floor_met"
            if passed
            else "frozen_reference_capability_floor_not_met"
        ),
    }


def _require_contracts(
    protocol: StairProtocolV1,
    protocol_v2: StairProtocolV2Draft,
    evaluation_protocol: EvaluationProtocol,
) -> None:
    if not isinstance(protocol, StairProtocolV1):
        raise TypeError("stair reference capability requires StairProtocolV1")
    if not isinstance(protocol_v2, StairProtocolV2Draft):
        raise TypeError("stair reference capability requires StairProtocolV2Draft")
    if not isinstance(evaluation_protocol, EvaluationProtocol):
        raise TypeError("stair reference capability requires EvaluationProtocol")


def _gate_contract(protocol_v2: StairProtocolV2Draft) -> Mapping[str, Any]:
    contract = _mapping(
        protocol_v2.to_public_dict().get("reference_capability_gate"),
        "reference_capability_gate",
    )
    if set(contract) != {
        "schema_version",
        "source_split",
        "seed_count",
        "low_step_bin_ids",
        "bin_coverage",
        "aggregation",
        "predicate",
        "minimum_macro_mean_success_rate",
        "minimum_macro_mean_max_progress_m",
        "threshold_relaxation_after_pilot_allowed",
    }:
        raise ValueError("stair reference capability contract fields mismatch")
    if contract.get("threshold_relaxation_after_pilot_allowed") is not False:
        raise ValueError("stair reference capability threshold can be relaxed")
    return contract


def _finite_bin_mapping(
    value: Any,
    *,
    low_bins: tuple[str, ...],
    field_name: str,
    minimum: float,
    maximum: float | None,
) -> dict[str, float]:
    raw = _mapping(value, field_name)
    if set(raw) != set(low_bins):
        raise ValueError(f"{field_name} bins mismatch")
    result = {bin_id: _finite(raw[bin_id], field_name) for bin_id in low_bins}
    if any(item < minimum for item in result.values()):
        raise ValueError(f"{field_name} is below its minimum")
    if maximum is not None and any(item > maximum for item in result.values()):
        raise ValueError(f"{field_name} exceeds its maximum")
    return result


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _finite(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{field_name} must be finite")
    return float(value)


__all__ = [
    "STAIR_REFERENCE_CAPABILITY_EVIDENCE_SCHEMA_VERSION",
    "STAIR_PILOT_TRAIN_BASE_SEED",
    "assert_stair_pilot_formal_bindings_disjoint",
    "build_stair_pilot_config",
    "build_stair_pilot_evaluation_binding",
    "build_stair_pilot_training_binding",
    "build_stair_reference_capability_evidence",
    "stair_pilot_training_seed_domain_hash",
    "stair_pilot_training_seed_index",
    "validate_stair_reference_capability_evidence",
]
