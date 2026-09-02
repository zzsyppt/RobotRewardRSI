"""Task-owned public projection for production Stair evaluation."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from typing import Any

from true_rl_agent.candidates.authoring_policy import (
    canonical_mechanism_runtime_identity,
    require_mechanism_operation_delta,
)
from true_rl_agent.evaluation.acceptance_protocol import AcceptanceProtocol
from true_rl_agent.evaluation.binding import is_sha256_digest
from true_rl_agent.evaluation.public_metric_contract import (
    RUNNER_OWNED_PUBLIC_EVAL_METRIC_KEYS,
    SEALED_VALIDATION_EVIDENCE_KEY,
)
from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.composition.cell_manifest import ResolvedCellManifestV1
from true_rl_agent.prompting.capabilities import (
    COMPLETE_REWARD_PROGRAM_CAPABILITY_ID,
)
from true_rl_agent.runner.budget import build_budget_accounting
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.evaluation_contract import (
    evaluate_stair_sealed_score_gate,
    evaluate_unified_stair_sealed_score_gate,
    validate_stair_split_summary,
)
from true_rl_agent.tasks.stair_climbing.evaluation_profile import (
    StairEvaluationProfile,
)
from true_rl_agent.tasks.stair_climbing.generalization_contract import (
    validate_stair_generalization_summary,
)
from true_rl_agent.tasks.stair_climbing.unified_scoring import (
    build_unified_reference_score_projection,
    build_unified_score_projection,
    build_unified_stair_prompt_feedback,
    validate_objective_family_mask,
    validate_unified_stair_summary,
)
from true_rl_agent.tasks.stair_climbing.runtime_profile import (
    STAIR_REWARD_ACTIVATION_PREFLIGHT_SCHEMA_VERSION,
)
from true_rl_agent.tasks.stair_climbing.training_distribution import (
    build_stair_execution_source_preflight_summary,
    stair_training_distribution_upstream_source_evidence,
)

_TASK_DERIVED_INPUT_KEYS = {
    "environment_source_preflight",
    "stair_evaluation",
    "public_dev",
    "public_accept",
    "acceptance_objective",
    "generalization_public_dev",
    "prompt_feedback",
    "accept_evidence_gate",
    "budget_accounting",
    "runtime_activation_gate",
    "stair_behavior_gate",
    "acceptance",
    "objective_family_mask",
}

_PUBLIC_DEV_OBJECTIVE_SCORE_SOURCE = (
    "generalization_public_dev.score_stats.mean"
)

STAIR_CANDIDATE_RUNTIME_EFFECT_EVIDENCE_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.candidate_runtime_effect_evidence.v1"
)


def project_stair_public_result(
    *,
    config: StairClimbingTaskConfig,
    public_result: PublicEvalResult,
    train_summary: Mapping[str, Any],
    mechanism_operation_delta: Mapping[str, Any] | None = None,
    resolved_cell_manifest: ResolvedCellManifestV1 | None = None,
    objective_family_mask: Mapping[str, Any] | None = None,
) -> PublicEvalResult:
    """Validate recorder summaries and derive every Stair acceptance input."""

    if not isinstance(config, StairClimbingTaskConfig):
        raise TypeError("stair public projection requires StairClimbingTaskConfig")
    if not isinstance(public_result, PublicEvalResult):
        raise TypeError("stair public projection requires PublicEvalResult")
    if not isinstance(train_summary, Mapping):
        raise TypeError("stair public projection train_summary must be an object")

    if config.is_unified_protocol:
        return _project_unified_stair_public_result(
            config=config,
            public_result=public_result,
            train_summary=train_summary,
            mechanism_operation_delta=mechanism_operation_delta,
            resolved_cell_manifest=resolved_cell_manifest,
            objective_family_mask=objective_family_mask,
        )

    return _project_legacy_stair_public_result(
        config=config,
        public_result=public_result,
        train_summary=train_summary,
        mechanism_operation_delta=mechanism_operation_delta,
    )


def _project_unified_stair_public_result(
    *,
    config: StairClimbingTaskConfig,
    public_result: PublicEvalResult,
    train_summary: Mapping[str, Any],
    mechanism_operation_delta: Mapping[str, Any] | None,
    resolved_cell_manifest: ResolvedCellManifestV1 | None,
    objective_family_mask: Mapping[str, Any] | None,
) -> PublicEvalResult:
    incoming = dict(public_result.metrics)
    unsupported = (
        set(incoming)
        - _TASK_DERIVED_INPUT_KEYS
        - RUNNER_OWNED_PUBLIC_EVAL_METRIC_KEYS
        - {"unified_score"}
    )
    if unsupported:
        raise ValueError(
            "unified Stair eval metrics contain unsupported field(s): "
            + ", ".join(sorted(unsupported))
        )
    source_preflight = _validate_environment_source_preflight_pair(
        config,
        train_summary=train_summary,
        eval_preflight=incoming.get("environment_source_preflight"),
    )
    raw_splits = incoming.get("stair_evaluation")
    if not isinstance(raw_splits, Mapping) or set(raw_splits) != {
        "public_dev",
        "public_accept",
    }:
        raise ValueError("unified Stair evaluation must contain two public splits")
    suite = config.ordered_evaluation_suite
    semantics = config.evaluation_semantics
    public_dev = validate_unified_stair_summary(
        suite=suite,
        semantics=semantics,
        split="public_dev",
        summary=raw_splits["public_dev"],
    )
    public_accept = validate_unified_stair_summary(
        suite=suite,
        semantics=semantics,
        split="public_accept",
        summary=raw_splits["public_accept"],
    )
    raw_mask = incoming.get("objective_family_mask")
    if raw_mask is None:
        raise ValueError("unified Stair result lacks its objective family mask")
    is_reference_projection = objective_family_mask is None
    raw_mask = validate_objective_family_mask(
        suite=suite,
        mask=raw_mask,
        public_accept_summary=(
            public_accept if is_reference_projection else None
        ),
        require_nonempty=not is_reference_projection,
    )
    if objective_family_mask is not None:
        expected_mask = validate_objective_family_mask(
            suite=suite,
            mask=objective_family_mask,
        )
        if raw_mask != expected_mask:
            raise ValueError("unified Stair result used a different Genesis mask")
    objective_family_mask = raw_mask
    score_builder = (
        build_unified_reference_score_projection
        if is_reference_projection
        else build_unified_score_projection
    )
    unified_score = score_builder(
        public_accept,
        objective_family_mask=objective_family_mask,
    )
    dev_score = score_builder(
        public_dev,
        objective_family_mask=objective_family_mask,
    )
    incoming_score = incoming.get("unified_score")
    if not isinstance(incoming_score, Mapping) or dict(incoming_score) != unified_score:
        raise ValueError("unified Stair runner score differs from frozen-mask replay")
    canonical_score = float(unified_score["j_primary"]["score_stats"]["mean"])
    if not math.isclose(
        public_result.public_score,
        canonical_score,
        rel_tol=1.0e-9,
        abs_tol=1.0e-9,
    ):
        raise ValueError("unified Stair public score differs from J_primary")
    _validate_unified_sealed_score_verdict(
        config,
        incoming.get(SEALED_VALIDATION_EVIDENCE_KEY),
        public_mean=canonical_score,
    )
    budget_gate = _budget_accounting_gate(config, train_summary)
    runtime_gate = build_stair_runtime_activation_gate(
        config,
        train_summary,
        mechanism_operation_delta=mechanism_operation_delta,
        resolved_cell_manifest=resolved_cell_manifest,
    )
    acceptance_protocol = config.acceptance_protocol
    named_guards = {}
    for guard in acceptance_protocol.named_guards:
        source = unified_score[guard.name]
        named_guards[guard.name] = {
            **guard.to_public_dict(),
            "score_stats": dict(source["score_stats"]),
            "score_samples": list(source["seed_macro_scores"]),
        }
    score_source = unified_score["j_primary"]
    acceptance = {
        "score_source": acceptance_protocol.score_source,
        "public_score_binding": acceptance_protocol.public_score_binding,
        "score_stats": dict(score_source["score_stats"]),
        "score_samples": list(score_source["seed_macro_scores"]),
        "named_guards": named_guards,
        "min_score_count": acceptance_protocol.min_score_count,
        "missing_required_score": False,
        "required_pass_gates": list(config.unified_protocol.required_pass_gates),
        "acceptance_protocol_hash": acceptance_protocol.protocol_hash,
        "absolute_accept_floor": acceptance_protocol.absolute_accept_floor,
        "min_delta": acceptance_protocol.min_delta,
        "max_public_to_sealed_score_drop": (
            acceptance_protocol.max_public_to_sealed_score_drop
        ),
    }
    metrics = {
        key: incoming[key]
        for key in RUNNER_OWNED_PUBLIC_EVAL_METRIC_KEYS
        if key in incoming
    }
    metrics.update(
        {
            "environment_source_preflight": source_preflight,
            "public_dev": public_dev,
            "public_accept": public_accept,
            "unified_score": unified_score,
            "objective_family_mask": dict(objective_family_mask),
            "generalization_public_dev": dev_score,
            "prompt_feedback": build_unified_stair_prompt_feedback(
                suite=suite,
                semantics=semantics,
                summary=public_dev,
                objective_family_mask=objective_family_mask,
            ),
            "accept_evidence_gate": {
                "status": "pass",
                "reason": "unified_paired_public_accept_evidence_complete",
                "observed_count": score_source["score_stats"]["count"],
                "required_count": acceptance_protocol.min_score_count,
                "statistics_unit": "one_stratified_macro_score_per_paired_seed",
            },
            "budget_accounting": budget_gate,
            "runtime_activation_gate": runtime_gate,
            "stair_behavior_gate": {
                "status": "pass",
                "reason": "unified_task_recorder_strata_complete",
                "source": "runner_owned_stair_transition_recorder",
                "validated_splits": ["public_dev", "public_accept"],
                "ordered_suite_hash": suite.suite_hash,
                "classification_warning_count": 0,
            },
            "acceptance": acceptance,
        }
    )
    return PublicEvalResult(
        status=public_result.status,
        public_score=canonical_score,
        metrics=metrics,
    )


def _project_legacy_stair_public_result(
    *,
    config: StairClimbingTaskConfig,
    public_result: PublicEvalResult,
    train_summary: Mapping[str, Any],
    mechanism_operation_delta: Mapping[str, Any] | None,
) -> PublicEvalResult:
    incoming = dict(public_result.metrics)
    if "objective_family_mask" in incoming:
        raise ValueError("legacy Stair eval cannot carry an objective family mask")
    unsupported = (
        set(incoming)
        - _TASK_DERIVED_INPUT_KEYS
        - RUNNER_OWNED_PUBLIC_EVAL_METRIC_KEYS
    )
    if unsupported:
        raise ValueError(
            "stair eval metrics contain unsupported field(s): "
            + ", ".join(sorted(unsupported))
        )
    source_preflight = _validate_environment_source_preflight_pair(
        config,
        train_summary=train_summary,
        eval_preflight=incoming.get("environment_source_preflight"),
    )
    raw_splits = incoming.get("stair_evaluation")
    if not isinstance(raw_splits, Mapping) or set(raw_splits) != {
        "public_dev",
        "public_accept",
    }:
        raise ValueError("stair evaluation must contain exactly the two public splits")

    evaluation_protocol = config.evaluation_protocol()
    if config.evaluation_profile_config is None:
        raise ValueError("production Stair projection requires evaluation profile")
    profile = config.evaluation_profile
    for split_name in ("public_dev", "public_accept"):
        split_payload = raw_splits.get(split_name)
        if not isinstance(split_payload, Mapping) or set(split_payload) != {
            "anchor",
            "generalization",
        }:
            raise ValueError(
                f"stair {split_name} must contain anchor and generalization summaries"
            )

    public_dev_raw = _mapping(raw_splits["public_dev"], "stair public_dev pair")
    public_accept_raw = _mapping(
        raw_splits["public_accept"],
        "stair public_accept pair",
    )
    public_dev = validate_stair_split_summary(
        config.protocol_v1,
        semantics=config.evaluation_semantics,
        evaluation_protocol=evaluation_protocol,
        split="public_dev",
        summary=_mapping(public_dev_raw["anchor"], "stair public_dev anchor"),
    )
    public_accept = validate_stair_split_summary(
        config.protocol_v1,
        semantics=config.evaluation_semantics,
        evaluation_protocol=evaluation_protocol,
        split="public_accept",
        summary=_mapping(
            public_accept_raw["anchor"],
            "stair public_accept anchor",
        ),
    )
    objective_dev = validate_stair_generalization_summary(
        profile=profile,
        semantics=config.evaluation_semantics,
        evaluation_protocol=evaluation_protocol,
        split="public_dev",
        summary=_mapping(
            public_dev_raw["generalization"],
            "stair public_dev generalization",
        ),
    )
    objective_accept = validate_stair_generalization_summary(
        profile=profile,
        semantics=config.evaluation_semantics,
        evaluation_protocol=evaluation_protocol,
        split="public_accept",
        summary=_mapping(
            public_accept_raw["generalization"],
            "stair public_accept generalization",
        ),
    )
    _validate_sealed_score_verdict(
        config,
        incoming.get(SEALED_VALIDATION_EVIDENCE_KEY),
        public_mean=float(objective_accept["score_stats"]["mean"]),
    )
    canonical_score = float(public_accept["score_stats"]["mean"])
    if not math.isclose(
        public_result.public_score,
        canonical_score,
        rel_tol=1.0e-9,
        abs_tol=1.0e-9,
    ):
        raise ValueError("stair public score differs from public_accept macro mean")

    budget_gate = _budget_accounting_gate(config, train_summary)
    runtime_gate = build_stair_runtime_activation_gate(
        config,
        train_summary,
        mechanism_operation_delta=mechanism_operation_delta,
    )
    behavior_gate = {
        "status": "pass",
        "reason": "task_recorder_public_split_evidence_complete",
        "source": "runner_owned_stair_transition_recorder",
        "validated_splits": ["public_dev", "public_accept"],
        "complete_seven_bin_coverage": True,
        "complete_height_tread_coverage": True,
        "classification_warning_count": 0,
    }
    accept_gate = {
        "status": "pass",
        "reason": "sufficient_recomputed_public_accept_evidence",
        "observed_count": objective_accept["score_stats"]["count"],
        "required_count": config.acceptance_protocol.min_score_count,
        "statistics_unit": "one_equal_height_tread_cell_macro_score_per_seed",
    }
    acceptance_protocol = config.acceptance_protocol
    required_gates = tuple(
        config.protocol_v1.to_dict()["acceptance"]["required_pass_gates"]
    )
    acceptance = {
        "score_source": acceptance_protocol.score_source,
        "public_score_binding": acceptance_protocol.public_score_binding,
        "score_stats": dict(objective_accept["score_stats"]),
        "guard_score_source": acceptance_protocol.guard_score_source,
        "guard_score_stats": dict(public_accept["score_stats"]),
        "guard_absolute_accept_floor": (
            acceptance_protocol.guard_absolute_accept_floor
        ),
        "guard_max_regression": acceptance_protocol.guard_max_regression,
        "min_score_count": acceptance_protocol.min_score_count,
        "missing_required_score": False,
        "required_pass_gates": list(required_gates),
        "acceptance_protocol_hash": acceptance_protocol.protocol_hash,
        "absolute_accept_floor": acceptance_protocol.absolute_accept_floor,
        "min_delta": acceptance_protocol.min_delta,
    }
    if acceptance_protocol.requires_paired_score_samples:
        acceptance["score_samples"] = list(objective_accept["seed_macro_scores"])
    if acceptance_protocol.requires_paired_guard_score_samples:
        acceptance.update(
            {
                "guard_score_samples": list(public_accept["seed_macro_scores"]),
                "guard_regression_margin": (
                    acceptance_protocol.guard_regression_margin
                ),
                "max_public_to_sealed_score_drop": (
                    acceptance_protocol.max_public_to_sealed_score_drop
                ),
            }
        )

    metrics = {
        key: incoming[key]
        for key in RUNNER_OWNED_PUBLIC_EVAL_METRIC_KEYS
        if key in incoming
    }
    metrics.update(
        {
            "environment_source_preflight": source_preflight,
            "public_dev": public_dev,
            "public_accept": public_accept,
            "generalization_public_dev": objective_dev,
            "acceptance_objective": objective_accept,
            "prompt_feedback": build_stair_prompt_feedback(
                public_dev,
                generalization_public_dev=objective_dev,
                evaluation_profile=profile,
            ),
            "accept_evidence_gate": accept_gate,
            "budget_accounting": budget_gate,
            "runtime_activation_gate": runtime_gate,
            "stair_behavior_gate": behavior_gate,
            "acceptance": acceptance,
        }
    )
    return PublicEvalResult(
        status=public_result.status,
        public_score=canonical_score,
        metrics=metrics,
    )


def _validate_unified_sealed_score_verdict(
    config: StairClimbingTaskConfig,
    evidence: Any,
    *,
    public_mean: float,
) -> None:
    if evidence is None:
        return
    protocol_label = "unified Stair"
    if not isinstance(evidence, Mapping):
        raise ValueError(f"{protocol_label} sealed validation evidence must be an object")
    score_stats = evidence.get("score_stats")
    if not isinstance(score_stats, Mapping):
        raise ValueError(f"{protocol_label} sealed score_stats must be an object")
    sealed_mean = score_stats.get("mean")
    if (
        isinstance(sealed_mean, bool)
        or not isinstance(sealed_mean, int | float)
        or not math.isfinite(float(sealed_mean))
    ):
        raise ValueError(f"{protocol_label} sealed score mean is invalid")
    passed, expected_reason = evaluate_unified_stair_sealed_score_gate(
        sealed_mean=float(sealed_mean),
        public_mean=public_mean,
        acceptance_protocol=config.acceptance_protocol,
        evaluation_subject=ExperimentSubject.from_mapping(
            evidence.get("evaluation_subject")
        ),
    )
    expected_status = "pass" if passed else "fail"
    if evidence.get("status") != expected_status or evidence.get("reason") != expected_reason:
        raise ValueError(f"{protocol_label} sealed score verdict differs from task policy")


def _validate_sealed_score_verdict(
    config: StairClimbingTaskConfig,
    evidence: Any,
    *,
    public_mean: float,
) -> None:
    if evidence is None:
        return
    if not isinstance(evidence, Mapping):
        raise ValueError("Stair sealed validation evidence must be an object")
    if not is_sha256_digest(evidence.get("task_evidence_hash")):
        raise ValueError(
            "Stair sealed validation evidence requires a canonical task evidence hash"
        )
    score_stats = evidence.get("score_stats")
    if not isinstance(score_stats, Mapping):
        raise ValueError("Stair sealed validation score_stats must be an object")
    expected_passed, expected_reason = evaluate_stair_sealed_score_gate(
        sealed_mean=score_stats.get("mean"),
        public_mean=public_mean,
        acceptance_protocol=config.acceptance_protocol,
        evaluation_profile=config.evaluation_profile,
    )
    if evidence.get("status") != ("pass" if expected_passed else "fail"):
        raise ValueError("Stair sealed validation status differs from task score gate")
    if evidence.get("reason") != expected_reason:
        raise ValueError("Stair sealed validation reason differs from task score gate")


def _validate_environment_source_preflight_pair(
    config: StairClimbingTaskConfig,
    *,
    train_summary: Mapping[str, Any],
    eval_preflight: Any,
) -> dict[str, str]:
    if config.is_unified_protocol:
        observed_train = train_summary.get("environment_source_preflight")
        if not isinstance(observed_train, Mapping) or not isinstance(
            eval_preflight, Mapping
        ):
            raise ValueError("unified Stair source preflight evidence is missing")
        train_evidence = dict(observed_train)
        eval_evidence = dict(eval_preflight)
        train_task_hash = train_evidence.get("task_protocol_hash")
        predecessor_training = (
            train_task_hash == config.training_binding_task_protocol_hash
            and train_task_hash != config.task_protocol_hash
        )
        predecessor_identity_matches = True
        if predecessor_training:
            lineage = config.unified_protocol.training_lineage
            resolved_identity = train_summary.get("resolved_cell_identity")
            predecessor_identity_matches = bool(
                lineage is not None
                and config.training_predecessor_manifest_hash is not None
                and train_summary.get("resolved_cell_manifest_hash")
                == config.training_predecessor_manifest_hash
                and isinstance(resolved_identity, Mapping)
                and resolved_identity.get("cell_variant_id")
                == lineage["source_cell_variant_id"]
                and resolved_identity.get("resolved_cell_task_protocol_hash")
                == config.training_binding_task_protocol_hash
            )
        source_manifests_match = (
            predecessor_training
            or eval_evidence.get("source_manifest_hash")
            == train_evidence.get("source_manifest_hash")
        )
        if (
            train_evidence.get("status") != "pass"
            or eval_evidence.get("status") != "pass"
            or train_task_hash
            not in {
                config.training_binding_task_protocol_hash,
                config.task_protocol_hash,
            }
            or eval_evidence.get("task_protocol_hash") != config.task_protocol_hash
            or not is_sha256_digest(train_evidence.get("source_manifest_hash"))
            or not is_sha256_digest(eval_evidence.get("source_manifest_hash"))
            or not source_manifests_match
            or not predecessor_identity_matches
        ):
            raise ValueError("unified Stair train/eval source preflight differs")
        return eval_evidence
    assert config.protocol_v2 is not None
    contract = config.protocol_v2.runner_owned_training_distribution
    if contract is None:
        raise ValueError("Stair Final V2 training distribution contract is missing")
    upstream_evidence = stair_training_distribution_upstream_source_evidence(contract)
    expected = build_stair_execution_source_preflight_summary(
        task_protocol_hash=config.task_protocol_hash,
        source_manifest_hash=str(upstream_evidence["source_manifest_hash"]),
    )
    observed = {
        "train": train_summary.get("environment_source_preflight"),
        "eval": eval_preflight,
    }
    for stage, value in observed.items():
        if not isinstance(value, Mapping) or dict(value) != expected:
            raise ValueError(
                f"Stair {stage} environment source preflight does not match Final V2"
            )
    return expected


def _budget_accounting_gate(
    config: StairClimbingTaskConfig,
    train_summary: Mapping[str, Any],
) -> dict[str, Any]:
    expected = build_budget_accounting(
        budget_steps=config.budget_steps,
        num_envs=config.train_num_envs,
        num_steps_per_env=config.train_steps_per_iteration,
        max_iterations=config.train_max_iterations,
    )
    training = train_summary.get("training")
    if not isinstance(training, Mapping):
        return {"status": "fail", "reason": "training_summary_missing"}
    observed = training.get("budget_accounting")
    if not isinstance(observed, Mapping):
        return {"status": "fail", "reason": "budget_accounting_missing"}
    if dict(observed) != expected:
        return {
            "status": "fail",
            "reason": "budget_accounting_differs_from_stair_config",
        }
    return dict(expected)


def build_stair_runtime_activation_gate(
    config: StairClimbingTaskConfig,
    train_summary: Mapping[str, Any],
    *,
    expected_control_calls: int | None = None,
    mechanism_operation_delta: Mapping[str, Any] | None = None,
    resolved_cell_manifest: ResolvedCellManifestV1 | None = None,
) -> dict[str, Any]:
    if expected_control_calls is None:
        expected_control_calls = (
            config.train_steps_per_iteration * config.train_max_iterations
        )
    elif (
        isinstance(expected_control_calls, bool)
        or not isinstance(expected_control_calls, int)
        or expected_control_calls <= 0
    ):
        raise ValueError("stair expected_control_calls must be a positive integer")
    expected_status = config.train_runtime_profile().to_public_dict()[
        "complete_reward_runtime"
    ]
    runtime = train_summary.get("runtime_activation_summary")
    if not isinstance(runtime, Mapping):
        return {
            "status": "fail",
            "reason": "complete_reward_runtime_evidence_missing",
            "runtime_status": expected_status,
        }
    calls = runtime.get("dispatcher_calls")
    term_calls = runtime.get("term_calls")
    term_count = runtime.get("declared_term_count")
    program = runtime.get("program")
    program_subject = program.get("subject") if isinstance(program, Mapping) else None
    component_records = runtime.get("component_activation_records")
    preflight = runtime.get("activation_preflight")
    preflight_calls = (
        preflight.get("real_transition_dispatcher_calls")
        if isinstance(preflight, Mapping)
        else None
    )
    runtime_effect = build_stair_candidate_runtime_effect_evidence(
        train_summary,
        mechanism_operation_delta=mechanism_operation_delta,
    )
    from true_rl_agent.tasks.stair_climbing.component_smoke import (
        StairClimbingComponentSmokeAdapter,
    )

    active_input_schema_hash = StairClimbingComponentSmokeAdapter(
        config,
        resolved_cell_manifest=resolved_cell_manifest,
    ).contract.input_schema_hash
    dispatcher_contract = (
        config.vnext_reward_authority.runtime_authority_payload()["dispatcher"]
        if config.is_unified_protocol
        else config.protocol_v2.complete_reward_runtime.dispatcher_contract
    )
    checks = {
        "runtime_status": runtime.get("runtime_status") == expected_status,
        "runtime_evidence_status": runtime.get("status") == "pass",
        "dispatcher_calls": (
            isinstance(calls, int) and not isinstance(calls, bool) and calls > 0
        ),
        "activation_preflight": (
            isinstance(preflight, Mapping)
            and preflight.get("schema_version")
            == STAIR_REWARD_ACTIVATION_PREFLIGHT_SCHEMA_VERSION
            and preflight.get("status") == "pass"
            and preflight_calls == 1
            and preflight.get("transition_common_step") == 1
            and preflight.get("repeat_count")
            == dispatcher_contract["determinism_preflight_repeats"]
            and isinstance(program, Mapping)
            and preflight.get("term_names") == program.get("term_names")
        ),
        "dispatcher_call_accounting": (
            isinstance(calls, int)
            and not isinstance(calls, bool)
            and calls == expected_control_calls + 1
        ),
        "term_call_accounting": (
            isinstance(term_calls, int)
            and not isinstance(term_calls, bool)
            and isinstance(term_count, int)
            and not isinstance(term_count, bool)
            and term_count > 0
            and isinstance(calls, int)
            and not isinstance(calls, bool)
            and term_calls == calls * term_count
        ),
        "input_guard": runtime.get("input_guard") == "pass",
        "rng_guard": runtime.get("rng_guard") == "pass",
        "source_alias_guard": runtime.get("source_alias_guard") == "pass",
        "single_reward_manager_term": runtime.get(
            "reward_manager_active_terms"
        )
        == ["complete_reward"],
        "manager_weight": runtime.get("reward_manager_term_weight") == 1.0,
        "training_recorder_disabled": runtime.get("training_recorder_enabled")
        is False,
        "generic_reward_term_calls": runtime.get("reward_term_calls")
        == term_calls,
        "generic_diagnostic_calls": runtime.get("diagnostic_calls") == 0,
        "generic_nonzero_flag": isinstance(
            runtime.get("nonzero_reward_contribution"),
            bool,
        ),
        "generic_component_records": (
            isinstance(component_records, list)
            and isinstance(program, Mapping)
            and len(component_records) == term_count
            and {
                record.get("name")
                for record in component_records
                if isinstance(record, Mapping)
            }
            == set(program.get("term_names", ()))
        ),
        "candidate_reward_operation_runtime_effect": (
            runtime_effect["status"] == "pass"
        ),
        "generic_capability_record": runtime.get(
            "capability_activation_records"
        )
        == [COMPLETE_REWARD_PROGRAM_CAPABILITY_ID],
        "program_identity": (
            isinstance(program, Mapping)
            and program.get("program_source_checksum")
            == train_summary.get("program_source_checksum")
            and program.get("subject_manifest_hash")
            == train_summary.get("subject_manifest_hash")
            and program.get("qualification_hash")
            == train_summary.get("reward_program_qualification_hash")
            and program.get("input_schema_hash") == active_input_schema_hash
            and isinstance(program_subject, Mapping)
            and program_subject.get("kind") == train_summary.get("subject_kind")
            and program_subject.get("id") == train_summary.get("subject_id")
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "status": "pass" if not failed else "fail",
        "reason": (
            "pre_reset_complete_reward_runtime_evidence_complete"
            if not failed
            else "complete_reward_runtime_evidence_incomplete"
        ),
        "runtime_status": expected_status,
        "failed_checks": failed,
        "dispatcher_calls": calls if checks["dispatcher_calls"] else None,
    }


def build_stair_candidate_runtime_effect_evidence(
    train_summary: Mapping[str, Any],
    *,
    mechanism_operation_delta: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate the candidate-owned reward delta's observed training effect.

    This is the single Stair-owned evaluator used by both public-dev screening
    and the later formal runtime gate.  It deliberately carries no acceptance
    authority.
    """

    if not isinstance(train_summary, Mapping):
        raise TypeError("Stair runtime-effect train summary must be an object")
    runtime = train_summary.get("runtime_activation_summary")
    program = runtime.get("program") if isinstance(runtime, Mapping) else None
    component_records = (
        runtime.get("component_activation_records")
        if isinstance(runtime, Mapping)
        else None
    )
    observed = _candidate_reward_operation_runtime_effect_is_observed(
        mechanism_operation_delta,
        program,
        component_records,
    )
    return {
        "schema_version": (
            STAIR_CANDIDATE_RUNTIME_EFFECT_EVIDENCE_SCHEMA_VERSION
        ),
        "status": "pass" if observed else "fail",
        "reason": (
            "candidate_reward_operation_runtime_effect_observed"
            if observed
            else "candidate_reward_operation_runtime_effect_not_observed"
        ),
        "acceptance_authority": False,
    }


def require_stair_candidate_runtime_effect_evidence(
    value: Any,
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "status",
        "reason",
        "acceptance_authority",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError("Stair candidate runtime-effect evidence fields mismatch")
    status = value.get("status")
    expected_reason = {
        "pass": "candidate_reward_operation_runtime_effect_observed",
        "fail": "candidate_reward_operation_runtime_effect_not_observed",
    }.get(status)
    if (
        value.get("schema_version")
        != STAIR_CANDIDATE_RUNTIME_EFFECT_EVIDENCE_SCHEMA_VERSION
        or value.get("reason") != expected_reason
        or value.get("acceptance_authority") is not False
    ):
        raise ValueError("Stair candidate runtime-effect evidence contract mismatch")
    return dict(value)


def _candidate_reward_operation_runtime_effect_is_observed(
    mechanism_operation_delta: Mapping[str, Any] | None,
    program: Any,
    component_records: Any,
) -> bool:
    if mechanism_operation_delta is None:
        return True
    try:
        delta = require_mechanism_operation_delta(mechanism_operation_delta)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Stair runtime-effect gate received an invalid verifier-owned "
            "mechanism operation delta"
        ) from exc
    if not isinstance(program, Mapping):
        return False
    term_names = program.get("term_names")
    if not isinstance(term_names, list):
        return False
    if not isinstance(component_records, list):
        return False

    operation = delta["operation"]
    component = (
        delta["before_component"]
        if operation == "remove"
        else delta["after_component"]
    )
    if not isinstance(component, Mapping):  # guarded by the strict delta validator
        raise ValueError("validated Stair mechanism operation lost its component")
    runtime_identity = canonical_mechanism_runtime_identity(
        component.get("mechanism_id")
    )
    if runtime_identity is None:  # guarded by the strict delta validator
        raise ValueError("validated Stair mechanism operation has no runtime identity")
    if runtime_identity[0] != "reward_term":
        raise ValueError(
            "Stair candidate runtime-effect gate supports RewardTerm deltas only"
        )

    if operation == "remove":
        return runtime_identity[1] not in term_names and not any(
            isinstance(record, Mapping)
            and record.get("component_kind") == runtime_identity[0]
            and record.get("name") == runtime_identity[1]
            for record in component_records
        )

    if runtime_identity[1] not in term_names:
        return False
    matches = [
        record
        for record in component_records
        if isinstance(record, Mapping)
        and record.get("component_kind") == runtime_identity[0]
        and record.get("name") == runtime_identity[1]
    ]
    if len(matches) != 1:
        return False
    contribution = matches[0].get("max_abs_weighted_contribution")
    return (
        isinstance(contribution, int | float)
        and not isinstance(contribution, bool)
        and math.isfinite(contribution)
        and contribution > 1.0e-12
    )


def build_stair_prompt_feedback(
    public_dev: Mapping[str, Any],
    *,
    generalization_public_dev: Mapping[str, Any] | None = None,
    evaluation_profile: StairEvaluationProfile | None = None,
) -> dict[str, Any]:
    """Build the bounded prompt projection from one validated public-dev split."""

    feedback_source = (
        generalization_public_dev
        if generalization_public_dev is not None
        else public_dev
    )
    failures = _mapping(
        feedback_source.get("primary_failure_breakdown"),
        "stair public failure breakdown",
    )
    failure_rates = {
        str(reason): float(rate) for reason, rate in failures.items()
    }
    diagnostic_rates = dict(
        _mapping(
            feedback_source.get("diagnostic_flag_rates"),
            "stair public diagnostic rates",
        )
    )
    failure_rate = min(1.0, max(0.0, sum(failure_rates.values())))
    climb_completion_rate = max(0.0, 1.0 - failure_rate)
    first_valid_top_arrival_rate = float(
        diagnostic_rates.get("first_valid_top_arrival_achieved", 0.0)
    )
    dominant_reason, dominant_rate = max(
        failure_rates.items(),
        key=lambda item: (float(item[1]), str(item[0])),
    )
    return {
        "schema_version": "true_rl_agent.task.stair_climbing.prompt_feedback.v9",
        "source_split": "public_dev",
        "mean_success_rate_by_bin": _mean_success_rate_by_bin(public_dev),
        "generalization_summary": (
            {
                "public_dev_objective_score": float(
                    generalization_public_dev["score_stats"]["mean"]
                ),
                "success_rate_by_height": dict(
                    generalization_public_dev[
                        "aggregate_success_rate_by_height"
                    ]
                ),
                "success_rate_by_tread": dict(
                    generalization_public_dev[
                        "aggregate_success_rate_by_tread"
                    ]
                ),
                "difficulty_diagnostics": dict(
                    generalization_public_dev["difficulty_diagnostics"]
                ),
                **_generalization_count_grid_projection(
                    evaluation_profile,
                    generalization_public_dev,
                ),
            }
            if generalization_public_dev is not None
            else {"status": "not_available"}
        ),
        "primary_failure_breakdown": failure_rates,
        "dominant_failure": {
            "reason": dominant_reason,
            "rate": float(dominant_rate),
        },
        "diagnostic_flag_rates": diagnostic_rates,
        "stable_success_summary": {
            "stable_top_success_rate": climb_completion_rate,
            "valid_top_arrival_rate": first_valid_top_arrival_rate,
            "top_arrival_without_stable_success_rate": max(
                0.0,
                first_valid_top_arrival_rate - climb_completion_rate,
            ),
        },
        "progress_summary": dict(
            _mapping(
                feedback_source.get("progress_summary"),
                "stair public progress summary",
            )
        ),
    }


def build_stair_prompt_score_semantics(
    acceptance_protocol: AcceptanceProtocol,
) -> dict[str, Any]:
    """Name the three Stair score roles exposed to candidate authoring."""

    if not isinstance(acceptance_protocol, AcceptanceProtocol):
        raise TypeError(
            "stair prompt score semantics requires an AcceptanceProtocol"
        )
    sources = {
        "objective_score": acceptance_protocol.score_source,
        "public_dev_objective_score": _PUBLIC_DEV_OBJECTIVE_SCORE_SOURCE,
    }
    if acceptance_protocol.uses_guard_score:
        sources["guard_score"] = str(acceptance_protocol.guard_score_source)
    if acceptance_protocol.uses_named_guards:
        sources["named_guards"] = {
            guard.name: guard.score_source
            for guard in acceptance_protocol.named_guards
        }
    return {
        "score_sources_by_role": sources,
        "diagnostic_only_roles": ["public_dev_objective_score"],
    }


def _generalization_count_grid_projection(
    profile: StairEvaluationProfile | None,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    if profile is None:
        raise ValueError(
            "Stair generalization prompt feedback requires an evaluation profile"
        )
    success_rates = _mapping(
        summary.get("aggregate_success_rate_by_cell"),
        "stair aggregate success rates by cell",
    )
    outcomes_by_cell = _mapping(
        summary.get("primary_outcome_counts_by_cell"),
        "stair primary outcomes by cell",
    )
    expected_cell_ids = {cell.cell_id for cell in profile.cells}
    if set(success_rates) != expected_cell_ids or set(outcomes_by_cell) != expected_cell_ids:
        raise ValueError("Stair generalization prompt cell coverage is incomplete")
    cell_by_coordinates = {
        (cell.bin_id, cell.tread_depth_m): cell for cell in profile.cells
    }
    denominator = (
        int(summary["seed_count"]) * profile.episodes_per_seed_per_cell
    )
    first_counts = _mapping(
        outcomes_by_cell[profile.cells[0].cell_id],
        "stair primary outcome counts",
    )
    reasons = tuple(str(reason) for reason in first_counts)
    if not reasons:
        raise ValueError("Stair generalization prompt outcome reasons are empty")
    failure_reasons = tuple(reason for reason in reasons if reason != "success")
    if len(failure_reasons) != len(reasons) - 1:
        raise ValueError("Stair generalization prompt outcomes require one success reason")

    canonical_counts: dict[str, dict[str, int]] = {}
    success_counts: dict[str, int] = {}
    for cell_id in expected_cell_ids:
        raw_counts = _mapping(
            outcomes_by_cell[cell_id],
            f"stair primary outcome counts {cell_id}",
        )
        if set(raw_counts) != set(reasons):
            raise ValueError("Stair generalization prompt outcome reasons mismatch")
        cell_counts: dict[str, int] = {}
        for reason in reasons:
            count = raw_counts[reason]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("Stair generalization prompt outcome count is invalid")
            cell_counts[reason] = count
        if sum(cell_counts.values()) != denominator:
            raise ValueError("Stair generalization prompt outcome total mismatch")
        rate_count = round(float(success_rates[cell_id]) * denominator)
        if rate_count != cell_counts.get("success"):
            raise ValueError("Stair generalization prompt success count mismatch")
        canonical_counts[cell_id] = cell_counts
        success_counts[cell_id] = rate_count

    def grid_for(values: Mapping[str, int]) -> list[list[int]]:
        return [
            [
                values[cell_by_coordinates[(anchor.bin_id, tread)].cell_id]
                for tread in profile.tread_depths_m
            ]
            for anchor in profile.anchor_cells
        ]

    return {
        "grid_axes": {
            "layout": "height_rows_x_tread_columns",
            "height_bin_ids": [cell.bin_id for cell in profile.anchor_cells],
            "tread_depths_m": list(profile.tread_depths_m),
            "value_encoding": "episode_count",
            "count_denominator": denominator,
        },
        "success_count_grid": grid_for(success_counts),
        "primary_failure_reasons": list(failure_reasons),
        "zero_count_grids_omitted": True,
        "primary_failure_count_grids": {
            reason: grid_for(
                {
                    cell_id: counts[reason]
                    for cell_id, counts in canonical_counts.items()
                }
            )
            for reason in failure_reasons
            if any(counts[reason] > 0 for counts in canonical_counts.values())
        },
    }


def _mean_success_rate_by_bin(
    public_dev: Mapping[str, Any],
) -> dict[str, float]:
    raw_seeds = public_dev.get("seed_summaries")
    if not isinstance(raw_seeds, list | tuple) or not raw_seeds:
        raise ValueError("stair public seed summaries must be a non-empty list")
    rates_by_seed: list[dict[str, float]] = []
    expected_bins: set[str] | None = None
    for index, raw_seed in enumerate(raw_seeds):
        seed = _mapping(raw_seed, f"stair public seed summary {index}")
        raw_rates = _mapping(
            seed.get("bin_success_rates"),
            f"stair public seed summary {index} bin success rates",
        )
        rates = {str(bin_id): float(rate) for bin_id, rate in raw_rates.items()}
        observed_bins = set(rates)
        if expected_bins is None:
            expected_bins = observed_bins
        elif observed_bins != expected_bins:
            raise ValueError("stair public seed summaries disagree on bin coverage")
        rates_by_seed.append(rates)
    if not expected_bins:
        raise ValueError("stair public seed summaries contain no bins")
    return {
        bin_id: statistics.fmean(seed[bin_id] for seed in rates_by_seed)
        for bin_id in sorted(expected_bins)
    }


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


__all__ = [
    "STAIR_CANDIDATE_RUNTIME_EFFECT_EVIDENCE_SCHEMA_VERSION",
    "build_stair_candidate_runtime_effect_evidence",
    "build_stair_prompt_feedback",
    "build_stair_prompt_score_semantics",
    "build_stair_runtime_activation_gate",
    "project_stair_public_result",
    "require_stair_candidate_runtime_effect_evidence",
]
