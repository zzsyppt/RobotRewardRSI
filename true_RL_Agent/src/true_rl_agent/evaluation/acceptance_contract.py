"""Shared acceptance evidence contract for eval decisions and audits."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from true_rl_agent.evaluation.acceptance_protocol import (
    PUBLIC_SCORE_BINDING_ACCEPTANCE_MEAN,
    PUBLIC_SCORE_BINDING_GUARD_MEAN,
    AcceptanceProtocol,
    validated_score_samples,
)
from true_rl_agent.evaluation.result_contract import validate_finite_number
from true_rl_agent.evaluation.sealed_validation import (
    CANDIDATE_CONTAMINATION_FIELDS,
    SEALED_VALIDATION_EVIDENCE_KEY,
    validate_sealed_validation_gate_contract,
    validate_sealed_validation_gate_evidence,
)

COMMON_ACCEPT_REQUIRED_GATES = (
    "accept_evidence_gate",
    "budget_accounting",
    "sealed_validation_gate",
)
BASELINE_ACCEPT_REQUIRED_GATES = (
    "accept_evidence_gate",
    "sealed_validation_gate",
)
CANDIDATE_ONLY_ACCEPT_GATES = frozenset({"budget_accounting"})
_PUBLIC_SCORE_UNSET = object()


@dataclass(frozen=True)
class AcceptanceEvidenceValidation:
    valid: bool
    reasons: tuple[str, ...]
    required_gates: tuple[str, ...]
    gate_statuses: Mapping[str, str] = field(default_factory=dict)
    missing_common_gates: tuple[str, ...] = ()
    failed_required_gates: tuple[str, ...] = ()
    missing_consumed_gates: tuple[str, ...] = ()
    gate_status_mismatches: tuple[str, ...] = ()
    eval_protocol_hash: str | None = None
    protocol_binding_valid: bool = False
    score_stats: Mapping[str, Any] = field(default_factory=dict)
    score_stats_binding_valid: bool = False
    score_count: int | None = None
    min_score_count: int | None = None
    score_mean: float | None = None
    standard_error: float | None = None
    score_samples: tuple[float, ...] = ()
    guard_score_stats: Mapping[str, Any] = field(default_factory=dict)
    guard_score_mean: float | None = None
    guard_standard_error: float | None = None
    guard_score_samples: tuple[float, ...] = ()
    guard_score_source: str | None = None
    guard_absolute_accept_floor: float | None = None
    guard_max_regression: float | None = None
    guard_regression_margin: float | None = None
    max_public_to_sealed_score_drop: float | None = None
    guard_score_binding_valid: bool = False
    named_guard_evidence: Mapping[str, Any] = field(default_factory=dict)
    named_guard_binding_valid: bool = False
    score_source: str | None = None
    public_score_binding: str | None = None
    public_score_binding_valid: bool = False
    canonical_public_score: float | None = None
    absolute_accept_floor: float | None = None
    min_delta: float | None = None
    acceptance_protocol_hash: str | None = None
    acceptance_protocol_binding_valid: bool = False
    sealed_validation_status: str = "missing"
    sealed_validation_valid: bool = False
    sealed_validation_complete: bool = False
    sealed_validation_evidence: Mapping[str, Any] = field(default_factory=dict)
    sealed_validation_gate: Mapping[str, Any] = field(default_factory=dict)
    reject_reason: str | None = None

    def has_reason(self, *prefixes: str) -> bool:
        return any(reason.startswith(prefix) for reason in self.reasons for prefix in prefixes)


def missing_common_accept_gates(
    required_gates: set[str],
    *,
    evidence_subject_kind: str = "candidate",
) -> tuple[str, ...]:
    common_gates = _common_required_gates(evidence_subject_kind)
    return tuple(gate for gate in common_gates if gate not in required_gates)


def validate_acceptance_evidence(
    metrics: Mapping[str, Any],
    *,
    decision: Mapping[str, Any] | None = None,
    acceptance_protocol: AcceptanceProtocol | None = None,
    public_score: Any = _PUBLIC_SCORE_UNSET,
    expected_sealed_seed_count: int | None = None,
    expected_checkpoint_hash: str | None = None,
    evidence_subject_kind: str = "candidate",
) -> AcceptanceEvidenceValidation:
    _common_required_gates(evidence_subject_kind)
    acceptance = _mapping(metrics.get("acceptance"))
    declared_required_gates = tuple(
        _string_list(acceptance.get("required_pass_gates"))
    )
    required_gates = tuple(
        gate
        for gate in declared_required_gates
        if not (
            evidence_subject_kind == "baseline"
            and gate in CANDIDATE_ONLY_ACCEPT_GATES
        )
    )
    required_gate_set = set(required_gates)
    missing_common = missing_common_accept_gates(
        required_gate_set,
        evidence_subject_kind=evidence_subject_kind,
    )
    reasons = [f"missing_common_accept_gate:{gate}" for gate in missing_common]
    reasons.extend(_candidate_contamination_reasons(metrics))

    declared_score_source = _optional_text(acceptance.get("score_source"))
    effective_score_source = (
        acceptance_protocol.score_source
        if acceptance_protocol is not None
        else declared_score_source
    )
    public_accept = _score_summary_from_source(metrics, effective_score_source)
    public_accept_stats = _mapping(public_accept.get("score_stats"))
    score_stats = _mapping(acceptance.get("score_stats"))
    if not public_accept_stats:
        reasons.append("public_accept_score_stats_missing")
    if not score_stats:
        reasons.append("acceptance_score_stats_missing")
    stats_mismatches: list[str] = []
    if public_accept_stats and score_stats:
        for key in ("count", "mean", "std", "standard_error", "eval_protocol_hash"):
            if key in public_accept_stats or key in score_stats:
                if not _values_match(public_accept_stats.get(key), score_stats.get(key)):
                    mismatch = f"acceptance_score_stats_mismatch:{key}"
                    reasons.append(mismatch)
                    stats_mismatches.append(mismatch)

    raw_min_score_count = acceptance.get("min_score_count")
    min_score_count = _positive_int(raw_min_score_count)
    if min_score_count is None:
        reasons.append("min_score_count_invalid")
    public_accept_count = (
        _positive_int(public_accept_stats.get("count"))
        if public_accept_stats
        else None
    )
    acceptance_count = (
        _positive_int(score_stats.get("count")) if score_stats else None
    )
    if public_accept_stats and public_accept_count is None:
        reasons.append("accept_score_count_invalid")
    if score_stats and acceptance_count is None:
        reasons.append("acceptance_score_count_invalid")
    if (
        public_accept_count is None
        or min_score_count is None
        or public_accept_count < min_score_count
    ):
        reasons.append("insufficient_accept_evidence")
    public_mean = _optional_number(public_accept_stats.get("mean")) if public_accept_stats else None
    acceptance_mean = _optional_number(score_stats.get("mean")) if score_stats else None
    if public_accept_stats and public_mean is None:
        reasons.append("accept_score_stats_untrusted:mean")
    if score_stats and acceptance_mean is None:
        reasons.append("acceptance_score_stats_untrusted:mean")
    public_standard_error = (
        _nonnegative_number(public_accept_stats.get("standard_error"))
        if public_accept_stats and "standard_error" in public_accept_stats
        else None
    )
    acceptance_standard_error = (
        _nonnegative_number(score_stats.get("standard_error"))
        if score_stats and "standard_error" in score_stats
        else None
    )
    if (
        min_score_count is not None
        and min_score_count >= 2
        and public_accept_stats
        and public_standard_error is None
    ):
        reasons.append("accept_score_stats_untrusted:standard_error")
    if (
        min_score_count is not None
        and min_score_count >= 2
        and score_stats
        and acceptance_standard_error is None
    ):
        reasons.append("acceptance_score_stats_untrusted:standard_error")

    public_accept_protocol_hash = _protocol_hash(public_accept_stats)
    acceptance_protocol_hash = _protocol_hash(score_stats)
    protocol_binding_valid = bool(
        public_accept_protocol_hash
        and acceptance_protocol_hash
        and public_accept_protocol_hash == acceptance_protocol_hash
    )
    eval_protocol_hash = (
        acceptance_protocol_hash if protocol_binding_valid else None
    )
    if (
        public_accept_stats
        and score_stats
        and (public_accept_protocol_hash is None or acceptance_protocol_hash is None)
    ):
        reasons.append("eval_protocol_hash_missing")

    score_source = declared_score_source
    public_score_binding = _optional_text(
        acceptance.get("public_score_binding")
    )
    absolute_accept_floor = _nonnegative_number(
        acceptance.get("absolute_accept_floor")
    )
    min_delta = _nonnegative_number(acceptance.get("min_delta"))
    acceptance_protocol_hash = _optional_text(
        acceptance.get("acceptance_protocol_hash")
    )
    if score_source is None:
        reasons.append("acceptance_protocol_score_source_invalid")
    allowed_bindings = {
        PUBLIC_SCORE_BINDING_ACCEPTANCE_MEAN,
        PUBLIC_SCORE_BINDING_GUARD_MEAN,
    }
    if public_score_binding not in allowed_bindings:
        reasons.append("acceptance_protocol_public_score_binding_invalid")
    if absolute_accept_floor is None:
        reasons.append("acceptance_protocol_absolute_floor_invalid")
    if min_delta is None:
        reasons.append("acceptance_protocol_min_delta_invalid")
    if acceptance_protocol_hash is None:
        reasons.append("acceptance_protocol_hash_missing")
    acceptance_protocol_binding_valid = bool(
        score_source is not None
        and public_score_binding in allowed_bindings
        and min_score_count is not None
        and absolute_accept_floor is not None
        and min_delta is not None
        and acceptance_protocol_hash is not None
    )
    if acceptance_protocol is not None:
        protocol_mismatches: list[str] = []
        if score_source != acceptance_protocol.score_source:
            protocol_mismatches.append("score_source")
        if public_score_binding != acceptance_protocol.public_score_binding:
            protocol_mismatches.append("public_score_binding")
        if min_score_count != acceptance_protocol.min_score_count:
            protocol_mismatches.append("min_score_count")
        if absolute_accept_floor is None or not math.isclose(
            absolute_accept_floor,
            acceptance_protocol.absolute_accept_floor,
        ):
            protocol_mismatches.append("absolute_accept_floor")
        if min_delta is None or not math.isclose(
            min_delta,
            acceptance_protocol.min_delta,
        ):
            protocol_mismatches.append("min_delta")
        if acceptance_protocol_hash != acceptance_protocol.protocol_hash:
            protocol_mismatches.append("acceptance_protocol_hash")
        if acceptance_protocol.uses_guard_score:
            if acceptance.get("guard_score_source") != acceptance_protocol.guard_score_source:
                protocol_mismatches.append("guard_score_source")
            if not _numbers_close(
                acceptance.get("guard_absolute_accept_floor"),
                acceptance_protocol.guard_absolute_accept_floor,
            ):
                protocol_mismatches.append("guard_absolute_accept_floor")
            if not _numbers_close(
                acceptance.get("guard_max_regression"),
                acceptance_protocol.guard_max_regression,
            ):
                protocol_mismatches.append("guard_max_regression")
        if acceptance_protocol.requires_paired_guard_score_samples:
            if not _numbers_close(
                acceptance.get("guard_regression_margin"),
                acceptance_protocol.guard_regression_margin,
            ):
                protocol_mismatches.append("guard_regression_margin")
            if not _numbers_close(
                acceptance.get("max_public_to_sealed_score_drop"),
                acceptance_protocol.max_public_to_sealed_score_drop,
            ):
                protocol_mismatches.append("max_public_to_sealed_score_drop")
        if acceptance_protocol.uses_named_guards:
            if not _numbers_close(
                acceptance.get("max_public_to_sealed_score_drop"),
                acceptance_protocol.max_public_to_sealed_score_drop,
            ):
                protocol_mismatches.append("max_public_to_sealed_score_drop")
        reasons.extend(
            f"acceptance_protocol_mismatch:{field}"
            for field in protocol_mismatches
        )
        acceptance_protocol_binding_valid = not protocol_mismatches and acceptance_protocol_binding_valid

    score_samples: tuple[float, ...] = ()
    score_samples_valid = True
    if (
        acceptance_protocol is not None
        and acceptance_protocol.requires_paired_score_samples
    ):
        try:
            if public_mean is None:
                raise ValueError("paired score mean is missing")
            score_samples = validated_score_samples(
                acceptance.get("score_samples"),
                expected_mean=public_mean,
                protocol=acceptance_protocol,
                field_name="acceptance.score_samples",
            )
        except ValueError:
            score_samples_valid = False
        else:
            score_samples_valid = (
                public_accept_count == acceptance_protocol.min_score_count
            )
            if acceptance_protocol.requires_paired_guard_score_samples:
                try:
                    source_score_samples = validated_score_samples(
                        public_accept.get("seed_macro_scores"),
                        expected_mean=public_mean,
                        protocol=acceptance_protocol,
                        field_name="acceptance score source seed_macro_scores",
                    )
                except ValueError:
                    score_samples_valid = False
                else:
                    score_samples_valid = score_samples == source_score_samples
        if not score_samples_valid:
            reasons.append("acceptance_score_samples_untrusted")

    score_stats_binding_valid = bool(
        public_accept_stats
        and score_stats
        and not stats_mismatches
        and public_accept_count is not None
        and acceptance_count is not None
        and public_mean is not None
        and acceptance_mean is not None
        and protocol_binding_valid
        and score_samples_valid
        and (
            min_score_count is not None
            and (
                min_score_count < 2
                or (
                    public_standard_error is not None
                    and acceptance_standard_error is not None
                )
            )
        )
    )
    canonical_score_stats: dict[str, Any] = {}
    if score_stats_binding_valid:
        canonical_score_stats = {
            "count": public_accept_count,
            "mean": public_mean,
            "eval_protocol_hash": eval_protocol_hash,
        }
        public_std = _nonnegative_number(public_accept_stats.get("std"))
        if public_std is not None:
            canonical_score_stats["std"] = public_std
        if public_standard_error is not None:
            canonical_score_stats["standard_error"] = public_standard_error
        if score_samples:
            canonical_score_stats["score_samples"] = list(score_samples)

    guard_score_stats: dict[str, Any] = {}
    guard_score_mean: float | None = None
    guard_standard_error: float | None = None
    guard_score_samples: tuple[float, ...] = ()
    guard_score_source: str | None = None
    guard_absolute_accept_floor: float | None = None
    guard_max_regression: float | None = None
    guard_regression_margin: float | None = None
    max_public_to_sealed_score_drop: float | None = None
    guard_score_binding_valid = not (
        acceptance_protocol is not None and acceptance_protocol.uses_guard_score
    )
    if acceptance_protocol is not None and acceptance_protocol.uses_guard_score:
        guard_score_source = _optional_text(acceptance.get("guard_score_source"))
        guard_absolute_accept_floor = _nonnegative_number(
            acceptance.get("guard_absolute_accept_floor")
        )
        guard_max_regression = _nonnegative_number(
            acceptance.get("guard_max_regression")
        )
        guard_regression_margin = _nonnegative_number(
            acceptance.get("guard_regression_margin")
        )
        max_public_to_sealed_score_drop = _nonnegative_number(
            acceptance.get("max_public_to_sealed_score_drop")
        )
        guard_source_summary = _score_summary_from_source(
            metrics,
            guard_score_source,
        )
        guard_declared_stats = _mapping(acceptance.get("guard_score_stats"))
        guard_validation = _validate_guard_score_evidence(
            source_stats=_mapping(guard_source_summary.get("score_stats")),
            declared_stats=guard_declared_stats,
            minimum_count=acceptance_protocol.min_score_count,
            expected_eval_protocol_hash=eval_protocol_hash,
        )
        reasons.extend(guard_validation["reasons"])
        guard_score_stats = dict(guard_validation["score_stats"])
        guard_score_mean = guard_validation["mean"]
        guard_standard_error = guard_validation["standard_error"]
        guard_samples_valid = not acceptance_protocol.requires_paired_guard_score_samples
        if acceptance_protocol.requires_paired_guard_score_samples:
            try:
                if guard_score_mean is None:
                    raise ValueError("paired guard score mean is missing")
                guard_score_samples = validated_score_samples(
                    acceptance.get("guard_score_samples"),
                    expected_mean=guard_score_mean,
                    protocol=acceptance_protocol,
                    field_name="acceptance.guard_score_samples",
                )
                source_guard_samples = validated_score_samples(
                    guard_source_summary.get("seed_macro_scores"),
                    expected_mean=guard_score_mean,
                    protocol=acceptance_protocol,
                    field_name="guard score source seed_macro_scores",
                )
                guard_samples_valid = guard_score_samples == source_guard_samples
            except ValueError:
                guard_samples_valid = False
            if not guard_samples_valid:
                reasons.append("acceptance_guard_score_samples_untrusted")
            elif guard_score_stats:
                guard_score_stats["score_samples"] = list(guard_score_samples)
        guard_score_binding_valid = bool(
            guard_validation["valid"]
            and guard_samples_valid
            and guard_score_source == acceptance_protocol.guard_score_source
            and _numbers_close(
                guard_absolute_accept_floor,
                acceptance_protocol.guard_absolute_accept_floor,
            )
            and _numbers_close(
                guard_max_regression,
                acceptance_protocol.guard_max_regression,
            )
            and (
                not acceptance_protocol.requires_paired_guard_score_samples
                or (
                    _numbers_close(
                        guard_regression_margin,
                        acceptance_protocol.guard_regression_margin,
                    )
                    and _numbers_close(
                        max_public_to_sealed_score_drop,
                        acceptance_protocol.max_public_to_sealed_score_drop,
                    )
                )
            )
        )
        if not guard_score_binding_valid:
            reasons.append("acceptance_guard_score_untrusted")
        acceptance_protocol_binding_valid = bool(
            acceptance_protocol_binding_valid and guard_score_binding_valid
        )

    named_guard_evidence: dict[str, Any] = {}
    named_guard_binding_valid = not (
        acceptance_protocol is not None and acceptance_protocol.uses_named_guards
    )
    if acceptance_protocol is not None and acceptance_protocol.uses_named_guards:
        declared_named_guards = _mapping(acceptance.get("named_guards"))
        expected_names = {guard.name for guard in acceptance_protocol.named_guards}
        if set(declared_named_guards) != expected_names:
            reasons.append("acceptance_named_guard_set_mismatch")
        named_guard_binding_valid = set(declared_named_guards) == expected_names
        for guard in acceptance_protocol.named_guards:
            declared = _mapping(declared_named_guards.get(guard.name))
            source_summary = _score_summary_from_source(metrics, guard.score_source)
            validation = _validate_guard_score_evidence(
                source_stats=_mapping(source_summary.get("score_stats")),
                declared_stats=_mapping(declared.get("score_stats")),
                minimum_count=acceptance_protocol.min_score_count,
                expected_eval_protocol_hash=eval_protocol_hash,
            )
            guard_reasons = [
                f"named_guard:{guard.name}:{reason}"
                for reason in validation["reasons"]
            ]
            reasons.extend(guard_reasons)
            fields_valid = bool(
                declared.get("score_source") == guard.score_source
                and _numbers_close(
                    declared.get("absolute_accept_floor"),
                    guard.absolute_accept_floor,
                )
                and _numbers_close(
                    declared.get("max_regression"), guard.max_regression
                )
                and _numbers_close(
                    declared.get("regression_margin"), guard.regression_margin
                )
            )
            if not fields_valid:
                reasons.append(f"named_guard:{guard.name}:contract_mismatch")
            samples: tuple[float, ...] = ()
            samples_valid = False
            try:
                mean = validation["mean"]
                if mean is None:
                    raise ValueError("named guard mean is missing")
                samples = validated_score_samples(
                    declared.get("score_samples"),
                    expected_mean=mean,
                    protocol=acceptance_protocol,
                    field_name=f"acceptance.named_guards.{guard.name}.score_samples",
                )
                source_samples = validated_score_samples(
                    source_summary.get("seed_macro_scores"),
                    expected_mean=mean,
                    protocol=acceptance_protocol,
                    field_name=f"named guard source samples:{guard.name}",
                )
                samples_valid = samples == source_samples
            except ValueError:
                samples_valid = False
            if not samples_valid:
                reasons.append(f"named_guard:{guard.name}:score_samples_untrusted")
            item_valid = bool(validation["valid"] and fields_valid and samples_valid)
            named_guard_binding_valid = named_guard_binding_valid and item_valid
            if item_valid:
                named_guard_evidence[guard.name] = {
                    **guard.to_public_dict(),
                    "score_stats": {
                        **dict(validation["score_stats"]),
                        "score_samples": list(samples),
                    },
                    "score_samples": list(samples),
                }
        if not named_guard_binding_valid:
            reasons.append("acceptance_named_guards_untrusted")
        acceptance_protocol_binding_valid = bool(
            acceptance_protocol_binding_valid and named_guard_binding_valid
        )

    public_score_binding_valid = False
    canonical_public_score: float | None = None
    if public_score is not _PUBLIC_SCORE_UNSET:
        try:
            parsed_public_score = validate_finite_number(
                public_score,
                field_name="public_result.public_score",
            )
        except ValueError:
            reasons.append("public_score_invalid")
        else:
            binding_mean = (
                guard_score_mean
                if public_score_binding == PUBLIC_SCORE_BINDING_GUARD_MEAN
                else public_mean
            )
            binding_ready = (
                guard_score_binding_valid
                if public_score_binding == PUBLIC_SCORE_BINDING_GUARD_MEAN
                else score_stats_binding_valid
            )
            if binding_ready and binding_mean is not None:
                if math.isclose(
                    parsed_public_score,
                    binding_mean,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    public_score_binding_valid = True
                    canonical_public_score = binding_mean
                else:
                    reasons.append("public_score_binding_mismatch")

    gate_statuses: dict[str, str] = {}
    failed_required_gates: list[str] = []
    sealed_validation_status = "missing"
    sealed_validation_valid = False
    sealed_validation_complete = False
    sealed_validation_evidence = _mapping(
        metrics.get(SEALED_VALIDATION_EVIDENCE_KEY)
    )
    for gate_name in required_gates:
        if gate_name == "sealed_validation_gate":
            validation = validate_sealed_validation_gate_contract(
                metrics.get(gate_name),
                rollout_evidence=sealed_validation_evidence,
                eval_protocol_hash=eval_protocol_hash,
                expected_seed_count=expected_sealed_seed_count,
                expected_checkpoint_hash=expected_checkpoint_hash,
            )
            status = validation.status
            sealed_validation_status = validation.status
            sealed_validation_valid = validation.valid
            sealed_validation_complete = validate_sealed_validation_gate_evidence(
                metrics.get(gate_name),
                rollout_evidence=sealed_validation_evidence,
                eval_protocol_hash=eval_protocol_hash,
                expected_seed_count=expected_sealed_seed_count,
                expected_checkpoint_hash=expected_checkpoint_hash,
            ).valid
            reasons.extend(f"sealed_validation_gate_invalid:{reason}" for reason in validation.reasons)
        else:
            status = _gate_status(_mapping(metrics.get(gate_name)))
        gate_statuses[gate_name] = status
        if status != "pass":
            failed_required_gates.append(gate_name)
            reasons.append(f"required_gate_not_pass:{gate_name}")

    missing_consumed_gates: tuple[str, ...] = ()
    gate_status_mismatches: tuple[str, ...] = ()
    if decision is not None:
        consumed_gates = _string_set(decision.get("consumed_gates"))
        missing_consumed = sorted(required_gate_set - consumed_gates)
        missing_consumed_gates = tuple(missing_consumed)
        reasons.extend(f"required_gate_not_consumed:{gate}" for gate in missing_consumed)
        decision_gate_statuses = _string_mapping(decision.get("gate_statuses"))
        mismatches = sorted(
            gate
            for gate, actual_status in gate_statuses.items()
            if gate in consumed_gates
            and decision_gate_statuses.get(gate, "missing") != actual_status
        )
        gate_status_mismatches = tuple(mismatches)
        reasons.extend(f"decision_gate_status_mismatch:{gate}" for gate in mismatches)

    deduped_reasons = tuple(dict.fromkeys(reasons))
    reject_reason = _canonical_reject_reason(
        deduped_reasons,
        missing_common=missing_common,
        failed_required_gates=tuple(failed_required_gates),
        missing_required_score=acceptance.get("missing_required_score") is True,
    )
    return AcceptanceEvidenceValidation(
        valid=not deduped_reasons,
        reasons=deduped_reasons,
        required_gates=required_gates,
        gate_statuses=gate_statuses,
        missing_common_gates=missing_common,
        failed_required_gates=tuple(failed_required_gates),
        missing_consumed_gates=missing_consumed_gates,
        gate_status_mismatches=gate_status_mismatches,
        eval_protocol_hash=eval_protocol_hash,
        protocol_binding_valid=protocol_binding_valid,
        score_stats=canonical_score_stats,
        score_stats_binding_valid=score_stats_binding_valid,
        score_count=public_accept_count if score_stats_binding_valid else None,
        min_score_count=min_score_count,
        score_mean=public_mean if score_stats_binding_valid else None,
        standard_error=(
            public_standard_error if score_stats_binding_valid else None
        ),
        score_samples=(score_samples if score_stats_binding_valid else ()),
        guard_score_stats=guard_score_stats,
        guard_score_mean=guard_score_mean,
        guard_standard_error=guard_standard_error,
        guard_score_samples=(
            guard_score_samples if guard_score_binding_valid else ()
        ),
        guard_score_source=guard_score_source,
        guard_absolute_accept_floor=guard_absolute_accept_floor,
        guard_max_regression=guard_max_regression,
        guard_regression_margin=guard_regression_margin,
        max_public_to_sealed_score_drop=max_public_to_sealed_score_drop,
        guard_score_binding_valid=guard_score_binding_valid,
        named_guard_evidence=named_guard_evidence,
        named_guard_binding_valid=named_guard_binding_valid,
        score_source=score_source,
        public_score_binding=public_score_binding,
        public_score_binding_valid=public_score_binding_valid,
        canonical_public_score=canonical_public_score,
        absolute_accept_floor=absolute_accept_floor,
        min_delta=min_delta,
        acceptance_protocol_hash=acceptance_protocol_hash,
        acceptance_protocol_binding_valid=acceptance_protocol_binding_valid,
        sealed_validation_status=sealed_validation_status,
        sealed_validation_valid=sealed_validation_valid,
        sealed_validation_complete=sealed_validation_complete,
        sealed_validation_evidence=dict(sealed_validation_evidence),
        sealed_validation_gate=dict(_mapping(metrics.get("sealed_validation_gate"))),
        reject_reason=reject_reason,
    )


def _common_required_gates(evidence_subject_kind: str) -> tuple[str, ...]:
    if evidence_subject_kind == "candidate":
        return COMMON_ACCEPT_REQUIRED_GATES
    if evidence_subject_kind == "reference":
        return COMMON_ACCEPT_REQUIRED_GATES
    if evidence_subject_kind == "baseline":
        return BASELINE_ACCEPT_REQUIRED_GATES
    raise ValueError(
        "evidence_subject_kind must be candidate, reference, or baseline"
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _score_summary_from_source(
    metrics: Mapping[str, Any],
    source: str | None,
) -> Mapping[str, Any]:
    if not isinstance(source, str):
        return {}
    parts = source.split(".")
    if (
        len(parts) < 3
        or parts[-2:] != ["score_stats", "mean"]
        or any(not part for part in parts)
    ):
        return {}
    current: Any = metrics
    for part in parts[:-2]:
        current = _mapping(current).get(part)
        if not isinstance(current, Mapping):
            return {}
    return current


def _validate_guard_score_evidence(
    *,
    source_stats: Mapping[str, Any],
    declared_stats: Mapping[str, Any],
    minimum_count: int,
    expected_eval_protocol_hash: str | None,
) -> dict[str, Any]:
    reasons: list[str] = []
    if not source_stats:
        reasons.append("guard_score_stats_missing")
    if not declared_stats:
        reasons.append("acceptance_guard_score_stats_missing")
    for key in ("count", "mean", "std", "standard_error", "eval_protocol_hash"):
        if source_stats or declared_stats:
            if not _values_match(source_stats.get(key), declared_stats.get(key)):
                reasons.append(f"acceptance_guard_score_stats_mismatch:{key}")
    count = _positive_int(source_stats.get("count"))
    mean = _optional_number(source_stats.get("mean"))
    std = _nonnegative_number(source_stats.get("std"))
    standard_error = _nonnegative_number(source_stats.get("standard_error"))
    protocol_hash = _protocol_hash(source_stats)
    if count is None or count < minimum_count:
        reasons.append("insufficient_guard_evidence")
    if mean is None or std is None or standard_error is None:
        reasons.append("guard_score_stats_untrusted")
    if protocol_hash is None or protocol_hash != expected_eval_protocol_hash:
        reasons.append("guard_eval_protocol_hash_mismatch")
    valid = not reasons
    canonical: dict[str, Any] = {}
    if valid:
        canonical = {
            "count": count,
            "mean": mean,
            "std": std,
            "standard_error": standard_error,
            "eval_protocol_hash": protocol_hash,
        }
    return {
        "valid": valid,
        "reasons": reasons,
        "score_stats": canonical,
        "mean": mean if valid else None,
        "standard_error": standard_error if valid else None,
    }


def _numbers_close(left: Any, right: Any) -> bool:
    left_number = _optional_number(left)
    right_number = _optional_number(right)
    return bool(
        left_number is not None
        and right_number is not None
        and math.isclose(left_number, right_number)
    )


def _candidate_contamination_reasons(metrics: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    for key in CANDIDATE_CONTAMINATION_FIELDS:
        if key not in metrics:
            continue
        value = metrics[key]
        if not isinstance(value, bool):
            reasons.append(f"candidate_contamination_flag_invalid:{key}")
        elif value:
            reasons.append(f"candidate_contamination_detected:{key}")
    return reasons


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return []
    return [str(item) for item in value if str(item)]


def _string_set(value: Any) -> set[str]:
    return set(_string_list(value))


def _string_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, str] = {}
    for key, raw_status in value.items():
        if isinstance(raw_status, str):
            result[str(key)] = raw_status
    return result


def _gate_status(gate: Mapping[str, Any]) -> str:
    status = gate.get("status")
    return status if isinstance(status, str) and status else "missing"


def _values_match(left: Any, right: Any) -> bool:
    left_number = _optional_number(left)
    right_number = _optional_number(right)
    if left_number is not None or right_number is not None:
        return left_number is not None and right_number is not None and math.isclose(left_number, right_number)
    return left == right


def _protocol_hash(value: Mapping[str, Any]) -> str | None:
    raw = value.get("eval_protocol_hash")
    return raw if isinstance(raw, str) and raw else None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _positive_int(value: Any) -> int | None:
    parsed = _optional_int(value)
    return parsed if parsed is not None and parsed > 0 else None


def _optional_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _nonnegative_number(value: Any) -> float | None:
    parsed = _optional_number(value)
    return parsed if parsed is not None and parsed >= 0.0 else None


def _optional_text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _canonical_reject_reason(
    reasons: tuple[str, ...],
    *,
    missing_common: tuple[str, ...],
    failed_required_gates: tuple[str, ...],
    missing_required_score: bool,
) -> str | None:
    def has(*prefixes: str) -> bool:
        return any(item.startswith(prefix) for item in reasons for prefix in prefixes)

    if has("candidate_contamination_detected", "candidate_contamination_flag_invalid"):
        return "candidate_contamination_detected"
    if has("acceptance_protocol_"):
        return "acceptance_protocol_mismatch"
    if has("public_score_binding_mismatch", "public_score_invalid"):
        return "public_score_binding_mismatch"
    if missing_required_score or has(
        "public_accept_score_stats_missing",
        "acceptance_score_stats_missing",
        "guard_score_stats_missing",
        "acceptance_guard_score_stats_missing",
        "named_guard:",
    ):
        return "required_score_missing"
    if has(
        "min_score_count_invalid",
        "accept_score_count_invalid",
        "acceptance_score_count_invalid",
        "insufficient_accept_evidence",
        "insufficient_guard_evidence",
    ):
        return "insufficient_accept_evidence"
    if has("eval_protocol_hash_missing"):
        return "eval_protocol_hash_missing"
    if has(
        "acceptance_score_stats_mismatch",
        "accept_score_stats_untrusted",
        "acceptance_score_stats_untrusted",
        "acceptance_score_samples_untrusted",
        "acceptance_guard_score_stats_mismatch",
        "guard_score_stats_untrusted",
        "guard_eval_protocol_hash_mismatch",
        "acceptance_guard_score_untrusted",
    ):
        return "accept_score_stats_untrusted"
    if "sealed_validation_gate" in missing_common:
        return "sealed_validation_gate_not_required"
    if missing_common:
        return "baseline_accept_gate_not_required"
    if failed_required_gates:
        return "required_gate_failed"
    return None
