"""Sequential, task-neutral replay of acceptance decisions in ledger order."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from true_rl_agent.agent_core.ledger import (
    NON_PROVIDER_AUTHORING_MODE,
    PROVIDER_AUTHORING_MODE,
    dispatch_ledger_record_schema,
    is_strict_ledger_record,
    primary_ledger_epoch_hash,
    require_ledger_authoring_mode,
    require_ledger_record_epoch_hash,
)
from true_rl_agent.evaluation.acceptance_phase import acceptance_comparison_entered
from true_rl_agent.evaluation.acceptance_protocol import (
    NOISE_AWARE_DELTA_RULE,
    AcceptanceProtocol,
    acceptance_delta_met,
    acceptance_delta_reason,
    acceptance_threshold,
    guard_regression_threshold,
    named_guard_regression_threshold,
    threshold_components_match,
)
from true_rl_agent.evaluation.binding import require_sha256_digest
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.evaluation.decision_record_contract import (
    DecisionRecordValidation,
    project_validated_decision,
    validate_decision_record,
)
from true_rl_agent.evaluation.round_stage_contract import EVALUATED_STAGE
from true_rl_agent.prompt_artifacts import validate_ledger_prompt_artifacts


@dataclass(frozen=True)
class GenerationNoHeadTransitionValidation:
    """Valid replay marker for a closed generation with no formal round."""

    valid: bool = True
    reasons: tuple[str, ...] = ()
    accepted: bool = False


@dataclass(frozen=True)
class AcceptanceReplayEntry:
    round_id: str
    validation: DecisionRecordValidation | GenerationNoHeadTransitionValidation
    sequence_reasons: tuple[str, ...]
    public_decision: Mapping[str, Any]
    current_best_before: CurrentBestEvalStats
    current_best_after: CurrentBestEvalStats

    @property
    def valid(self) -> bool:
        return self.validation.valid and not self.sequence_reasons

    @property
    def valid_policy_accept(self) -> bool:
        return self.valid and self.public_decision.get("accepted") is True


@dataclass(frozen=True)
class AcceptanceReplayResult:
    entries: tuple[AcceptanceReplayEntry, ...]
    current_best: CurrentBestEvalStats

    @property
    def invalid_declared_accepts(self) -> tuple[AcceptanceReplayEntry, ...]:
        return tuple(
            entry
            for entry in self.entries
            if entry.validation.accepted and not entry.valid_policy_accept
        )

    @property
    def invalid_session_bindings(self) -> tuple[AcceptanceReplayEntry, ...]:
        return tuple(
            entry
            for entry in self.entries
            if any(
                _is_session_binding_reason(reason)
                for reason in (
                    *entry.validation.reasons,
                    *entry.sequence_reasons,
                )
            )
        )

    def raise_for_invalid_declared_accepts(self) -> None:
        self._raise_entries(
            self.invalid_declared_accepts,
            "ledger contains invalid declared acceptance; refusing continuation",
        )

    def raise_for_invalid_session_bindings(self) -> None:
        self._raise_entries(
            self.invalid_session_bindings,
            "ledger contains invalid session binding; refusing continuation",
        )

    def raise_for_continuation(self) -> None:
        self.raise_for_invalid_session_bindings()
        self.raise_for_invalid_declared_accepts()

    @staticmethod
    def _raise_entries(
        entries: Sequence[AcceptanceReplayEntry],
        message: str,
    ) -> None:
        if not entries:
            return
        details: list[str] = []
        for entry in entries[:8]:
            reasons = tuple(
                dict.fromkeys(
                    (*entry.validation.reasons, *entry.sequence_reasons)
                )
            )
            details.append(
                f"{entry.round_id}({','.join(reasons[:8]) or 'invalid_record'})"
            )
        if len(entries) > len(details):
            details.append(f"+{len(entries) - len(details)} more")
        raise ValueError(message + ": " + "; ".join(details))


@dataclass(frozen=True)
class ProjectedPublicHistory:
    """Typed collection boundary for sequentially validated public decisions."""

    records: tuple[dict[str, Any], ...]
    initial_current_best: CurrentBestEvalStats | None
    session_genesis_hash: str | None
    acceptance_protocol_hash: str | None
    current_best: CurrentBestEvalStats | None = None
    public_score_source: str | None = None


def project_public_history(
    records: Sequence[Mapping[str, Any]],
    *,
    initial_current_best: CurrentBestEvalStats | None = None,
    expected_session_genesis_hash: str | None = None,
    acceptance_protocol: AcceptanceProtocol | None,
    expected_initial_policy_state_hash: str | None = None,
    expected_authoring_mode: str = NON_PROVIDER_AUTHORING_MODE,
    expected_ledger_epoch_hash: str | None = None,
) -> ProjectedPublicHistory:
    """Project a full ordered history once before any public consumer sees it."""

    has_evaluated_history = any(
        (
            isinstance(record.get("public_result"), Mapping)
            and bool(record.get("public_result"))
        )
        or dispatch_ledger_record_schema(record).record_kind == "generation"
        for record in records
    )
    if acceptance_protocol is None:
        if has_evaluated_history:
            raise ValueError(
                "acceptance_protocol is required to project evaluated history"
            )
        projected = tuple(
            {
                **dict(record),
                "decision": project_validated_decision(record),
            }
            for record in records
        )
        return ProjectedPublicHistory(projected, None, None, None, None, None)
    _require_replay_anchor(
        initial_current_best=initial_current_best,
        expected_session_genesis_hash=expected_session_genesis_hash,
        acceptance_protocol=acceptance_protocol,
        expected_initial_policy_state_hash=expected_initial_policy_state_hash,
    )
    assert initial_current_best is not None
    assert expected_session_genesis_hash is not None
    replay = replay_acceptance_ledger(
        records,
        initial_current_best=initial_current_best,
        expected_session_genesis_hash=expected_session_genesis_hash,
        acceptance_protocol=acceptance_protocol,
        expected_initial_policy_state_hash=expected_initial_policy_state_hash,
        expected_authoring_mode=expected_authoring_mode,
        expected_ledger_epoch_hash=expected_ledger_epoch_hash,
    )
    replay.raise_for_invalid_session_bindings()
    projected = tuple(
        _project_public_history_record(
            record,
            entry=entry,
            acceptance_protocol=acceptance_protocol,
        )
        for record, entry in zip(records, replay.entries, strict=True)
    )
    return ProjectedPublicHistory(
        projected,
        initial_current_best,
        expected_session_genesis_hash,
        acceptance_protocol.protocol_hash,
        replay.current_best,
        acceptance_protocol.public_score_source,
    )


def project_replayed_records(
    records: Sequence[Mapping[str, Any]],
    *,
    initial_current_best: CurrentBestEvalStats,
    expected_session_genesis_hash: str,
    acceptance_protocol: AcceptanceProtocol,
    expected_initial_policy_state_hash: str | None = None,
    expected_authoring_mode: str = NON_PROVIDER_AUTHORING_MODE,
    expected_ledger_epoch_hash: str | None = None,
) -> tuple[dict[str, Any], ...]:
    replay = replay_acceptance_ledger(
        records,
        initial_current_best=initial_current_best,
        expected_session_genesis_hash=expected_session_genesis_hash,
        acceptance_protocol=acceptance_protocol,
        expected_initial_policy_state_hash=expected_initial_policy_state_hash,
        expected_authoring_mode=expected_authoring_mode,
        expected_ledger_epoch_hash=expected_ledger_epoch_hash,
    )
    replay.raise_for_invalid_session_bindings()
    return tuple(
        _project_public_history_record(
            record,
            entry=entry,
            acceptance_protocol=acceptance_protocol,
        )
        for record, entry in zip(records, replay.entries, strict=True)
    )


def _project_public_history_record(
    record: Mapping[str, Any],
    *,
    entry: AcceptanceReplayEntry,
    acceptance_protocol: AcceptanceProtocol,
) -> dict[str, Any]:
    if dispatch_ledger_record_schema(record).record_kind != "generation":
        return {**dict(record), "decision": dict(entry.public_decision)}
    from true_rl_agent.agent_core.generation_record import (
        project_generation_public_history_record,
    )

    return project_generation_public_history_record(
        record,
        public_decision=entry.public_decision,
        acceptance_protocol=acceptance_protocol,
        expected_parent=entry.current_best_before,
    )


def replay_acceptance_ledger(
    records: Sequence[Mapping[str, Any]],
    *,
    initial_current_best: CurrentBestEvalStats,
    expected_session_genesis_hash: str,
    acceptance_protocol: AcceptanceProtocol,
    expected_initial_policy_state_hash: str | None = None,
    expected_authoring_mode: str = NON_PROVIDER_AUTHORING_MODE,
    expected_ledger_epoch_hash: str | None = None,
) -> AcceptanceReplayResult:
    """Replay records from one verified, full session anchor."""

    _require_replay_anchor(
        initial_current_best=initial_current_best,
        expected_session_genesis_hash=expected_session_genesis_hash,
        acceptance_protocol=acceptance_protocol,
        expected_initial_policy_state_hash=expected_initial_policy_state_hash,
    )
    record_kinds = {
        dispatch_ledger_record_schema(record).record_kind for record in records
    }
    if len(record_kinds) > 1:
        raise ValueError("acceptance replay cannot mix round and generation records")
    if record_kinds == {"generation"}:
        return _replay_generation_ledger(
            records,
            initial_current_best=initial_current_best,
            expected_session_genesis_hash=expected_session_genesis_hash,
            acceptance_protocol=acceptance_protocol,
            expected_initial_policy_state_hash=expected_initial_policy_state_hash,
            expected_authoring_mode=expected_authoring_mode,
            expected_ledger_epoch_hash=expected_ledger_epoch_hash,
        )
    expected_binding = initial_current_best.evaluation_binding()
    authoring_mode = require_ledger_authoring_mode(expected_authoring_mode)
    legacy_unbound_epoch_allowed = expected_ledger_epoch_hash is None
    ledger_epoch_hash = (
        primary_ledger_epoch_hash(expected_session_genesis_hash)
        if expected_ledger_epoch_hash is None
        else require_sha256_digest(
            expected_ledger_epoch_hash,
            "expected_ledger_epoch_hash",
        )
    )
    current_best = initial_current_best
    entries: list[AcceptanceReplayEntry] = []
    for record in records:
        before = current_best
        validation = validate_decision_record(
            record,
            acceptance_protocol=acceptance_protocol,
            expected_session_genesis_hash=expected_session_genesis_hash,
            expected_evaluation_binding=expected_binding,
            expected_initial_policy_state_hash=(
                expected_initial_policy_state_hash
            ),
        )
        sequence_reasons = tuple(
            dict.fromkeys(
                (
                    *validate_ledger_prompt_artifacts(
                        record,
                        expected_authoring_mode=authoring_mode,
                    ),
                    *_ledger_epoch_reasons(
                        record,
                        expected_ledger_epoch_hash=ledger_epoch_hash,
                        legacy_unbound_epoch_allowed=(
                            legacy_unbound_epoch_allowed
                        ),
                    ),
                    *_source_parent_reasons(record, current_best=before),
                    *_sequence_reasons(
                        record,
                        validation=validation,
                        current_best=before,
                        acceptance_protocol=acceptance_protocol,
                    ),
                )
            )
        )
        public_decision = project_validated_decision(
            record,
            validation=validation,
            acceptance_protocol=acceptance_protocol,
            extra_validation_reasons=sequence_reasons,
        )
        if validation.valid_policy_accept and not sequence_reasons:
            eval_job = _mapping(_mapping(record.get("jobs")).get("eval"))
            checkpoint = _mapping(eval_job.get("checkpoint"))
            subject = _mapping(eval_job.get("subject"))
            try:
                restored = CurrentBestEvalStats.from_public_result_mapping(
                    round_id=_text(record.get("round_id")),
                    candidate_id=_text(subject.get("id")),
                    public_result=_mapping(record.get("public_result")),
                    acceptance_protocol=acceptance_protocol,
                    subject_manifest_hash=_text(
                        eval_job.get("subject_manifest_hash")
                    ),
                    program_source_checksum=_text(
                        eval_job.get("program_source_checksum")
                    ),
                    training_binding_hash=_text(
                        eval_job.get("training_binding_hash")
                    ),
                    capability_profile_hash=_text(
                        eval_job.get("capability_profile_hash")
                    ),
                    checkpoint_hash=_text(checkpoint.get("checksum")) or None,
                    expected_evaluation_binding=expected_binding,
                )
            except (TypeError, ValueError):
                restored = None
            if restored is not None and restored.mode == "accepted":
                current_best = restored
            else:
                sequence_reasons = (
                    "acceptance_replay_restored_best_incomplete",
                )
                public_decision = project_validated_decision(
                    record,
                    validation=validation,
                    acceptance_protocol=acceptance_protocol,
                    extra_validation_reasons=sequence_reasons,
                )
        entries.append(
            AcceptanceReplayEntry(
                round_id=_text(record.get("round_id")) or "<unknown-round>",
                validation=validation,
                sequence_reasons=sequence_reasons,
                public_decision=public_decision,
                current_best_before=before,
                current_best_after=current_best,
            )
        )
    return AcceptanceReplayResult(tuple(entries), current_best)


def _replay_generation_ledger(
    records: Sequence[Mapping[str, Any]],
    *,
    initial_current_best: CurrentBestEvalStats,
    expected_session_genesis_hash: str,
    acceptance_protocol: AcceptanceProtocol,
    expected_initial_policy_state_hash: str | None,
    expected_authoring_mode: str,
    expected_ledger_epoch_hash: str | None,
) -> AcceptanceReplayResult:
    """Replay each generation through its sole optional P2 round projection."""

    from true_rl_agent.agent_core.generation_record import (
        project_generation_formal_round_record,
        validate_generation_ledger_record,
    )

    if require_ledger_authoring_mode(expected_authoring_mode) != PROVIDER_AUTHORING_MODE:
        raise ValueError("generation replay requires provider authoring mode")
    ledger_epoch_hash = (
        primary_ledger_epoch_hash(expected_session_genesis_hash)
        if expected_ledger_epoch_hash is None
        else require_sha256_digest(
            expected_ledger_epoch_hash,
            "expected_ledger_epoch_hash",
        )
    )
    current = initial_current_best
    entries: list[AcceptanceReplayEntry] = []
    generation_session_contract_hash: str | None = None
    for expected_index, record in enumerate(records, start=1):
        if record.get("generation_index") != expected_index:
            raise ValueError("generation replay index sequence mismatch")
        if record.get("session_genesis_hash") != expected_session_genesis_hash:
            raise ValueError("generation replay Session Genesis mismatch")
        if record.get("ledger_epoch_hash") != ledger_epoch_hash:
            raise ValueError("generation replay ledger epoch mismatch")
        record_contract_hash = require_sha256_digest(
            record.get("generation_session_contract_hash"),
            "generation replay session contract hash",
        )
        if generation_session_contract_hash is None:
            generation_session_contract_hash = record_contract_hash
        elif record_contract_hash != generation_session_contract_hash:
            raise ValueError("generation session contract changed within the ledger")
        before = current
        current = validate_generation_ledger_record(
            record,
            acceptance_protocol=acceptance_protocol,
            expected_parent=before,
            expected_generation_session_contract_hash=(
                generation_session_contract_hash
            ),
        )
        formal = project_generation_formal_round_record(
            record,
            acceptance_protocol=acceptance_protocol,
            expected_parent=before,
        )
        if formal is None:
            validation: (
                DecisionRecordValidation | GenerationNoHeadTransitionValidation
            ) = GenerationNoHeadTransitionValidation()
            public_decision = {
                "outcome": "generation_head_retained",
                "accepted": False,
                "reason": record.get("generation_outcome"),
                "generation_id": record.get("generation_id"),
                "acceptance_authority": False,
            }
            sequence_reasons: tuple[str, ...] = ()
        else:
            nested = replay_acceptance_ledger(
                [formal],
                initial_current_best=before,
                expected_session_genesis_hash=expected_session_genesis_hash,
                acceptance_protocol=acceptance_protocol,
                expected_initial_policy_state_hash=(
                    expected_initial_policy_state_hash
                ),
                expected_authoring_mode=expected_authoring_mode,
                expected_ledger_epoch_hash=ledger_epoch_hash,
            ).entries[0]
            validation = nested.validation
            public_decision = {
                **dict(nested.public_decision),
                "generation_id": record.get("generation_id"),
            }
            sequence_reasons = nested.sequence_reasons
        entries.append(
            AcceptanceReplayEntry(
                round_id=str(record.get("generation_id")),
                validation=validation,
                sequence_reasons=sequence_reasons,
                public_decision=public_decision,
                current_best_before=before,
                current_best_after=current,
            )
        )
    return AcceptanceReplayResult(tuple(entries), current)


def _require_replay_anchor(
    *,
    initial_current_best: CurrentBestEvalStats | None,
    expected_session_genesis_hash: str | None,
    acceptance_protocol: AcceptanceProtocol,
    expected_initial_policy_state_hash: str | None = None,
) -> None:
    if not isinstance(initial_current_best, CurrentBestEvalStats):
        raise TypeError("initial_current_best must be CurrentBestEvalStats")
    require_sha256_digest(
        expected_session_genesis_hash,
        "expected_session_genesis_hash",
    )
    if expected_initial_policy_state_hash is not None:
        require_sha256_digest(
            expected_initial_policy_state_hash,
            "expected_initial_policy_state_hash",
        )
    binding = initial_current_best.evaluation_binding()
    if binding.acceptance_protocol_hash != acceptance_protocol.protocol_hash:
        raise ValueError("initial current-best acceptance protocol mismatch")
    if initial_current_best.mode in {
        "accepted",
        "reference",
    } and not initial_current_best.has_full_eval_stats(acceptance_protocol):
        raise ValueError(
            f"initial {initial_current_best.mode} current-best is incomplete"
        )


def _sequence_reasons(
    record: Mapping[str, Any],
    *,
    validation: DecisionRecordValidation,
    current_best: CurrentBestEvalStats,
    acceptance_protocol: AcceptanceProtocol,
) -> list[str]:
    if validation.round_stage.stage != EVALUATED_STAGE:
        return []
    surface = _mapping(record.get("candidate_surface"))
    public_result = _mapping(record.get("public_result"))
    if not acceptance_comparison_entered(
        policy_eligible=surface.get("eligible_for_policy_accept") is True,
        public_result_status=public_result.get("status"),
    ):
        return []
    decision = _mapping(record.get("decision"))
    comparison = _mapping(decision.get("comparison"))
    if not comparison:
        return ["acceptance_replay_comparison_missing"]

    reasons: list[str] = []
    current_best_public = current_best.to_public_dict()
    if comparison.get("mode") != current_best.mode:
        reasons.append("acceptance_replay_comparison_mode_mismatch")
    _compare_optional_number(
        reasons,
        "acceptance_replay_current_best_score_mismatch",
        comparison.get("current_best_score"),
        current_best.score_for_comparison(),
    )
    expected_best_score_stats = _mapping(
        current_best_public.get("public_accept_stats")
    )
    if dict(_mapping(comparison.get("best_score_stats"))) != dict(
        expected_best_score_stats
    ):
        reasons.append("acceptance_replay_best_score_stats_mismatch")
    evidence = validation.acceptance_evidence
    candidate_guard = evidence.guard_score_mean
    best_guard = current_best.guard_score_for_comparison()
    guard_met: bool | None = None
    if acceptance_protocol.uses_guard_score:
        expected_best_guard_stats = _mapping(
            current_best_public.get("guard_score_stats")
        )
        if dict(_mapping(comparison.get("best_guard_score_stats"))) != dict(
            expected_best_guard_stats
        ):
            reasons.append("acceptance_replay_best_guard_score_stats_mismatch")
        _compare_optional_number(
            reasons,
            "acceptance_replay_candidate_guard_score_mismatch",
            comparison.get("candidate_guard_score"),
            candidate_guard,
        )
        _compare_optional_number(
            reasons,
            "acceptance_replay_current_best_guard_score_mismatch",
            comparison.get("current_best_guard_score"),
            best_guard,
        )
        guard_threshold_result = None
        if candidate_guard is not None and best_guard is not None:
            try:
                guard_threshold_result = guard_regression_threshold(
                    acceptance_protocol,
                    statistics_version=evidence.guard_score_stats.get("statistics_version"),
                    candidate_guard_score=candidate_guard,
                    best_guard_score=best_guard,
                    candidate_guard_score_samples=evidence.guard_score_samples,
                    best_guard_score_samples=(
                        current_best.guard_score_samples_for_comparison(
                            acceptance_protocol
                        )
                    ),
                )
            except ValueError:
                reasons.append("acceptance_replay_guard_threshold_evidence_invalid")
        effective_regression = (
            guard_threshold_result.threshold
            if guard_threshold_result is not None
            else None
        )
        _compare_optional_number(
            reasons,
            "acceptance_replay_guard_threshold_mismatch",
            comparison.get("guard_regression_threshold"),
            effective_regression,
        )
        expected_components = (
            guard_threshold_result.to_public_dict()
            if guard_threshold_result is not None
            else {}
        )
        if not threshold_components_match(
            comparison.get("guard_threshold_components"),
            expected_components,
        ):
            reasons.append("acceptance_replay_guard_threshold_components_mismatch")
        guard_met = (
            candidate_guard >= best_guard - effective_regression
            if candidate_guard is not None
            and best_guard is not None
            and effective_regression is not None
            else None
        )
        if comparison.get("guard_met") is not guard_met:
            reasons.append("acceptance_replay_guard_met_mismatch")

    named_guard_mets: dict[str, bool | None] = {}
    if acceptance_protocol.uses_named_guards:
        declared_guards = _mapping(comparison.get("named_guards"))
        best_named_guards = _mapping(
            current_best_public.get("named_guard_evidence")
        )
        if set(declared_guards) != {
            guard.name for guard in acceptance_protocol.named_guards
        }:
            reasons.append("acceptance_replay_named_guard_set_mismatch")
        for guard in acceptance_protocol.named_guards:
            declared = _mapping(declared_guards.get(guard.name))
            candidate_item = _mapping(evidence.named_guard_evidence.get(guard.name))
            candidate_stats = _mapping(candidate_item.get("score_stats"))
            best_item = _mapping(best_named_guards.get(guard.name))
            best_stats = _mapping(best_item.get("score_stats"))
            candidate_value = _finite_number(candidate_stats.get("mean"))
            best_value = current_best.named_guard_score_for_comparison(guard.name)
            if dict(_mapping(declared.get("best_score_stats"))) != dict(best_stats):
                reasons.append(
                    f"acceptance_replay_best_named_guard_stats_mismatch:{guard.name}"
                )
            threshold_result = None
            if candidate_value is not None and best_value is not None:
                try:
                    threshold_result = named_guard_regression_threshold(
                        acceptance_protocol,
                        guard,
                        statistics_version=candidate_stats.get("statistics_version"),
                        candidate_guard_score=candidate_value,
                        best_guard_score=best_value,
                        candidate_guard_score_samples=candidate_item.get(
                            "score_samples"
                        ),
                        best_guard_score_samples=(
                            current_best.named_guard_score_samples_for_comparison(
                                guard.name, acceptance_protocol
                            )
                        ),
                    )
                except ValueError:
                    reasons.append(
                        "acceptance_replay_named_guard_threshold_evidence_invalid:"
                        f"{guard.name}"
                    )
            effective = (
                threshold_result.threshold
                if threshold_result is not None
                else None
            )
            _compare_optional_number(
                reasons,
                f"acceptance_replay_named_guard_threshold_mismatch:{guard.name}",
                declared.get("regression_threshold"),
                effective,
            )
            expected_components = (
                threshold_result.to_public_dict()
                if threshold_result is not None
                else {}
            )
            if not threshold_components_match(
                declared.get("threshold_components"), expected_components
            ):
                reasons.append(
                    "acceptance_replay_named_guard_threshold_components_mismatch:"
                    f"{guard.name}"
                )
            met = (
                candidate_value >= best_value - effective
                if candidate_value is not None
                and best_value is not None
                and effective is not None
                else None
            )
            named_guard_mets[guard.name] = met
            if declared.get("met") is not met:
                reasons.append(
                    f"acceptance_replay_named_guard_met_mismatch:{guard.name}"
                )

    declared_best = _mapping(comparison.get("current_best"))
    for field, expected in current_best.comparison_identity().items():
        if declared_best.get(field) != expected:
            reasons.append(f"acceptance_replay_current_best_{field}_mismatch")

    candidate_score = evidence.score_mean
    best_score = current_best.score_for_comparison()
    candidate_se = evidence.standard_error
    best_se = _finite_number(current_best.public_accept_stats.get("standard_error"))
    metrics = _mapping(public_result.get("metrics"))
    candidate_binding = _mapping(metrics.get("evaluation_binding"))
    evaluation_binding_matches = bool(
        candidate_binding.get("evaluation_binding_hash")
        == current_best.evaluation_binding_hash
    )
    expected_delta = (
        candidate_score - best_score
        if candidate_score is not None and best_score is not None
        else None
    )
    expected_delta_met = None
    compare_delta = (
        acceptance_protocol.decision_rule == NOISE_AWARE_DELTA_RULE
        or (
            current_best.mode in {"accepted", "reference"}
            and evaluation_binding_matches
        )
    )
    if compare_delta:
        try:
            threshold_result = acceptance_threshold(
                acceptance_protocol,
                statistics_version=evidence.score_stats.get("statistics_version"),
                candidate_score=candidate_score,
                best_score=best_score,
                candidate_score_samples=evidence.score_samples,
                best_score_samples=current_best.score_samples_for_comparison(
                    acceptance_protocol
                ),
                candidate_standard_error=candidate_se,
                best_standard_error=best_se,
            )
        except ValueError:
            threshold_result = None
            threshold = acceptance_protocol.min_delta
            reasons.append("acceptance_replay_threshold_evidence_invalid")
        else:
            threshold = threshold_result.threshold
        _compare_optional_number(
            reasons,
            "acceptance_replay_threshold_mismatch",
            comparison.get("threshold"),
            threshold,
        )
        if (
            acceptance_protocol.decision_rule != NOISE_AWARE_DELTA_RULE
            and threshold_result is not None
            and not threshold_components_match(
                comparison.get("threshold_components"),
                threshold_result.to_public_dict(),
            )
        ):
            reasons.append("acceptance_replay_threshold_components_mismatch")
        _compare_optional_number(
            reasons,
            "acceptance_replay_delta_mismatch",
            comparison.get("delta"),
            expected_delta,
        )
        expected_delta_met = (
            acceptance_delta_met(
                acceptance_protocol,
                expected_delta,
                threshold,
            )
            if expected_delta is not None
            else None
        )
        if comparison.get("delta_met") is not expected_delta_met:
            reasons.append("acceptance_replay_delta_met_mismatch")

    accepted = decision.get("accepted") is True
    reason = decision.get("reason")
    if accepted:
        if current_best.mode == "bootstrap":
            expected_bootstrap_reason = (
                "absolute_and_named_guard_accept_floors"
                if acceptance_protocol.uses_named_guards
                else "absolute_and_guard_accept_floors"
                if acceptance_protocol.uses_guard_score
                else "absolute_accept_floor"
            )
            named_floor_failed = any(
                _finite_number(
                    _mapping(
                        _mapping(evidence.named_guard_evidence.get(guard.name)).get(
                            "score_stats"
                        )
                    ).get("mean")
                )
                is None
                or _finite_number(
                    _mapping(
                        _mapping(evidence.named_guard_evidence.get(guard.name)).get(
                            "score_stats"
                        )
                    ).get("mean")
                )
                < guard.absolute_accept_floor
                for guard in acceptance_protocol.named_guards
            )
            if (
                candidate_score is None
                or candidate_score < acceptance_protocol.absolute_accept_floor
                or (
                    acceptance_protocol.uses_guard_score
                    and (
                        candidate_guard is None
                        or acceptance_protocol.guard_absolute_accept_floor is None
                        or candidate_guard
                        < acceptance_protocol.guard_absolute_accept_floor
                    )
                )
                or named_floor_failed
                or reason != expected_bootstrap_reason
            ):
                reasons.append("acceptance_replay_bootstrap_accept_invalid")
        elif current_best.mode in {"accepted", "reference"}:
            if evidence.eval_protocol_hash != current_best.eval_protocol_hash:
                reasons.append("acceptance_replay_eval_protocol_mismatch")
            valid_accept_reasons = {
                acceptance_delta_reason(acceptance_protocol, accepted=True)
            }
            if acceptance_protocol.decision_rule == NOISE_AWARE_DELTA_RULE:
                valid_accept_reasons.add("public_score_delta_met")
            if (
                expected_delta_met is not True
                or (acceptance_protocol.uses_guard_score and guard_met is not True)
                or (
                    acceptance_protocol.uses_named_guards
                    and any(value is not True for value in named_guard_mets.values())
                )
                or reason not in valid_accept_reasons
            ):
                reasons.append("acceptance_replay_delta_accept_invalid")
        else:
            reasons.append("acceptance_replay_incomplete_best_accept_invalid")
    elif reason == "absolute_accept_floor_not_met":
        if (
            current_best.mode != "bootstrap"
            or candidate_score is None
            or candidate_score >= acceptance_protocol.absolute_accept_floor
        ):
            reasons.append("acceptance_replay_bootstrap_reject_invalid")
    elif reason == "absolute_or_guard_accept_floor_not_met":
        guard_floor = acceptance_protocol.guard_absolute_accept_floor
        if (
            not acceptance_protocol.uses_guard_score
            or current_best.mode != "bootstrap"
            or candidate_score is None
            or candidate_guard is None
            or guard_floor is None
            or (
                candidate_score >= acceptance_protocol.absolute_accept_floor
                and candidate_guard >= guard_floor
            )
        ):
            reasons.append("acceptance_replay_bootstrap_reject_invalid")
    elif reason == "absolute_or_named_guard_accept_floor_not_met":
        named_scores = {
            guard.name: _finite_number(
                _mapping(
                    _mapping(evidence.named_guard_evidence.get(guard.name)).get(
                        "score_stats"
                    )
                ).get("mean")
            )
            for guard in acceptance_protocol.named_guards
        }
        if (
            not acceptance_protocol.uses_named_guards
            or current_best.mode != "bootstrap"
            or candidate_score is None
            or any(value is None for value in named_scores.values())
            or (
                candidate_score >= acceptance_protocol.absolute_accept_floor
                and all(
                    named_scores[guard.name] >= guard.absolute_accept_floor
                    for guard in acceptance_protocol.named_guards
                )
            )
        ):
            reasons.append("acceptance_replay_bootstrap_reject_invalid")
    elif reason == "guard_score_regression":
        if (
            not acceptance_protocol.uses_guard_score
            or current_best.mode not in {"accepted", "reference"}
            or guard_met is not False
        ):
            reasons.append("acceptance_replay_guard_reject_invalid")
    elif reason == "named_guard_score_regression":
        if (
            not acceptance_protocol.uses_named_guards
            or current_best.mode not in {"accepted", "reference"}
            or not any(value is False for value in named_guard_mets.values())
        ):
            reasons.append("acceptance_replay_named_guard_reject_invalid")
    elif reason in {
        acceptance_delta_reason(acceptance_protocol, accepted=False),
        *(
            ("public_score_delta_not_met",)
            if acceptance_protocol.decision_rule == NOISE_AWARE_DELTA_RULE
            else ()
        ),
    }:
        if (
            current_best.mode not in {"accepted", "reference"}
            or expected_delta_met is not False
        ):
            reasons.append("acceptance_replay_delta_reject_invalid")
    elif reason == "eval_protocol_hash_mismatch":
        if (
            current_best.mode not in {"accepted", "reference"}
            or evidence.eval_protocol_hash is None
            or evidence.eval_protocol_hash == current_best.eval_protocol_hash
        ):
            reasons.append("acceptance_replay_eval_protocol_reject_invalid")
    return reasons


def _source_parent_reasons(
    record: Mapping[str, Any],
    *,
    current_best: CurrentBestEvalStats,
) -> list[str]:
    """Bind new ledger records to the replay head used to author the round."""

    strict = is_strict_ledger_record(record)
    field_present = "parent_program_source_checksum" in record
    declared = record.get("parent_program_source_checksum")
    expected = current_best.program_source_checksum
    if expected is None:
        if field_present and declared is not None:
            return ["acceptance_replay_parent_program_source_unexpected"]
        return []
    if not field_present:
        return (
            ["acceptance_replay_parent_program_source_checksum_missing"]
            if strict
            else []
        )
    if declared != expected:
        return ["acceptance_replay_parent_program_source_checksum_mismatch"]
    return []


def _ledger_epoch_reasons(
    record: Mapping[str, Any],
    *,
    expected_ledger_epoch_hash: str,
    legacy_unbound_epoch_allowed: bool,
) -> list[str]:
    descriptor = dispatch_ledger_record_schema(record)
    if not descriptor.ledger_epoch_bound:
        return (
            []
            if legacy_unbound_epoch_allowed
            else ["ledger_epoch_legacy_unbound"]
        )
    try:
        actual = require_ledger_record_epoch_hash(record)
    except (TypeError, ValueError):
        return ["ledger_epoch_hash_invalid"]
    if actual != expected_ledger_epoch_hash:
        return ["ledger_epoch_hash_mismatch"]
    return []


def _is_session_binding_reason(reason: str) -> bool:
    return (
        reason.startswith("session_genesis_")
        or reason.startswith("ledger_epoch_")
        or "evaluation_binding" in reason
        or "initial_policy_state" in reason
        or "parent_program_source" in reason
        or reason.startswith("candidate_runtime_error_")
        or reason.startswith("candidate_failure_contract_")
        or reason.startswith("prompt_artifact_")
        or reason.startswith("authoring_evidence_")
        or reason.startswith("mechanism_attribution_")
    )


def _compare_optional_number(
    reasons: list[str],
    reason: str,
    declared: Any,
    expected: float | None,
) -> None:
    parsed = _finite_number(declared)
    if parsed is None or expected is None:
        if parsed is not None or expected is not None:
            reasons.append(reason)
        return
    if not math.isclose(parsed, expected, rel_tol=1e-9, abs_tol=5e-7):
        reasons.append(reason)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


__all__ = [
    "AcceptanceReplayEntry",
    "AcceptanceReplayResult",
    "ProjectedPublicHistory",
    "project_public_history",
    "project_replayed_records",
    "replay_acceptance_ledger",
]
