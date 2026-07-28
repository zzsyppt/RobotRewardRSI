"""Sequential, task-neutral replay of acceptance decisions in ledger order."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from true_rl_agent.evaluation.acceptance_phase import acceptance_comparison_entered
from true_rl_agent.evaluation.acceptance_protocol import (
    AcceptanceProtocol,
    noise_aware_threshold,
)
from true_rl_agent.evaluation.binding import require_sha256_digest
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.evaluation.decision_record_contract import (
    DecisionRecordValidation,
    project_validated_decision,
    validate_decision_record,
)
from true_rl_agent.evaluation.round_stage_contract import EVALUATED_STAGE


@dataclass(frozen=True)
class AcceptanceReplayEntry:
    round_id: str
    validation: DecisionRecordValidation
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


def project_public_history(
    records: Sequence[Mapping[str, Any]],
    *,
    initial_current_best: CurrentBestEvalStats | None = None,
    expected_session_genesis_hash: str | None = None,
    acceptance_protocol: AcceptanceProtocol | None,
) -> ProjectedPublicHistory:
    """Project a full ordered history once before any public consumer sees it."""

    has_evaluated_history = any(
        isinstance(record.get("public_result"), Mapping)
        and bool(record.get("public_result"))
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
        return ProjectedPublicHistory(projected, None, None, None)
    _require_replay_anchor(
        initial_current_best=initial_current_best,
        expected_session_genesis_hash=expected_session_genesis_hash,
        acceptance_protocol=acceptance_protocol,
    )
    assert initial_current_best is not None
    assert expected_session_genesis_hash is not None
    projected = project_replayed_records(
        records,
        initial_current_best=initial_current_best,
        expected_session_genesis_hash=expected_session_genesis_hash,
        acceptance_protocol=acceptance_protocol,
    )
    return ProjectedPublicHistory(
        projected,
        initial_current_best,
        expected_session_genesis_hash,
        acceptance_protocol.protocol_hash,
    )


def project_replayed_records(
    records: Sequence[Mapping[str, Any]],
    *,
    initial_current_best: CurrentBestEvalStats,
    expected_session_genesis_hash: str,
    acceptance_protocol: AcceptanceProtocol,
) -> tuple[dict[str, Any], ...]:
    replay = replay_acceptance_ledger(
        records,
        initial_current_best=initial_current_best,
        expected_session_genesis_hash=expected_session_genesis_hash,
        acceptance_protocol=acceptance_protocol,
    )
    replay.raise_for_invalid_session_bindings()
    return tuple(
        {**dict(record), "decision": dict(entry.public_decision)}
        for record, entry in zip(records, replay.entries, strict=True)
    )


def replay_acceptance_ledger(
    records: Sequence[Mapping[str, Any]],
    *,
    initial_current_best: CurrentBestEvalStats,
    expected_session_genesis_hash: str,
    acceptance_protocol: AcceptanceProtocol,
) -> AcceptanceReplayResult:
    """Replay records from one verified, full session anchor."""

    _require_replay_anchor(
        initial_current_best=initial_current_best,
        expected_session_genesis_hash=expected_session_genesis_hash,
        acceptance_protocol=acceptance_protocol,
    )
    expected_binding = initial_current_best.evaluation_binding()
    current_best = initial_current_best
    entries: list[AcceptanceReplayEntry] = []
    for record in records:
        before = current_best
        validation = validate_decision_record(
            record,
            acceptance_protocol=acceptance_protocol,
            expected_session_genesis_hash=expected_session_genesis_hash,
            expected_evaluation_binding=expected_binding,
        )
        sequence_reasons = tuple(
            dict.fromkeys(
                _sequence_reasons(
                    record,
                    validation=validation,
                    current_best=before,
                    acceptance_protocol=acceptance_protocol,
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


def _require_replay_anchor(
    *,
    initial_current_best: CurrentBestEvalStats | None,
    expected_session_genesis_hash: str | None,
    acceptance_protocol: AcceptanceProtocol,
) -> None:
    if not isinstance(initial_current_best, CurrentBestEvalStats):
        raise TypeError("initial_current_best must be CurrentBestEvalStats")
    require_sha256_digest(
        expected_session_genesis_hash,
        "expected_session_genesis_hash",
    )
    binding = initial_current_best.evaluation_binding()
    if binding.acceptance_protocol_hash != acceptance_protocol.protocol_hash:
        raise ValueError("initial current-best acceptance protocol mismatch")
    if initial_current_best.mode in {
        "accepted",
        "reference",
    } and not initial_current_best.has_full_eval_stats():
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
    if comparison.get("mode") != current_best.mode:
        reasons.append("acceptance_replay_comparison_mode_mismatch")
    _compare_optional_number(
        reasons,
        "acceptance_replay_current_best_score_mismatch",
        comparison.get("current_best_score"),
        current_best.score_for_comparison(),
    )
    if dict(_mapping(comparison.get("best_score_stats"))) != dict(
        current_best.public_accept_stats
    ):
        reasons.append("acceptance_replay_best_score_stats_mismatch")

    declared_best = _mapping(comparison.get("current_best"))
    for field, expected in current_best.comparison_identity().items():
        if declared_best.get(field) != expected:
            reasons.append(f"acceptance_replay_current_best_{field}_mismatch")

    evidence = validation.acceptance_evidence
    candidate_score = evidence.score_mean
    best_score = current_best.score_for_comparison()
    candidate_se = evidence.standard_error
    best_se = _finite_number(current_best.public_accept_stats.get("standard_error"))
    threshold = noise_aware_threshold(
        acceptance_protocol,
        candidate_standard_error=candidate_se,
        best_standard_error=best_se,
    )
    _compare_optional_number(
        reasons,
        "acceptance_replay_threshold_mismatch",
        comparison.get("threshold"),
        threshold,
    )
    expected_delta = (
        candidate_score - best_score
        if candidate_score is not None and best_score is not None
        else None
    )
    _compare_optional_number(
        reasons,
        "acceptance_replay_delta_mismatch",
        comparison.get("delta"),
        expected_delta,
    )
    expected_delta_met = (
        expected_delta > threshold if expected_delta is not None else None
    )
    if comparison.get("delta_met") is not expected_delta_met:
        reasons.append("acceptance_replay_delta_met_mismatch")

    accepted = decision.get("accepted") is True
    reason = decision.get("reason")
    if accepted:
        if current_best.mode == "bootstrap":
            if (
                candidate_score is None
                or candidate_score < acceptance_protocol.absolute_accept_floor
                or reason != "absolute_accept_floor"
            ):
                reasons.append("acceptance_replay_bootstrap_accept_invalid")
        elif current_best.mode in {"accepted", "reference"}:
            if evidence.eval_protocol_hash != current_best.eval_protocol_hash:
                reasons.append("acceptance_replay_eval_protocol_mismatch")
            if expected_delta_met is not True or reason not in {
                "noise_aware_delta",
                "public_score_delta_met",
            }:
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
    elif reason in {"noise_aware_delta_not_met", "public_score_delta_not_met"}:
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


def _is_session_binding_reason(reason: str) -> bool:
    return (
        reason.startswith("session_genesis_")
        or "evaluation_binding" in reason
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
