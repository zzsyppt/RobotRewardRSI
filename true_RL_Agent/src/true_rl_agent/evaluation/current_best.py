"""Current-best evaluation state used by runner-owned acceptance decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from true_rl_agent.evaluation.acceptance_contract import validate_acceptance_evidence
from true_rl_agent.evaluation.acceptance_protocol import (
    AcceptanceProtocol,
    validated_score_samples,
)
from true_rl_agent.evaluation.binding import (
    EvaluationBindingV1,
    is_sha256_checksum,
    is_sha256_digest,
    require_sha256_digest,
)
from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.evaluation.result_contract import validate_public_eval_result_mapping
from true_rl_agent.evaluation.score_stats import validate_score_stats
from true_rl_agent.evaluation.sealed_validation import (
    validate_sealed_rollout_evidence,
    validate_sealed_validation_gate_contract,
)
from true_rl_agent.evaluation.subject import ExperimentSubject


@dataclass(frozen=True)
class CurrentBestEvalStats:
    """Public-safe, replayable identity and evaluation state for the policy best."""

    mode: str
    anchor_kind: str
    anchor_id: str
    public_score: float | None = None
    public_accept_stats: Mapping[str, Any] = field(default_factory=dict)
    guard_score_stats: Mapping[str, Any] = field(default_factory=dict)
    named_guard_evidence: Mapping[str, Any] = field(default_factory=dict)
    sealed_validation_evidence: Mapping[str, Any] = field(default_factory=dict)
    sealed_validation_gate: Mapping[str, Any] = field(default_factory=dict)
    gate_statuses: Mapping[str, str] = field(default_factory=dict)
    checkpoint_hash: str | None = None
    subject_manifest_hash: str | None = None
    program_source_checksum: str | None = None
    training_binding_hash: str | None = None
    capability_profile_hash: str | None = None
    task_protocol_hash: str | None = None
    eval_protocol_hash: str | None = None
    acceptance_protocol_hash: str | None = None
    seed_cohort_hash: str | None = None
    evaluation_binding_hash: str | None = None
    min_score_count: int = 1
    accepted_round_id: str | None = None
    accepted_candidate_id: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"bootstrap", "reference", "accepted", "incomplete"}:
            raise ValueError(f"unknown current-best eval stats mode: {self.mode}")
        if self.anchor_kind not in {"bootstrap", "baseline", "reference", "candidate"}:
            raise ValueError(f"unknown current-best anchor kind: {self.anchor_kind}")
        if not isinstance(self.anchor_id, str) or not self.anchor_id:
            raise ValueError("current-best anchor_id must be non-empty")
        if isinstance(self.min_score_count, bool) or not isinstance(
            self.min_score_count, int
        ) or self.min_score_count <= 0:
            raise ValueError("current-best min_score_count must be positive")
        if self.mode == "bootstrap" and self.anchor_kind != "bootstrap":
            raise ValueError("bootstrap current-best requires a bootstrap anchor")
        if self.mode == "accepted" and self.anchor_kind not in {
            "baseline",
            "candidate",
        }:
            raise ValueError("accepted current-best requires baseline or candidate anchor")
        if self.mode == "reference" and self.anchor_kind != "reference":
            raise ValueError("reference current-best requires a reference anchor")
        if self.anchor_kind == "candidate":
            ExperimentSubject.candidate(self.anchor_id)
            if self.accepted_candidate_id != self.anchor_id:
                raise ValueError("candidate anchor_id must match accepted_candidate_id")
            if not self.accepted_round_id:
                raise ValueError("candidate current-best requires accepted_round_id")
        elif self.accepted_candidate_id is not None:
            raise ValueError("non-candidate current-best cannot carry accepted_candidate_id")
        if self.anchor_kind in {"baseline", "reference", "candidate"}:
            if not is_sha256_digest(self.subject_manifest_hash):
                raise ValueError("evaluated current-best requires subject_manifest_hash")
        elif self.subject_manifest_hash is not None:
            raise ValueError("bootstrap current-best cannot carry subject_manifest_hash")
        if self.anchor_kind in {"reference", "candidate"}:
            if not is_sha256_checksum(self.program_source_checksum):
                raise ValueError("program current-best requires program_source_checksum")
            for field_name in ("training_binding_hash", "capability_profile_hash"):
                if not is_sha256_digest(getattr(self, field_name)):
                    raise ValueError(f"program current-best requires {field_name}")
        elif self.program_source_checksum is not None or self.training_binding_hash is not None:
            raise ValueError("non-program current-best cannot carry program lineage")
        if self.anchor_kind == "baseline" and not is_sha256_digest(
            self.capability_profile_hash
        ):
            raise ValueError("baseline current-best requires capability_profile_hash")
        if self.mode in {"accepted", "reference"}:
            if not is_sha256_checksum(self.checkpoint_hash):
                raise ValueError("accepted current-best requires checkpoint_hash")
            self.evaluation_binding()
        for field_name in (
            "public_accept_stats",
            "guard_score_stats",
            "named_guard_evidence",
            "sealed_validation_evidence",
            "sealed_validation_gate",
            "gate_statuses",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, Mapping):
                raise TypeError(f"current-best {field_name} must be an object")
            object.__setattr__(self, field_name, _freeze_json_value(value))

    @classmethod
    def bootstrap(
        cls,
        *,
        initial_best_score: float,
        acceptance_protocol: AcceptanceProtocol,
        evaluation_binding: EvaluationBindingV1,
        anchor_id: str,
    ) -> "CurrentBestEvalStats":
        _assert_binding_matches_acceptance(evaluation_binding, acceptance_protocol)
        return cls(
            mode="bootstrap",
            anchor_kind="bootstrap",
            anchor_id=anchor_id,
            public_score=_required_number(
                initial_best_score,
                "initial_best_score",
            ),
            task_protocol_hash=evaluation_binding.task_protocol_hash,
            eval_protocol_hash=evaluation_binding.eval_protocol_hash,
            acceptance_protocol_hash=evaluation_binding.acceptance_protocol_hash,
            seed_cohort_hash=evaluation_binding.seed_cohort_hash,
            evaluation_binding_hash=evaluation_binding.binding_hash,
        )

    @classmethod
    def from_public_result(
        cls,
        *,
        round_id: str,
        candidate_id: str,
        public_result: PublicEvalResult,
        acceptance_protocol: AcceptanceProtocol,
        subject_manifest_hash: str,
        program_source_checksum: str,
        training_binding_hash: str,
        capability_profile_hash: str,
        checkpoint_hash: str | None = None,
        expected_evaluation_binding: EvaluationBindingV1 | None = None,
    ) -> "CurrentBestEvalStats":
        return cls.from_public_result_mapping(
            round_id=round_id,
            candidate_id=candidate_id,
            public_result=public_result.to_public_dict(),
            acceptance_protocol=acceptance_protocol,
            subject_manifest_hash=subject_manifest_hash,
            program_source_checksum=program_source_checksum,
            training_binding_hash=training_binding_hash,
            capability_profile_hash=capability_profile_hash,
            checkpoint_hash=checkpoint_hash,
            expected_evaluation_binding=expected_evaluation_binding,
        )

    @classmethod
    def from_public_result_mapping(
        cls,
        *,
        round_id: str,
        candidate_id: str,
        public_result: Mapping[str, Any],
        acceptance_protocol: AcceptanceProtocol,
        subject_manifest_hash: str,
        program_source_checksum: str,
        training_binding_hash: str,
        capability_profile_hash: str,
        checkpoint_hash: str | None = None,
        expected_evaluation_binding: EvaluationBindingV1 | None = None,
    ) -> "CurrentBestEvalStats":
        """Restore an accepted candidate anchor from its runner-owned result."""

        return cls._from_evaluated_anchor(
            anchor_kind="candidate",
            anchor_id=candidate_id,
            round_id=round_id,
            public_result=public_result,
            acceptance_protocol=acceptance_protocol,
            checkpoint_hash=checkpoint_hash,
            subject_manifest_hash=subject_manifest_hash,
            program_source_checksum=program_source_checksum,
            training_binding_hash=training_binding_hash,
            capability_profile_hash=capability_profile_hash,
            expected_evaluation_binding=expected_evaluation_binding,
        )

    @classmethod
    def from_baseline_public_result_mapping(
        cls,
        *,
        baseline_id: str,
        baseline_manifest_hash: str,
        checkpoint_hash: str,
        public_result: Mapping[str, Any],
        acceptance_protocol: AcceptanceProtocol,
        expected_evaluation_binding: EvaluationBindingV1,
        capability_profile_hash: str,
    ) -> "CurrentBestEvalStats":
        """Derive a baseline current-best without creating a candidate record."""

        require_sha256_digest(
            baseline_manifest_hash,
            "baseline_manifest_hash",
        )
        return cls._from_evaluated_anchor(
            anchor_kind="baseline",
            anchor_id=baseline_id,
            round_id=None,
            public_result=public_result,
            acceptance_protocol=acceptance_protocol,
            checkpoint_hash=checkpoint_hash,
            subject_manifest_hash=baseline_manifest_hash,
            program_source_checksum=None,
            training_binding_hash=None,
            capability_profile_hash=capability_profile_hash,
            expected_evaluation_binding=expected_evaluation_binding,
        )

    @classmethod
    def from_reference_public_result_mapping(
        cls,
        *,
        reference_id: str,
        reference_manifest_hash: str,
        program_source_checksum: str,
        training_binding_hash: str,
        capability_profile_hash: str,
        checkpoint_hash: str,
        public_result: Mapping[str, Any],
        acceptance_protocol: AcceptanceProtocol,
        expected_evaluation_binding: EvaluationBindingV1,
    ) -> "CurrentBestEvalStats":
        return cls._from_evaluated_anchor(
            anchor_kind="reference",
            anchor_id=reference_id,
            round_id=None,
            public_result=public_result,
            acceptance_protocol=acceptance_protocol,
            checkpoint_hash=checkpoint_hash,
            subject_manifest_hash=reference_manifest_hash,
            program_source_checksum=program_source_checksum,
            training_binding_hash=training_binding_hash,
            capability_profile_hash=capability_profile_hash,
            expected_evaluation_binding=expected_evaluation_binding,
        )

    @classmethod
    def _from_evaluated_anchor(
        cls,
        *,
        anchor_kind: str,
        anchor_id: str,
        round_id: str | None,
        public_result: Mapping[str, Any],
        acceptance_protocol: AcceptanceProtocol,
        checkpoint_hash: str | None,
        subject_manifest_hash: str | None,
        program_source_checksum: str | None,
        training_binding_hash: str | None,
        capability_profile_hash: str,
        expected_evaluation_binding: EvaluationBindingV1 | None,
    ) -> "CurrentBestEvalStats":
        if not isinstance(anchor_id, str) or not anchor_id:
            raise ValueError("evaluated current-best anchor_id must be non-empty")
        fields = validate_public_eval_result_mapping(
            public_result,
            field_name=f"{anchor_kind} current-best public result",
        )
        metrics = fields.metrics
        binding = EvaluationBindingV1.from_mapping(metrics.get("evaluation_binding"))
        _assert_binding_matches_acceptance(binding, acceptance_protocol)
        if expected_evaluation_binding is not None and binding != expected_evaluation_binding:
            raise ValueError("current-best evaluation binding mismatch")

        expected_subject = ExperimentSubject(
            kind=anchor_kind,
            subject_id=anchor_id,
        )
        if ExperimentSubject.from_mapping(
            metrics.get("evaluation_subject")
        ) != expected_subject:
            raise ValueError("current-best evaluation subject mismatch")
        expected_lineage = {
            "subject_manifest_hash": subject_manifest_hash,
            "program_source_checksum": program_source_checksum,
            "training_binding_hash": training_binding_hash,
            "capability_profile_hash": capability_profile_hash,
        }
        for field_name, expected_value in expected_lineage.items():
            if metrics.get(field_name) != expected_value:
                raise ValueError(
                    f"current-best lineage mismatch for {field_name}"
                )

        sealed_evidence = _mapping(metrics.get("sealed_validation"))
        rollout_validation = validate_sealed_rollout_evidence(
            sealed_evidence,
            expected_eval_protocol_hash=binding.eval_protocol_hash,
            expected_evaluation_binding_hash=binding.binding_hash,
            expected_evaluation_subject=expected_subject,
            expected_subject_manifest_hash=subject_manifest_hash,
            expected_program_source_checksum=program_source_checksum,
            expected_training_binding_hash=training_binding_hash,
            expected_capability_profile_hash=capability_profile_hash,
            expected_checkpoint_hash=checkpoint_hash,
        )
        if not rollout_validation.passed:
            detail = ", ".join(rollout_validation.reasons) or "status is not pass"
            raise ValueError(f"current-best sealed rollout invalid: {detail}")
        resolved_checkpoint_hash = rollout_validation.checkpoint_hash
        if checkpoint_hash is not None and resolved_checkpoint_hash != checkpoint_hash:
            raise ValueError("current-best checkpoint hash mismatch")
        if not is_sha256_checksum(resolved_checkpoint_hash):
            raise ValueError("current-best checkpoint hash is missing")

        acceptance_validation = validate_acceptance_evidence(
            metrics,
            acceptance_protocol=acceptance_protocol,
            public_score=fields.public_score,
            expected_sealed_seed_count=rollout_validation.expected_seed_count,
            expected_checkpoint_hash=resolved_checkpoint_hash,
            evidence_subject_kind=anchor_kind,
        )
        complete = fields.status == "pass" and acceptance_validation.valid
        mode = (
            "reference"
            if complete and anchor_kind == "reference"
            else "accepted"
            if complete
            else "incomplete"
        )
        return cls(
            mode=mode,
            anchor_kind=anchor_kind,
            anchor_id=anchor_id,
            public_score=acceptance_validation.canonical_public_score,
            public_accept_stats=dict(acceptance_validation.score_stats),
            guard_score_stats=dict(acceptance_validation.guard_score_stats),
            named_guard_evidence=dict(
                acceptance_validation.named_guard_evidence
            ),
            sealed_validation_evidence=dict(
                acceptance_validation.sealed_validation_evidence
            ),
            sealed_validation_gate=dict(
                acceptance_validation.sealed_validation_gate
            ),
            gate_statuses=dict(acceptance_validation.gate_statuses),
            checkpoint_hash=resolved_checkpoint_hash,
            subject_manifest_hash=subject_manifest_hash,
            program_source_checksum=program_source_checksum,
            training_binding_hash=training_binding_hash,
            capability_profile_hash=capability_profile_hash,
            task_protocol_hash=binding.task_protocol_hash,
            eval_protocol_hash=binding.eval_protocol_hash,
            acceptance_protocol_hash=binding.acceptance_protocol_hash,
            seed_cohort_hash=binding.seed_cohort_hash,
            evaluation_binding_hash=binding.binding_hash,
            min_score_count=acceptance_validation.min_score_count or 1,
            accepted_round_id=round_id,
            accepted_candidate_id=(anchor_id if anchor_kind == "candidate" else None),
        )

    @classmethod
    def from_ledger_records(
        cls,
        records: Sequence[Mapping[str, Any]],
        *,
        initial_current_best: "CurrentBestEvalStats",
        session_genesis_hash: str,
        acceptance_protocol: AcceptanceProtocol,
        expected_ledger_epoch_hash: str | None = None,
    ) -> "CurrentBestEvalStats":
        from true_rl_agent.evaluation.acceptance_replay import (  # noqa: PLC0415
            replay_acceptance_ledger,
        )

        return replay_acceptance_ledger(
            records,
            initial_current_best=initial_current_best,
            expected_session_genesis_hash=session_genesis_hash,
            acceptance_protocol=acceptance_protocol,
            expected_ledger_epoch_hash=expected_ledger_epoch_hash,
        ).current_best

    def evaluation_binding(self) -> EvaluationBindingV1:
        binding = EvaluationBindingV1(
            eval_protocol_hash=require_sha256_digest(
                self.eval_protocol_hash,
                "current-best eval_protocol_hash",
            ),
            acceptance_protocol_hash=require_sha256_digest(
                self.acceptance_protocol_hash,
                "current-best acceptance_protocol_hash",
            ),
            task_protocol_hash=require_sha256_digest(
                self.task_protocol_hash,
                "current-best task_protocol_hash",
            ),
            seed_cohort_hash=require_sha256_digest(
                self.seed_cohort_hash,
                "current-best seed_cohort_hash",
            ),
        )
        if binding.binding_hash != self.evaluation_binding_hash:
            raise ValueError("current-best evaluation_binding_hash mismatch")
        return binding

    def has_full_eval_stats(
        self,
        acceptance_protocol: AcceptanceProtocol | None = None,
    ) -> bool:
        if self.mode not in {"accepted", "reference"}:
            return False
        try:
            binding = self.evaluation_binding()
            subject = ExperimentSubject(
                kind=self.anchor_kind,
                subject_id=self.anchor_id,
            )
            rollout = validate_sealed_rollout_evidence(
                self.sealed_validation_evidence,
                expected_eval_protocol_hash=binding.eval_protocol_hash,
                expected_evaluation_binding_hash=binding.binding_hash,
                expected_evaluation_subject=subject,
                expected_subject_manifest_hash=self.subject_manifest_hash,
                expected_program_source_checksum=self.program_source_checksum,
                expected_training_binding_hash=self.training_binding_hash,
                expected_capability_profile_hash=self.capability_profile_hash,
                expected_checkpoint_hash=self.checkpoint_hash,
            )
        except (TypeError, ValueError):
            return False
        stats_validation = validate_score_stats(
            self.public_accept_stats,
            minimum_count=self.min_score_count,
            require_std=False,
            require_standard_error=self.min_score_count >= 2,
            require_protocol_hash=True,
        )
        gate_validation = validate_sealed_validation_gate_contract(
            self.sealed_validation_gate,
            rollout_evidence=self.sealed_validation_evidence,
            eval_protocol_hash=binding.eval_protocol_hash,
            expected_seed_count=rollout.expected_seed_count,
            expected_checkpoint_hash=self.checkpoint_hash,
        )
        score_samples_valid = True
        guard_score_valid = not (
            acceptance_protocol is not None and acceptance_protocol.uses_guard_score
        )
        named_guards_valid = not (
            acceptance_protocol is not None and acceptance_protocol.uses_named_guards
        )
        if (
            acceptance_protocol is not None
            and acceptance_protocol.requires_paired_score_samples
        ):
            try:
                if stats_validation.mean is None:
                    raise ValueError("current-best score mean is missing")
                validated_score_samples(
                    self.public_accept_stats.get("score_samples"),
                    expected_mean=stats_validation.mean,
                    protocol=acceptance_protocol,
                    field_name="current_best.public_accept_stats.score_samples",
                )
            except ValueError:
                score_samples_valid = False
        if acceptance_protocol is not None and acceptance_protocol.uses_guard_score:
            guard_validation = validate_score_stats(
                self.guard_score_stats,
                minimum_count=self.min_score_count,
                require_std=True,
                require_standard_error=self.min_score_count >= 2,
                require_protocol_hash=True,
            )
            guard_score_valid = bool(
                guard_validation.valid
                and guard_validation.eval_protocol_hash == binding.eval_protocol_hash
            )
            if (
                guard_score_valid
                and acceptance_protocol.requires_paired_guard_score_samples
            ):
                try:
                    if guard_validation.mean is None:
                        raise ValueError("current-best guard score mean is missing")
                    validated_score_samples(
                        self.guard_score_stats.get("score_samples"),
                        expected_mean=guard_validation.mean,
                        protocol=acceptance_protocol,
                        field_name=(
                            "current_best.guard_score_stats.score_samples"
                        ),
                    )
                except ValueError:
                    guard_score_valid = False
        if acceptance_protocol is not None and acceptance_protocol.uses_named_guards:
            named_guards_valid = set(self.named_guard_evidence) == {
                guard.name for guard in acceptance_protocol.named_guards
            }
            for guard in acceptance_protocol.named_guards:
                item = _mapping(self.named_guard_evidence.get(guard.name))
                stats = _mapping(item.get("score_stats"))
                validation = validate_score_stats(
                    stats,
                    minimum_count=self.min_score_count,
                    require_std=True,
                    require_standard_error=self.min_score_count >= 2,
                    require_protocol_hash=True,
                )
                item_valid = bool(
                    validation.valid
                    and validation.eval_protocol_hash == binding.eval_protocol_hash
                    and item.get("score_source") == guard.score_source
                )
                try:
                    if validation.mean is None:
                        raise ValueError("current-best named guard mean is missing")
                    validated_score_samples(
                        item.get("score_samples"),
                        expected_mean=validation.mean,
                        protocol=acceptance_protocol,
                        field_name=(
                            f"current_best.named_guard_evidence.{guard.name}.score_samples"
                        ),
                    )
                except ValueError:
                    item_valid = False
                named_guards_valid = named_guards_valid and item_valid
        return bool(
            stats_validation.valid
            and score_samples_valid
            and guard_score_valid
            and named_guards_valid
            and stats_validation.eval_protocol_hash == binding.eval_protocol_hash
            and rollout.passed
            and gate_validation.valid
            and self.gate_statuses
            and all(status == "pass" for status in self.gate_statuses.values())
        )

    def score_for_comparison(self) -> float | None:
        mean = _optional_number(self.public_accept_stats.get("mean"))
        if mean is not None:
            return mean
        return _optional_number(self.public_score)

    def score_samples_for_comparison(
        self,
        acceptance_protocol: AcceptanceProtocol,
    ) -> tuple[float, ...]:
        mean = self.score_for_comparison()
        if mean is None:
            return ()
        try:
            return validated_score_samples(
                self.public_accept_stats.get("score_samples"),
                expected_mean=mean,
                protocol=acceptance_protocol,
                field_name="current_best.public_accept_stats.score_samples",
            )
        except ValueError:
            return ()

    def guard_score_for_comparison(self) -> float | None:
        return _optional_number(self.guard_score_stats.get("mean"))

    def guard_score_samples_for_comparison(
        self,
        acceptance_protocol: AcceptanceProtocol,
    ) -> tuple[float, ...]:
        mean = self.guard_score_for_comparison()
        if mean is None:
            return ()
        try:
            return validated_score_samples(
                self.guard_score_stats.get("score_samples"),
                expected_mean=mean,
                protocol=acceptance_protocol,
                field_name="current_best.guard_score_stats.score_samples",
            )
        except ValueError:
            return ()

    def named_guard_score_for_comparison(self, name: str) -> float | None:
        item = _mapping(self.named_guard_evidence.get(name))
        return _optional_number(_mapping(item.get("score_stats")).get("mean"))

    def named_guard_score_samples_for_comparison(
        self,
        name: str,
        acceptance_protocol: AcceptanceProtocol,
    ) -> tuple[float, ...]:
        item = _mapping(self.named_guard_evidence.get(name))
        mean = self.named_guard_score_for_comparison(name)
        if mean is None:
            return ()
        try:
            return validated_score_samples(
                item.get("score_samples"),
                expected_mean=mean,
                protocol=acceptance_protocol,
                field_name=f"current_best.named_guard_evidence.{name}.score_samples",
            )
        except ValueError:
            return ()

    def comparison_identity(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "anchor_kind": self.anchor_kind,
            "anchor_id": self.anchor_id,
            "checkpoint_hash": self.checkpoint_hash,
            "subject_manifest_hash": self.subject_manifest_hash,
            "program_source_checksum": self.program_source_checksum,
            "training_binding_hash": self.training_binding_hash,
            "capability_profile_hash": self.capability_profile_hash,
            "accepted_round_id": self.accepted_round_id,
            "accepted_candidate_id": self.accepted_candidate_id,
            "task_protocol_hash": self.task_protocol_hash,
            "eval_protocol_hash": self.eval_protocol_hash,
            "acceptance_protocol_hash": self.acceptance_protocol_hash,
            "seed_cohort_hash": self.seed_cohort_hash,
            "evaluation_binding_hash": self.evaluation_binding_hash,
        }

    def to_public_dict(self) -> dict[str, Any]:
        return {
            **self.comparison_identity(),
            "public_score": self.public_score,
            "public_accept_stats": _thaw_json_value(self.public_accept_stats),
            "guard_score_stats": _thaw_json_value(self.guard_score_stats),
            "named_guard_evidence": _thaw_json_value(
                self.named_guard_evidence
            ),
            "sealed_validation_evidence": _thaw_json_value(
                self.sealed_validation_evidence
            ),
            "sealed_validation_gate": _thaw_json_value(
                self.sealed_validation_gate
            ),
            "gate_statuses": _thaw_json_value(self.gate_statuses),
            "min_score_count": self.min_score_count,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "CurrentBestEvalStats":
        """Restore the exact public current-best state without reinterpreting it."""

        expected_fields = {
            "mode",
            "anchor_kind",
            "anchor_id",
            "checkpoint_hash",
            "subject_manifest_hash",
            "program_source_checksum",
            "training_binding_hash",
            "capability_profile_hash",
            "accepted_round_id",
            "accepted_candidate_id",
            "task_protocol_hash",
            "eval_protocol_hash",
            "acceptance_protocol_hash",
            "seed_cohort_hash",
            "evaluation_binding_hash",
            "public_score",
            "public_accept_stats",
            "guard_score_stats",
            "named_guard_evidence",
            "sealed_validation_evidence",
            "sealed_validation_gate",
            "gate_statuses",
            "min_score_count",
        }
        legacy_fields = expected_fields - {"named_guard_evidence"}
        if not isinstance(value, Mapping) or frozenset(value) not in {
            frozenset(expected_fields),
            frozenset(legacy_fields),
        }:
            raise ValueError("current-best mapping fields mismatch")
        payload = dict(value)
        payload.setdefault("named_guard_evidence", {})
        return cls(**payload)


def attach_eval_protocol_hash(
    public_result: PublicEvalResult,
    *,
    eval_protocol_hash: str,
) -> PublicEvalResult:
    """Attach runner-owned eval protocol hash to candidate score stats."""

    if not is_sha256_digest(eval_protocol_hash):
        return public_result
    metrics = dict(public_result.metrics)
    metrics["eval_protocol_hash"] = eval_protocol_hash
    acceptance = (
        dict(metrics.get("acceptance", {}))
        if isinstance(metrics.get("acceptance"), Mapping)
        else {}
    )
    if acceptance:
        score_stats = (
            dict(acceptance.get("score_stats", {}))
            if isinstance(acceptance.get("score_stats"), Mapping)
            else {}
        )
        if score_stats:
            score_stats["eval_protocol_hash"] = eval_protocol_hash
            acceptance["score_stats"] = score_stats
        metrics["acceptance"] = acceptance
    public_accept = (
        dict(metrics.get("public_accept", {}))
        if isinstance(metrics.get("public_accept"), Mapping)
        else {}
    )
    accept_stats = (
        dict(public_accept.get("score_stats", {}))
        if isinstance(public_accept.get("score_stats"), Mapping)
        else {}
    )
    if accept_stats:
        accept_stats["eval_protocol_hash"] = eval_protocol_hash
        public_accept["score_stats"] = accept_stats
        metrics["public_accept"] = public_accept
    return PublicEvalResult(
        status=public_result.status,
        public_score=public_result.public_score,
        metrics=metrics,
    )


def _assert_binding_matches_acceptance(
    binding: EvaluationBindingV1,
    acceptance_protocol: AcceptanceProtocol,
) -> None:
    if binding.acceptance_protocol_hash != acceptance_protocol.protocol_hash:
        raise ValueError("evaluation binding acceptance protocol mismatch")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _freeze_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_json_value(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _thaw_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _thaw_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [_thaw_json_value(item) for item in value]
    return value


def _required_number(value: Any, field_name: str) -> float:
    parsed = _optional_number(value)
    if parsed is None:
        raise ValueError(f"{field_name} must be a finite non-boolean number")
    return parsed


def _optional_number(value: Any) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None
