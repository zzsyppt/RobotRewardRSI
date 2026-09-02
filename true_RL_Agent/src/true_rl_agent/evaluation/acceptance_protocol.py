"""Task-neutral fixed acceptance policy contract."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Mapping

ACCEPTANCE_PROTOCOL_SCHEMA_VERSION = "true_rl_agent.acceptance.protocol.v2"
PAIRED_HEADROOM_ACCEPTANCE_PROTOCOL_SCHEMA_VERSION = (
    "true_rl_agent.acceptance.protocol.v3"
)
GUARDED_PAIRED_HEADROOM_ACCEPTANCE_PROTOCOL_SCHEMA_VERSION = (
    "true_rl_agent.acceptance.protocol.v4"
)
BOUNDED_PAIRED_GUARD_ACCEPTANCE_PROTOCOL_SCHEMA_VERSION = (
    "true_rl_agent.acceptance.protocol.v5"
)
NAMED_GUARDS_ACCEPTANCE_PROTOCOL_SCHEMA_VERSION = (
    "true_rl_agent.acceptance.protocol.v6"
)
NOISE_AWARE_DELTA_RULE = "bootstrap_floor_then_noise_aware_delta_v1"
PAIRED_HEADROOM_DELTA_RULE = "bootstrap_floor_then_paired_headroom_delta_v1"
GUARDED_PAIRED_HEADROOM_DELTA_RULE = (
    "bootstrap_guard_floor_then_paired_headroom_delta_v1"
)
BOUNDED_PAIRED_GUARD_HEADROOM_DELTA_RULE = (
    "bootstrap_bounded_paired_guard_then_paired_headroom_delta_v1"
)
NAMED_GUARDS_PAIRED_HEADROOM_DELTA_RULE = (
    "bootstrap_named_guard_floors_then_paired_headroom_delta_v1"
)
PUBLIC_SCORE_BINDING_ACCEPTANCE_MEAN = "acceptance_score_mean"
PUBLIC_SCORE_BINDING_GUARD_MEAN = "guard_score_mean"

_ACCEPTANCE_DELTA_REASONS_BY_RULE = {
    NOISE_AWARE_DELTA_RULE: (
        "noise_aware_delta",
        "noise_aware_delta_not_met",
    ),
    PAIRED_HEADROOM_DELTA_RULE: (
        "paired_headroom_delta",
        "paired_headroom_delta_not_met",
    ),
    GUARDED_PAIRED_HEADROOM_DELTA_RULE: (
        "guarded_paired_headroom_delta",
        "guarded_paired_headroom_delta_not_met",
    ),
    BOUNDED_PAIRED_GUARD_HEADROOM_DELTA_RULE: (
        "bounded_paired_guard_headroom_delta",
        "bounded_paired_guard_headroom_delta_not_met",
    ),
    NAMED_GUARDS_PAIRED_HEADROOM_DELTA_RULE: (
        "named_guards_paired_headroom_delta",
        "named_guards_paired_headroom_delta_not_met",
    ),
}
ACCEPTANCE_DELTA_ACCEPT_REASONS = frozenset(
    reasons[0] for reasons in _ACCEPTANCE_DELTA_REASONS_BY_RULE.values()
)
ACCEPTANCE_DELTA_REJECT_REASONS = frozenset(
    reasons[1] for reasons in _ACCEPTANCE_DELTA_REASONS_BY_RULE.values()
)


@dataclass(frozen=True)
class AcceptanceThreshold:
    """Canonical materiality threshold and its replayable components."""

    decision_rule: str
    threshold: float
    min_delta: float
    headroom_delta: float | None = None
    paired_standard_error: float | None = None
    paired_noise_delta: float | None = None
    pooled_standard_error: float | None = None

    def to_public_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "decision_rule": self.decision_rule,
            "min_delta": self.min_delta,
            "threshold": self.threshold,
        }
        for name in (
            "headroom_delta",
            "paired_standard_error",
            "paired_noise_delta",
            "pooled_standard_error",
        ):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        return payload


@dataclass(frozen=True)
class GuardRegressionThreshold:
    """Canonical bounded guard tolerance and its replayable components."""

    decision_rule: str
    threshold: float
    hard_cap: float
    base_margin: float | None = None
    paired_standard_error: float | None = None
    paired_noise_allowance: float | None = None

    def to_public_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "decision_rule": self.decision_rule,
            "threshold": self.threshold,
            "hard_cap": self.hard_cap,
        }
        for name in (
            "base_margin",
            "paired_standard_error",
            "paired_noise_allowance",
        ):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        return payload


@dataclass(frozen=True)
class NamedGuardSpec:
    """One task-named score protected by task-neutral acceptance math."""

    name: str
    score_source: str
    absolute_accept_floor: float
    max_regression: float
    regression_margin: float

    def __post_init__(self) -> None:
        for field_name in ("name", "score_source"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"named guard {field_name} must be non-empty")
        for field_name in (
            "absolute_accept_floor",
            "max_regression",
            "regression_margin",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(
                    f"named guard {field_name} must be a finite non-negative number"
                )
        if self.regression_margin > self.max_regression:
            raise ValueError(
                "named guard regression_margin cannot exceed max_regression"
            )

    @classmethod
    def from_mapping(cls, payload: Any) -> "NamedGuardSpec":
        if not isinstance(payload, Mapping) or set(payload) != {
            "name",
            "score_source",
            "absolute_accept_floor",
            "max_regression",
            "regression_margin",
        }:
            raise ValueError("named guard fields mismatch")
        return cls(
            name=_required_text(payload, "name"),
            score_source=_required_text(payload, "score_source"),
            absolute_accept_floor=_required_number(
                payload, "absolute_accept_floor"
            ),
            max_regression=_required_number(payload, "max_regression"),
            regression_margin=_required_number(payload, "regression_margin"),
        )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score_source": self.score_source,
            "absolute_accept_floor": float(self.absolute_accept_floor),
            "max_regression": float(self.max_regression),
            "regression_margin": float(self.regression_margin),
        }


@dataclass(frozen=True)
class AcceptanceProtocol:
    """Runner-owned acceptance constants supplied by a task plugin."""

    protocol_id: str
    score_source: str
    min_score_count: int
    absolute_accept_floor: float
    min_delta: float
    public_score_binding: str
    decision_rule: str = NOISE_AWARE_DELTA_RULE
    score_ceiling: float | None = None
    headroom_reduction_fraction: float | None = None
    paired_standard_error_multiplier: float | None = None
    guard_score_source: str | None = None
    guard_absolute_accept_floor: float | None = None
    guard_max_regression: float | None = None
    guard_regression_margin: float | None = None
    max_public_to_sealed_score_drop: float | None = None
    named_guards: tuple[NamedGuardSpec, ...] = ()

    @property
    def public_score_source(self) -> str:
        """Return the protocol-owned source represented by ``public_score``."""

        if self.public_score_binding == PUBLIC_SCORE_BINDING_GUARD_MEAN:
            if self.guard_score_source is None:
                raise ValueError("guard-bound public score has no guard score source")
            return self.guard_score_source
        return self.score_source

    def __post_init__(self) -> None:
        if not isinstance(self.protocol_id, str) or not self.protocol_id:
            raise ValueError("acceptance protocol_id must be non-empty")
        if not isinstance(self.score_source, str) or not self.score_source:
            raise ValueError("acceptance score_source must be non-empty")
        allowed_public_bindings = {PUBLIC_SCORE_BINDING_ACCEPTANCE_MEAN}
        if self.decision_rule in {
            GUARDED_PAIRED_HEADROOM_DELTA_RULE,
            BOUNDED_PAIRED_GUARD_HEADROOM_DELTA_RULE,
        }:
            allowed_public_bindings = {PUBLIC_SCORE_BINDING_GUARD_MEAN}
        if self.public_score_binding not in allowed_public_bindings:
            raise ValueError(
                "unsupported acceptance public_score_binding: "
                f"{self.public_score_binding}"
            )
        if (
            isinstance(self.min_score_count, bool)
            or not isinstance(self.min_score_count, int)
            or self.min_score_count <= 0
        ):
            raise ValueError("acceptance min_score_count must be a positive integer")
        for name in ("absolute_accept_floor", "min_delta"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"acceptance {name} must be a finite non-negative number")
        adaptive_fields = (
            self.score_ceiling,
            self.headroom_reduction_fraction,
            self.paired_standard_error_multiplier,
        )
        if self.decision_rule == NOISE_AWARE_DELTA_RULE:
            if any(value is not None for value in adaptive_fields):
                raise ValueError(
                    "noise-aware acceptance cannot define paired headroom fields"
                )
        elif self.decision_rule in {
            PAIRED_HEADROOM_DELTA_RULE,
            GUARDED_PAIRED_HEADROOM_DELTA_RULE,
            BOUNDED_PAIRED_GUARD_HEADROOM_DELTA_RULE,
            NAMED_GUARDS_PAIRED_HEADROOM_DELTA_RULE,
        }:
            if any(value is None for value in adaptive_fields):
                raise ValueError(
                    "paired headroom acceptance requires all adaptive fields"
                )
            assert self.score_ceiling is not None
            assert self.headroom_reduction_fraction is not None
            assert self.paired_standard_error_multiplier is not None
            for name in (
                "score_ceiling",
                "headroom_reduction_fraction",
                "paired_standard_error_multiplier",
            ):
                value = getattr(self, name)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int | float)
                    or not math.isfinite(float(value))
                    or float(value) <= 0.0
                ):
                    raise ValueError(
                        f"acceptance {name} must be a finite positive number"
                    )
            if self.headroom_reduction_fraction > 1.0:
                raise ValueError(
                    "acceptance headroom_reduction_fraction cannot exceed 1"
                )
            if self.absolute_accept_floor > self.score_ceiling:
                raise ValueError(
                    "acceptance absolute_accept_floor cannot exceed score_ceiling"
                )
            if self.min_delta > self.score_ceiling:
                raise ValueError("acceptance min_delta cannot exceed score_ceiling")
        else:
            raise ValueError(f"unsupported acceptance decision_rule: {self.decision_rule}")
        if not isinstance(self.named_guards, tuple) or any(
            not isinstance(item, NamedGuardSpec) for item in self.named_guards
        ):
            raise TypeError("acceptance named_guards must be a tuple of NamedGuardSpec")
        guard_names = tuple(item.name for item in self.named_guards)
        guard_sources = tuple(item.score_source for item in self.named_guards)
        if len(set(guard_names)) != len(guard_names):
            raise ValueError("acceptance named guard names must be unique")
        if len(set(guard_sources)) != len(guard_sources):
            raise ValueError("acceptance named guard score sources must be unique")
        if self.uses_named_guards:
            if not self.named_guards:
                raise ValueError("named-guard acceptance requires at least one guard")
            assert self.score_ceiling is not None
            for guard in self.named_guards:
                if guard.absolute_accept_floor > self.score_ceiling:
                    raise ValueError("named guard floor cannot exceed score_ceiling")
                if guard.max_regression > self.score_ceiling:
                    raise ValueError(
                        "named guard regression cannot exceed score_ceiling"
                    )
        elif self.named_guards:
            raise ValueError("non-named acceptance cannot define named_guards")
        guard_fields = (
            self.guard_score_source,
            self.guard_absolute_accept_floor,
            self.guard_max_regression,
        )
        if self.uses_guard_score:
            if any(value is None for value in guard_fields):
                raise ValueError("guarded acceptance requires all guard fields")
            if not isinstance(self.guard_score_source, str) or not self.guard_score_source:
                raise ValueError("acceptance guard_score_source must be non-empty")
            for name in ("guard_absolute_accept_floor", "guard_max_regression"):
                value = getattr(self, name)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int | float)
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                ):
                    raise ValueError(
                        f"acceptance {name} must be a finite non-negative number"
                    )
            assert self.score_ceiling is not None
            assert self.guard_absolute_accept_floor is not None
            assert self.guard_max_regression is not None
            if self.guard_absolute_accept_floor > self.score_ceiling:
                raise ValueError("acceptance guard floor cannot exceed score_ceiling")
            if self.guard_max_regression > self.score_ceiling:
                raise ValueError("acceptance guard regression cannot exceed score_ceiling")
        elif any(value is not None for value in guard_fields):
            raise ValueError("unguarded acceptance cannot define guard fields")
        bounded_guard_fields = (self.guard_regression_margin,)
        if self.requires_paired_guard_score_samples:
            if any(value is None for value in bounded_guard_fields):
                raise ValueError(
                    "bounded paired guard acceptance requires all bounded guard fields"
                )
            for name in (
                "guard_regression_margin",
                "max_public_to_sealed_score_drop",
            ):
                value = getattr(self, name)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int | float)
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                ):
                    raise ValueError(
                        f"acceptance {name} must be a finite non-negative number"
                    )
            assert self.guard_regression_margin is not None
            assert self.guard_max_regression is not None
            assert self.score_ceiling is not None
            if self.guard_regression_margin > self.guard_max_regression:
                raise ValueError(
                    "acceptance guard_regression_margin cannot exceed guard_max_regression"
                )
        elif any(value is not None for value in bounded_guard_fields):
            raise ValueError(
                "non-bounded acceptance cannot define bounded guard fields"
            )
        if self.requires_paired_guard_score_samples or self.uses_named_guards:
            value = self.max_public_to_sealed_score_drop
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(
                    "bounded acceptance requires finite max_public_to_sealed_score_drop"
                )
            assert self.score_ceiling is not None
            if float(value) > self.score_ceiling:
                raise ValueError(
                    "acceptance max_public_to_sealed_score_drop cannot exceed score_ceiling"
                )
        elif self.max_public_to_sealed_score_drop is not None:
            raise ValueError(
                "non-bounded acceptance cannot define max_public_to_sealed_score_drop"
            )

    @property
    def requires_paired_score_samples(self) -> bool:
        return self.decision_rule in {
            PAIRED_HEADROOM_DELTA_RULE,
            GUARDED_PAIRED_HEADROOM_DELTA_RULE,
            BOUNDED_PAIRED_GUARD_HEADROOM_DELTA_RULE,
            NAMED_GUARDS_PAIRED_HEADROOM_DELTA_RULE,
        }

    @property
    def uses_guard_score(self) -> bool:
        return self.decision_rule in {
            GUARDED_PAIRED_HEADROOM_DELTA_RULE,
            BOUNDED_PAIRED_GUARD_HEADROOM_DELTA_RULE,
        }

    @property
    def requires_paired_guard_score_samples(self) -> bool:
        return self.decision_rule == BOUNDED_PAIRED_GUARD_HEADROOM_DELTA_RULE

    @property
    def uses_named_guards(self) -> bool:
        return self.decision_rule == NAMED_GUARDS_PAIRED_HEADROOM_DELTA_RULE

    @property
    def requires_paired_named_guard_score_samples(self) -> bool:
        return self.uses_named_guards

    @classmethod
    def from_mapping(cls, payload: Any) -> "AcceptanceProtocol":
        if not isinstance(payload, Mapping):
            raise ValueError("acceptance protocol must be a mapping")
        schema_version = payload.get("schema_version")
        if schema_version not in {
            ACCEPTANCE_PROTOCOL_SCHEMA_VERSION,
            PAIRED_HEADROOM_ACCEPTANCE_PROTOCOL_SCHEMA_VERSION,
            GUARDED_PAIRED_HEADROOM_ACCEPTANCE_PROTOCOL_SCHEMA_VERSION,
            BOUNDED_PAIRED_GUARD_ACCEPTANCE_PROTOCOL_SCHEMA_VERSION,
            NAMED_GUARDS_ACCEPTANCE_PROTOCOL_SCHEMA_VERSION,
        }:
            raise ValueError("acceptance protocol schema_version mismatch")
        adaptive = schema_version in {
            PAIRED_HEADROOM_ACCEPTANCE_PROTOCOL_SCHEMA_VERSION,
            GUARDED_PAIRED_HEADROOM_ACCEPTANCE_PROTOCOL_SCHEMA_VERSION,
            BOUNDED_PAIRED_GUARD_ACCEPTANCE_PROTOCOL_SCHEMA_VERSION,
            NAMED_GUARDS_ACCEPTANCE_PROTOCOL_SCHEMA_VERSION,
        }
        guarded = schema_version in {
            GUARDED_PAIRED_HEADROOM_ACCEPTANCE_PROTOCOL_SCHEMA_VERSION,
            BOUNDED_PAIRED_GUARD_ACCEPTANCE_PROTOCOL_SCHEMA_VERSION,
        }
        bounded_guard = (
            schema_version == BOUNDED_PAIRED_GUARD_ACCEPTANCE_PROTOCOL_SCHEMA_VERSION
        )
        named_guards = (
            schema_version == NAMED_GUARDS_ACCEPTANCE_PROTOCOL_SCHEMA_VERSION
        )
        expected_fields = {
            "schema_version",
            "protocol_id",
            "score_source",
            "min_score_count",
            "absolute_accept_floor",
            "min_delta",
            "public_score_binding",
            "decision_rule",
            "acceptance_protocol_hash",
        }
        if adaptive:
            expected_fields.update(
                {
                    "score_ceiling",
                    "headroom_reduction_fraction",
                    "paired_standard_error_multiplier",
                }
            )
        if guarded:
            expected_fields.update(
                {
                    "guard_score_source",
                    "guard_absolute_accept_floor",
                    "guard_max_regression",
                }
            )
        if bounded_guard:
            expected_fields.update(
                {
                    "guard_regression_margin",
                    "max_public_to_sealed_score_drop",
                }
            )
        if named_guards:
            expected_fields.update({"named_guards", "max_public_to_sealed_score_drop"})
        if set(payload) != expected_fields:
            raise ValueError("acceptance protocol fields mismatch")
        protocol = cls(
            protocol_id=_required_text(payload, "protocol_id"),
            score_source=_required_text(payload, "score_source"),
            min_score_count=_required_positive_int(payload, "min_score_count"),
            absolute_accept_floor=_required_number(payload, "absolute_accept_floor"),
            min_delta=_required_number(payload, "min_delta"),
            public_score_binding=_required_text(payload, "public_score_binding"),
            decision_rule=_required_text(payload, "decision_rule"),
            score_ceiling=(
                _required_number(payload, "score_ceiling") if adaptive else None
            ),
            headroom_reduction_fraction=(
                _required_number(payload, "headroom_reduction_fraction")
                if adaptive
                else None
            ),
            paired_standard_error_multiplier=(
                _required_number(payload, "paired_standard_error_multiplier")
                if adaptive
                else None
            ),
            guard_score_source=(
                _required_text(payload, "guard_score_source") if guarded else None
            ),
            guard_absolute_accept_floor=(
                _required_number(payload, "guard_absolute_accept_floor")
                if guarded
                else None
            ),
            guard_max_regression=(
                _required_number(payload, "guard_max_regression")
                if guarded
                else None
            ),
            guard_regression_margin=(
                _required_number(payload, "guard_regression_margin")
                if bounded_guard
                else None
            ),
            max_public_to_sealed_score_drop=(
                _required_number(payload, "max_public_to_sealed_score_drop")
                if bounded_guard or named_guards
                else None
            ),
            named_guards=(
                tuple(
                    NamedGuardSpec.from_mapping(item)
                    for item in _required_mapping_sequence(payload, "named_guards")
                )
                if named_guards
                else ()
            ),
        )
        expected_schema = (
            NAMED_GUARDS_ACCEPTANCE_PROTOCOL_SCHEMA_VERSION
            if protocol.uses_named_guards
            else BOUNDED_PAIRED_GUARD_ACCEPTANCE_PROTOCOL_SCHEMA_VERSION
            if protocol.requires_paired_guard_score_samples
            else GUARDED_PAIRED_HEADROOM_ACCEPTANCE_PROTOCOL_SCHEMA_VERSION
            if protocol.uses_guard_score
            else PAIRED_HEADROOM_ACCEPTANCE_PROTOCOL_SCHEMA_VERSION
            if protocol.requires_paired_score_samples
            else ACCEPTANCE_PROTOCOL_SCHEMA_VERSION
        )
        if schema_version != expected_schema:
            raise ValueError("acceptance protocol schema and decision_rule mismatch")
        declared_hash = _required_text(payload, "acceptance_protocol_hash")
        if declared_hash != protocol.protocol_hash:
            raise ValueError("acceptance protocol hash mismatch")
        return protocol

    @property
    def protocol_hash(self) -> str:
        encoded = json.dumps(
            self._hash_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_public_dict(self) -> dict[str, Any]:
        return {
            **self._hash_payload(),
            "acceptance_protocol_hash": self.protocol_hash,
        }

    def _hash_payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": (
                BOUNDED_PAIRED_GUARD_ACCEPTANCE_PROTOCOL_SCHEMA_VERSION
                if self.requires_paired_guard_score_samples
                else NAMED_GUARDS_ACCEPTANCE_PROTOCOL_SCHEMA_VERSION
                if self.uses_named_guards
                else GUARDED_PAIRED_HEADROOM_ACCEPTANCE_PROTOCOL_SCHEMA_VERSION
                if self.uses_guard_score
                else PAIRED_HEADROOM_ACCEPTANCE_PROTOCOL_SCHEMA_VERSION
                if self.requires_paired_score_samples
                else ACCEPTANCE_PROTOCOL_SCHEMA_VERSION
            ),
            "protocol_id": self.protocol_id,
            "score_source": self.score_source,
            "min_score_count": int(self.min_score_count),
            "absolute_accept_floor": float(self.absolute_accept_floor),
            "min_delta": float(self.min_delta),
            "public_score_binding": self.public_score_binding,
            "decision_rule": self.decision_rule,
        }
        if self.requires_paired_score_samples:
            assert self.score_ceiling is not None
            assert self.headroom_reduction_fraction is not None
            assert self.paired_standard_error_multiplier is not None
            payload.update(
                {
                    "score_ceiling": float(self.score_ceiling),
                    "headroom_reduction_fraction": float(
                        self.headroom_reduction_fraction
                    ),
                    "paired_standard_error_multiplier": float(
                        self.paired_standard_error_multiplier
                    ),
                }
            )
        if self.uses_guard_score:
            assert self.guard_score_source is not None
            assert self.guard_absolute_accept_floor is not None
            assert self.guard_max_regression is not None
            payload.update(
                {
                    "guard_score_source": self.guard_score_source,
                    "guard_absolute_accept_floor": float(
                        self.guard_absolute_accept_floor
                    ),
                    "guard_max_regression": float(self.guard_max_regression),
                }
            )
        if self.requires_paired_guard_score_samples:
            assert self.guard_regression_margin is not None
            assert self.max_public_to_sealed_score_drop is not None
            payload.update(
                {
                    "guard_regression_margin": float(
                        self.guard_regression_margin
                    ),
                    "max_public_to_sealed_score_drop": float(
                        self.max_public_to_sealed_score_drop
                    ),
                }
            )
        if self.uses_named_guards:
            assert self.max_public_to_sealed_score_drop is not None
            payload.update(
                {
                    "named_guards": [
                        guard.to_public_dict() for guard in self.named_guards
                    ],
                    "max_public_to_sealed_score_drop": float(
                        self.max_public_to_sealed_score_drop
                    ),
                }
            )
        return payload


def acceptance_threshold(
    protocol: AcceptanceProtocol,
    *,
    candidate_score: float | None = None,
    best_score: float | None = None,
    candidate_score_samples: Sequence[float] | None = None,
    best_score_samples: Sequence[float] | None = None,
    candidate_standard_error: float | None = None,
    best_standard_error: float | None = None,
) -> AcceptanceThreshold:
    """Compute the single threshold used by live decisions and ledger replay."""

    if protocol.decision_rule == NOISE_AWARE_DELTA_RULE:
        pooled = None
        if candidate_standard_error is not None and best_standard_error is not None:
            pooled = math.sqrt(
                candidate_standard_error**2 + best_standard_error**2
            )
        elif candidate_standard_error is not None:
            pooled = candidate_standard_error
        threshold = max(
            protocol.min_delta,
            pooled if pooled is not None else protocol.min_delta,
        )
        return AcceptanceThreshold(
            decision_rule=protocol.decision_rule,
            threshold=threshold,
            min_delta=protocol.min_delta,
            pooled_standard_error=pooled,
        )

    if protocol.decision_rule not in {
        PAIRED_HEADROOM_DELTA_RULE,
        GUARDED_PAIRED_HEADROOM_DELTA_RULE,
        BOUNDED_PAIRED_GUARD_HEADROOM_DELTA_RULE,
        NAMED_GUARDS_PAIRED_HEADROOM_DELTA_RULE,
    }:
        raise ValueError(f"unsupported acceptance decision_rule: {protocol.decision_rule}")
    if candidate_score is None or best_score is None:
        raise ValueError("paired headroom acceptance requires both bound scores")
    candidate = validated_score_samples(
        candidate_score_samples,
        expected_mean=candidate_score,
        protocol=protocol,
        field_name="candidate_score_samples",
    )
    best = validated_score_samples(
        best_score_samples,
        expected_mean=best_score,
        protocol=protocol,
        field_name="best_score_samples",
    )
    if len(candidate) != len(best):
        raise ValueError("paired headroom score sample counts differ")
    paired_deltas = [left - right for left, right in zip(candidate, best, strict=True)]
    paired_standard_error = (
        statistics.stdev(paired_deltas) / math.sqrt(len(paired_deltas))
        if len(paired_deltas) > 1
        else 0.0
    )
    assert protocol.score_ceiling is not None
    assert protocol.headroom_reduction_fraction is not None
    assert protocol.paired_standard_error_multiplier is not None
    headroom_delta = protocol.headroom_reduction_fraction * max(
        0.0,
        protocol.score_ceiling - best_score,
    )
    paired_noise_delta = (
        protocol.paired_standard_error_multiplier * paired_standard_error
    )
    threshold = max(protocol.min_delta, headroom_delta, paired_noise_delta)
    return AcceptanceThreshold(
        decision_rule=protocol.decision_rule,
        threshold=threshold,
        min_delta=protocol.min_delta,
        headroom_delta=headroom_delta,
        paired_standard_error=paired_standard_error,
        paired_noise_delta=paired_noise_delta,
    )


def noise_aware_threshold(
    protocol: AcceptanceProtocol,
    *,
    candidate_standard_error: float | None,
    best_standard_error: float | None,
) -> float:
    """Compute the one canonical threshold used by decisions and ledger replay."""

    return acceptance_threshold(
        protocol,
        candidate_standard_error=candidate_standard_error,
        best_standard_error=best_standard_error,
    ).threshold


def acceptance_delta_reason(protocol: AcceptanceProtocol, *, accepted: bool) -> str:
    try:
        accept_reason, reject_reason = _ACCEPTANCE_DELTA_REASONS_BY_RULE[
            protocol.decision_rule
        ]
    except KeyError as exc:
        raise ValueError(
            f"unsupported acceptance decision_rule: {protocol.decision_rule}"
        ) from exc
    return accept_reason if accepted else reject_reason


def acceptance_delta_met(
    protocol: AcceptanceProtocol,
    delta: float,
    threshold: float,
) -> bool:
    """Apply strict materiality without accepting floating-point equality noise."""

    if not math.isfinite(delta) or not math.isfinite(threshold):
        return False
    if protocol.decision_rule == NOISE_AWARE_DELTA_RULE:
        return delta > threshold
    return delta > threshold and not math.isclose(
        delta,
        threshold,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    )


def guard_regression_threshold(
    protocol: AcceptanceProtocol,
    *,
    candidate_guard_score: float | None,
    best_guard_score: float | None,
    candidate_guard_score_samples: Sequence[float] | None = None,
    best_guard_score_samples: Sequence[float] | None = None,
) -> GuardRegressionThreshold:
    """Compute the one bounded guard tolerance used by decisions and replay."""

    if not protocol.uses_guard_score or protocol.guard_max_regression is None:
        raise ValueError("guard regression threshold requires a guarded protocol")
    hard_cap = float(protocol.guard_max_regression)
    if not protocol.requires_paired_guard_score_samples:
        return GuardRegressionThreshold(
            decision_rule=protocol.decision_rule,
            threshold=hard_cap,
            hard_cap=hard_cap,
        )
    if candidate_guard_score is None or best_guard_score is None:
        raise ValueError("paired guard threshold requires both bound guard scores")
    candidate = validated_score_samples(
        candidate_guard_score_samples,
        expected_mean=candidate_guard_score,
        protocol=protocol,
        field_name="candidate_guard_score_samples",
    )
    best = validated_score_samples(
        best_guard_score_samples,
        expected_mean=best_guard_score,
        protocol=protocol,
        field_name="best_guard_score_samples",
    )
    if len(candidate) != len(best):
        raise ValueError("paired guard score sample counts differ")
    paired_deltas = [left - right for left, right in zip(candidate, best, strict=True)]
    paired_standard_error = (
        statistics.stdev(paired_deltas) / math.sqrt(len(paired_deltas))
        if len(paired_deltas) > 1
        else 0.0
    )
    assert protocol.guard_regression_margin is not None
    assert protocol.paired_standard_error_multiplier is not None
    paired_noise_allowance = (
        protocol.paired_standard_error_multiplier * paired_standard_error
    )
    threshold = min(
        hard_cap,
        protocol.guard_regression_margin + paired_noise_allowance,
    )
    return GuardRegressionThreshold(
        decision_rule=protocol.decision_rule,
        threshold=threshold,
        hard_cap=hard_cap,
        base_margin=protocol.guard_regression_margin,
        paired_standard_error=paired_standard_error,
        paired_noise_allowance=paired_noise_allowance,
    )


def named_guard_regression_threshold(
    protocol: AcceptanceProtocol,
    guard: NamedGuardSpec,
    *,
    candidate_guard_score: float | None,
    best_guard_score: float | None,
    candidate_guard_score_samples: Sequence[float] | None,
    best_guard_score_samples: Sequence[float] | None,
) -> GuardRegressionThreshold:
    """Compute one v6 guard tolerance without interpreting its task-owned name."""

    if not protocol.uses_named_guards or guard not in protocol.named_guards:
        raise ValueError("named guard threshold requires a declared v6 guard")
    if candidate_guard_score is None or best_guard_score is None:
        raise ValueError("paired named guard threshold requires both bound scores")
    candidate = validated_score_samples(
        candidate_guard_score_samples,
        expected_mean=candidate_guard_score,
        protocol=protocol,
        field_name=f"candidate_named_guard_samples:{guard.name}",
    )
    best = validated_score_samples(
        best_guard_score_samples,
        expected_mean=best_guard_score,
        protocol=protocol,
        field_name=f"best_named_guard_samples:{guard.name}",
    )
    if len(candidate) != len(best):
        raise ValueError("paired named guard score sample counts differ")
    paired_deltas = [left - right for left, right in zip(candidate, best, strict=True)]
    paired_standard_error = (
        statistics.stdev(paired_deltas) / math.sqrt(len(paired_deltas))
        if len(paired_deltas) > 1
        else 0.0
    )
    assert protocol.paired_standard_error_multiplier is not None
    paired_noise_allowance = (
        protocol.paired_standard_error_multiplier * paired_standard_error
    )
    threshold = min(
        guard.max_regression,
        guard.regression_margin + paired_noise_allowance,
    )
    return GuardRegressionThreshold(
        decision_rule=protocol.decision_rule,
        threshold=threshold,
        hard_cap=guard.max_regression,
        base_margin=guard.regression_margin,
        paired_standard_error=paired_standard_error,
        paired_noise_allowance=paired_noise_allowance,
    )


def validated_score_samples(
    value: Sequence[float] | None,
    *,
    expected_mean: float,
    protocol: AcceptanceProtocol,
    field_name: str,
) -> tuple[float, ...]:
    if value is None or isinstance(value, str | bytes):
        raise ValueError(f"{field_name} must be a numeric sequence")
    samples: list[float] = []
    for raw in value:
        if isinstance(raw, bool) or not isinstance(raw, int | float):
            raise ValueError(f"{field_name} must contain only numbers")
        parsed = float(raw)
        if not math.isfinite(parsed):
            raise ValueError(f"{field_name} must contain only finite numbers")
        if protocol.score_ceiling is not None and parsed > protocol.score_ceiling:
            raise ValueError(f"{field_name} exceeds score_ceiling")
        samples.append(parsed)
    if len(samples) != protocol.min_score_count:
        raise ValueError(f"{field_name} count differs from the paired cohort")
    if not math.isclose(
        statistics.fmean(samples),
        expected_mean,
        rel_tol=1.0e-9,
        abs_tol=1.0e-9,
    ):
        raise ValueError(f"{field_name} mean mismatch")
    return tuple(samples)


def threshold_components_match(
    declared: Any,
    expected: Mapping[str, Any],
) -> bool:
    """Compare a flat threshold projection with strict shape and numeric tolerance."""

    if not isinstance(declared, Mapping) or set(declared) != set(expected):
        return False
    for key, expected_value in expected.items():
        declared_value = declared[key]
        if isinstance(expected_value, bool):
            if type(declared_value) is not bool or declared_value is not expected_value:
                return False
        elif isinstance(expected_value, int | float):
            if (
                isinstance(declared_value, bool)
                or not isinstance(declared_value, int | float)
                or not math.isfinite(float(declared_value))
                or not math.isfinite(float(expected_value))
                or not math.isclose(
                    float(declared_value),
                    float(expected_value),
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-15,
                )
            ):
                return False
        elif (
            type(declared_value) is not type(expected_value)
            or declared_value != expected_value
        ):
            return False
    return True


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"acceptance {key} must be non-empty")
    return value


def _required_positive_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"acceptance {key} must be a positive integer")
    return value


def _required_number(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"acceptance {key} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"acceptance {key} must be finite")
    return parsed


def _required_mapping_sequence(
    payload: Mapping[str, Any], key: str
) -> tuple[Mapping[str, Any], ...]:
    value = payload.get(key)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"acceptance {key} must be a sequence")
    if any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"acceptance {key} must contain only objects")
    return tuple(value)


__all__ = [
    "ACCEPTANCE_PROTOCOL_SCHEMA_VERSION",
    "ACCEPTANCE_DELTA_ACCEPT_REASONS",
    "ACCEPTANCE_DELTA_REJECT_REASONS",
    "PAIRED_HEADROOM_ACCEPTANCE_PROTOCOL_SCHEMA_VERSION",
    "GUARDED_PAIRED_HEADROOM_ACCEPTANCE_PROTOCOL_SCHEMA_VERSION",
    "BOUNDED_PAIRED_GUARD_ACCEPTANCE_PROTOCOL_SCHEMA_VERSION",
    "NAMED_GUARDS_ACCEPTANCE_PROTOCOL_SCHEMA_VERSION",
    "AcceptanceThreshold",
    "GuardRegressionThreshold",
    "NamedGuardSpec",
    "AcceptanceProtocol",
    "NOISE_AWARE_DELTA_RULE",
    "PAIRED_HEADROOM_DELTA_RULE",
    "GUARDED_PAIRED_HEADROOM_DELTA_RULE",
    "BOUNDED_PAIRED_GUARD_HEADROOM_DELTA_RULE",
    "NAMED_GUARDS_PAIRED_HEADROOM_DELTA_RULE",
    "PUBLIC_SCORE_BINDING_ACCEPTANCE_MEAN",
    "PUBLIC_SCORE_BINDING_GUARD_MEAN",
    "acceptance_delta_met",
    "acceptance_delta_reason",
    "acceptance_threshold",
    "guard_regression_threshold",
    "named_guard_regression_threshold",
    "noise_aware_threshold",
    "threshold_components_match",
    "validated_score_samples",
]
