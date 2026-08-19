"""Dependency-light validation for prompt-facing exploration projections."""

from __future__ import annotations

from typing import Any, Mapping

from true_rl_agent.agent_core.public_safety import sanitize_prompt_public_value
from true_rl_agent.evaluation.binding import require_sha256_digest
from true_rl_agent.prompt_serialization import canonical_prompt_json_bytes


EXPLORATION_PROMPT_PROJECTION_SCHEMA_VERSION = (
    "true_rl_agent.exploration_prompt_projection.v2"
)
EXPLORATION_PROMPT_MAX_BYTES = 6 * 1024
MAX_EXPLORATION_PROMPT_INPUT_FIELDS = 32
MAX_EXPLORATION_PROMPT_TEXT_CHARS = 512

_CARD_FIELDS = frozenset(
    {
        "trial_id",
        "card_hash",
        "verifier_equivalence_key",
        "operation",
        "component_kind",
        "public_input_fields",
        "evidence_tier",
        "observed_effect",
        "value_label",
        "selection_status",
        "revisit_disposition",
        "public_evidence_summary",
        "reason_codes",
    }
)
_CARD_REQUIRED_FIELDS = frozenset(
    {
        "trial_id",
        "card_hash",
        "verifier_equivalence_key",
        "operation",
        "component_kind",
        "public_input_fields",
        "observed_effect",
        "value_label",
        "revisit_disposition",
    }
)


def compact_exploration_prompt_projection(value: Any) -> dict[str, Any]:
    """Validate and allowlist one already-fitted prompt projection."""

    return _normalize_exploration_prompt_projection(
        value,
        require_within_byte_limit=True,
    )


def fit_exploration_prompt_projection(value: Any) -> dict[str, Any]:
    """Deterministically fit a rich projection into the prompt envelope.

    Full evidence cards belong to the disposable exploration index and may be
    larger than this prompt-facing view. Every step strictly changes the
    normalized projection, so an oversized card can be omitted but can never
    make a long-running generation spin.
    """

    candidate = _normalize_exploration_prompt_projection(
        value,
        require_within_byte_limit=False,
    )
    maximum_steps = (
        len(candidate["priority_or_eligible_cards"])
        + len(candidate["verifier_equivalent_avoidance_cards"])
        + len(candidate["repeated_negative_input_signature_advisories"])
        + 2
    )
    for _ in range(maximum_steps):
        if canonical_prompt_json_bytes(candidate) <= EXPLORATION_PROMPT_MAX_BYTES:
            return compact_exploration_prompt_projection(candidate)
        degraded = _degrade_normalized_projection(candidate)
        if degraded is None:
            raise ValueError(
                "minimum exploration prompt projection exceeds its byte limit"
            )
        normalized = _normalize_exploration_prompt_projection(
            degraded,
            require_within_byte_limit=False,
        )
        if normalized == candidate:
            raise ValueError("exploration prompt compaction made no progress")
        candidate = normalized
    raise ValueError("exploration prompt compaction did not converge")


def _normalize_exploration_prompt_projection(
    value: Any,
    *,
    require_within_byte_limit: bool,
) -> dict[str, Any]:
    """Validate one projection shape, optionally enforcing its total size."""

    mapping = _exact_mapping(
        value,
        {
            "schema_version",
            "task_id",
            "source_index_hash",
            "current_head_hash",
            "semantics",
            "priority_or_eligible_cards",
            "verifier_equivalent_avoidance_cards",
            "repeated_negative_input_signature_advisories",
            "blocked_verifier_equivalence_count",
            "omitted_verifier_equivalent_avoidance_count",
            "frontier_backlog_count",
            "omitted_frontier_card_count",
            "deferred_frontier_card_count",
            "ineligible_frontier_card_count",
            "acceptance_authority",
        },
        "exploration prompt projection",
    )
    if (
        mapping["schema_version"]
        != EXPLORATION_PROMPT_PROJECTION_SCHEMA_VERSION
        or mapping["acceptance_authority"] is not False
    ):
        raise ValueError("exploration prompt projection contract mismatch")
    _bounded_text(mapping["task_id"], "exploration prompt task_id")
    require_sha256_digest(
        mapping["source_index_hash"], "exploration prompt source index hash"
    )
    require_sha256_digest(
        mapping["current_head_hash"], "exploration prompt current head hash"
    )
    _bounded_text(mapping["semantics"], "exploration prompt semantics")
    backlog = _nonnegative_int(
        mapping["frontier_backlog_count"],
        "exploration prompt frontier backlog count",
    )
    omitted = _nonnegative_int(
        mapping["omitted_frontier_card_count"],
        "exploration prompt omitted frontier count",
    )
    ineligible = _nonnegative_int(
        mapping["ineligible_frontier_card_count"],
        "exploration prompt ineligible frontier count",
    )
    blocked = _nonnegative_int(
        mapping["blocked_verifier_equivalence_count"],
        "exploration prompt blocked verifier equivalence count",
    )
    omitted_equivalent = _nonnegative_int(
        mapping["omitted_verifier_equivalent_avoidance_count"],
        "exploration prompt omitted verifier-equivalent avoidance count",
    )
    deferred = _nonnegative_int(
        mapping["deferred_frontier_card_count"],
        "exploration prompt deferred frontier count",
    )
    positive = _compact_cards(
        mapping["priority_or_eligible_cards"],
        maximum=2,
        label="priority exploration card",
    )
    negative = _compact_cards(
        mapping["verifier_equivalent_avoidance_cards"],
        maximum=2,
        label="verifier-equivalent-avoidance exploration card",
    )
    advisories = _compact_input_signature_advisories(
        mapping["repeated_negative_input_signature_advisories"]
    )
    if (
        blocked != len(negative) + omitted_equivalent
        or len(positive) > backlog
        or omitted != backlog - len(positive)
        or deferred > omitted
    ):
        raise ValueError("exploration prompt omission accounting mismatch")
    compact = _public_mapping(
        {
            "schema_version": EXPLORATION_PROMPT_PROJECTION_SCHEMA_VERSION,
            "task_id": mapping["task_id"],
            "source_index_hash": mapping["source_index_hash"],
            "current_head_hash": mapping["current_head_hash"],
            "semantics": mapping["semantics"],
            "priority_or_eligible_cards": positive,
            "verifier_equivalent_avoidance_cards": negative,
            "repeated_negative_input_signature_advisories": advisories,
            "blocked_verifier_equivalence_count": blocked,
            "omitted_verifier_equivalent_avoidance_count": omitted_equivalent,
            "frontier_backlog_count": backlog,
            "omitted_frontier_card_count": omitted,
            "deferred_frontier_card_count": deferred,
            "ineligible_frontier_card_count": ineligible,
            "acceptance_authority": False,
        }
    )
    if (
        require_within_byte_limit
        and canonical_prompt_json_bytes(compact) > EXPLORATION_PROMPT_MAX_BYTES
    ):
        raise ValueError("exploration prompt projection exceeds its byte limit")
    return compact


def degrade_exploration_prompt_projection(value: Any) -> dict[str, Any] | None:
    """Return the next deterministic lower-detail memory projection."""

    compact = compact_exploration_prompt_projection(value)
    candidate = _degrade_normalized_projection(compact)
    return (
        compact_exploration_prompt_projection(candidate)
        if candidate is not None
        else None
    )


def _degrade_normalized_projection(
    compact: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return one strictly lower-detail step from a normalized projection."""

    positive = list(compact["priority_or_eligible_cards"])
    negative = list(compact["verifier_equivalent_avoidance_cards"])
    advisories = list(compact["repeated_negative_input_signature_advisories"])
    omitted = int(compact["omitted_frontier_card_count"])
    omitted_equivalent = int(
        compact["omitted_verifier_equivalent_avoidance_count"]
    )
    if len(positive) > 1:
        positive.pop()
        omitted += 1
    elif len(negative) > 1:
        negative.pop()
        omitted_equivalent += 1
    elif len(advisories) > 1:
        advisories.pop()
    elif positive and set(positive[0]) != set(
        minimal_exploration_prompt_card(positive[0])
    ):
        positive[0] = minimal_exploration_prompt_card(positive[0])
    elif negative:
        negative.pop()
        omitted_equivalent += 1
    elif advisories:
        advisories.pop()
    elif positive:
        positive.pop()
        omitted += 1
    else:
        return None
    candidate = dict(compact)
    candidate["priority_or_eligible_cards"] = positive
    candidate["verifier_equivalent_avoidance_cards"] = negative
    candidate["repeated_negative_input_signature_advisories"] = advisories
    candidate["omitted_frontier_card_count"] = omitted
    candidate["omitted_verifier_equivalent_avoidance_count"] = (
        omitted_equivalent
    )
    return candidate


def _compact_input_signature_advisories(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > 2:
        raise ValueError("negative input-signature advisories are invalid")
    result: list[dict[str, Any]] = []
    fields = {
        "component_kind",
        "public_input_fields",
        "negative_trial_count",
        "distinct_verifier_equivalence_count",
        "latest_generation_index",
        "blocking",
        "reason",
        "instruction",
    }
    for item in value:
        mapping = _exact_mapping(
            item,
            fields,
            "negative input-signature advisory",
        )
        _bounded_text(mapping["component_kind"], "advisory component kind")
        input_fields = mapping["public_input_fields"]
        if (
            not isinstance(input_fields, list)
            or not input_fields
            or len(input_fields) > MAX_EXPLORATION_PROMPT_INPUT_FIELDS
        ):
            raise ValueError("advisory public input fields are invalid")
        for field in input_fields:
            _bounded_text(field, "advisory public input field")
        negative_count = _nonnegative_int(
            mapping["negative_trial_count"],
            "advisory negative trial count",
        )
        distinct_count = _nonnegative_int(
            mapping["distinct_verifier_equivalence_count"],
            "advisory distinct verifier equivalence count",
        )
        generation_index = _nonnegative_int(
            mapping["latest_generation_index"],
            "advisory latest generation index",
        )
        if (
            negative_count < 2
            or distinct_count < 2
            or distinct_count > negative_count
            or generation_index <= 0
            or mapping["blocking"] is not False
        ):
            raise ValueError("negative input-signature advisory contract mismatch")
        _bounded_text(mapping["reason"], "advisory reason")
        _bounded_text(mapping["instruction"], "advisory instruction")
        result.append(_public_mapping(mapping))
    return result


def _compact_cards(value: Any, *, maximum: int, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError(f"{label}s are invalid")
    result: list[dict[str, Any]] = []
    for item in value:
        if (
            not isinstance(item, Mapping)
            or not set(item).issubset(_CARD_FIELDS)
            or not _CARD_REQUIRED_FIELDS.issubset(item)
        ):
            raise ValueError(f"{label} fields mismatch")
        _bounded_text(item["trial_id"], f"{label} trial_id")
        require_sha256_digest(item["card_hash"], f"{label} hash")
        require_sha256_digest(
            item["verifier_equivalence_key"],
            f"{label} verifier equivalence key",
        )
        for key in ("operation", "component_kind", "observed_effect", "value_label", "revisit_disposition"):
            _bounded_text(item[key], f"{label} {key}")
        fields = item["public_input_fields"]
        if (
            not isinstance(fields, list)
            or len(fields) > MAX_EXPLORATION_PROMPT_INPUT_FIELDS
        ):
            raise ValueError(f"{label} public input fields are invalid")
        for field in fields:
            _bounded_text(field, f"{label} public input field")
        if "reason_codes" in item:
            reasons = item["reason_codes"]
            if not isinstance(reasons, list) or len(reasons) > 8:
                raise ValueError(f"{label} reason codes are invalid")
            for reason in reasons:
                _bounded_text(reason, f"{label} reason code")
        if "public_evidence_summary" in item:
            summary = _public_mapping(item["public_evidence_summary"])
            if canonical_prompt_json_bytes(summary) > 4096:
                raise ValueError(f"{label} evidence summary exceeds its byte limit")
        result.append(_public_mapping(dict(item)))
    return result


def minimal_exploration_prompt_card(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in _CARD_REQUIRED_FIELDS
        if key in value
    }


def _public_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("public exploration value must be an object")
    clean = sanitize_prompt_public_value(dict(value))
    if not isinstance(clean, Mapping):
        raise ValueError("public exploration value was not preserved")
    return dict(clean)


def _bounded_text(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > MAX_EXPLORATION_PROMPT_TEXT_CHARS
    ):
        raise ValueError(f"{label} must be a non-empty bounded string")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _exact_mapping(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields mismatch")
    return dict(value)


__all__ = [
    "EXPLORATION_PROMPT_MAX_BYTES",
    "EXPLORATION_PROMPT_PROJECTION_SCHEMA_VERSION",
    "MAX_EXPLORATION_PROMPT_INPUT_FIELDS",
    "MAX_EXPLORATION_PROMPT_TEXT_CHARS",
    "compact_exploration_prompt_projection",
    "degrade_exploration_prompt_projection",
    "fit_exploration_prompt_projection",
    "minimal_exploration_prompt_card",
]
