"""Public-safe provider failure records and repair feedback."""

from __future__ import annotations

import re
from typing import Any, Mapping

from true_rl_agent.agent_core.context import sanitize_public_context
from true_rl_agent.candidates.prompt_projection import assert_public_only_prompt_payload


PROVIDER_FAILURE_CATEGORIES = frozenset(
    {
        "json_parse_error",
        "schema_error",
        "candidate_materialize_error",
        "verifier_error",
        "runtime_error",
    }
)


class ProviderCandidateError(RuntimeError):
    """A provider-side candidate failure that may be repaired by another attempt."""

    def __init__(
        self,
        category: str,
        reason: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if category not in PROVIDER_FAILURE_CATEGORIES:
            raise ValueError(f"unknown provider failure category: {category}")
        self.category = category
        self.reason = _public_string(reason)
        self.details = _sanitize_provider_value(dict(details or {}))
        super().__init__(f"{self.category}: {self.reason}")

    def to_public_dict(self) -> dict[str, Any]:
        return _sanitize_provider_value(
            {
                "schema_version": "true_rl_agent.provider_failure.v1",
                "category": self.category,
                "reason": self.reason,
                "details": self.details,
            }
        )


def build_provider_repair_feedback(error: ProviderCandidateError) -> dict[str, Any]:
    failure = error.to_public_dict()
    feedback = _sanitize_provider_value(
        {
            "schema_version": "true_rl_agent.provider_repair_feedback.v1",
            "status": "repair_required",
            "provider_status": "fail",
            "failure_category": failure["category"],
            "reason": failure["reason"],
            "failure": failure,
            "instructions": [
                "Return one valid JSON object with candidate_id, rationale, declared_changed_components, and components_py.",
                "Fix only the candidate package response or training_code/components.py needed for this failure.",
                "Do not ask to change runner, eval protocol, training budget, seed policy, checkpoints, PPO, or acceptance rules.",
            ],
        }
    )
    assert_public_only_prompt_payload(feedback)
    return feedback


def _sanitize_provider_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return sanitize_public_context(
            {str(key): _sanitize_provider_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return [_sanitize_provider_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_provider_value(item) for item in value]
    if isinstance(value, str):
        return _public_string(value)
    return value


def _public_string(value: str) -> str:
    compact = " ".join(str(value).split())
    compact = re.sub(r"sk-[A-Za-z0-9_-]+", "[redacted]", compact)
    compact = re.sub(r"[A-Za-z]:\\[^\s\"']+", "[path]", compact)
    compact = re.sub(r"[A-Za-z]:/[^\s\"']+", "[path]", compact)
    compact = re.sub(r"/(?:data|home|mnt|tmp|Users|root|var|etc)/[^\s\"']+", "[path]", compact)
    return compact[:500]
