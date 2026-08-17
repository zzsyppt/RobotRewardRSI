"""Read-only identity validation for frozen historical Stair Phase D evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.json_contract import load_strict_json_object
from true_rl_agent.tasks.stair_climbing.config import STAIR_TASK_ID

STAIR_PHASE_D_PROMPT_PREFLIGHT_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.phase_d_prompt_preflight.v1"
)
EXPECTED_STAIR_PHASE_D_PROMPT_PREFLIGHT_REPORT_HASH = (
    "c939988ad491f9e1661dc94fb2ac39508382a39f08a51bbbe66a405010c16a83"
)
_REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "task_id",
        "status",
        "protocol_binding",
        "capability_exposure",
        "reference_source",
        "prompt_payload",
        "provider_candidate_response",
        "tokenizer_evidence",
        "source_read_budget",
        "report_hash",
    }
)
_FORBIDDEN_REPORT_KEYS = frozenset(
    {
        "checkpoint",
        "components_py",
        "env_origin",
        "pilot",
        "raw_seed",
        "seed_value",
        "source_text",
    }
)


def validate_stair_phase_d_prompt_preflight_report(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the exact committed public-safe historical report identity."""

    if not isinstance(value, Mapping):
        raise TypeError("Stair Phase D prompt preflight report must be an object")
    report = dict(value)
    if set(report) != _REPORT_FIELDS:
        raise ValueError("Stair Phase D prompt preflight report fields mismatch")
    if report.get("schema_version") != STAIR_PHASE_D_PROMPT_PREFLIGHT_SCHEMA_VERSION:
        raise ValueError("Stair Phase D prompt preflight schema mismatch")
    if report.get("task_id") != STAIR_TASK_ID or report.get("status") != "pass":
        raise ValueError("Stair Phase D prompt preflight did not pass")
    _assert_report_contains_no_private_or_source_payload(report)

    report_hash = canonical_json_hash(
        {key: item for key, item in report.items() if key != "report_hash"}
    )
    if report.get("report_hash") != report_hash:
        raise ValueError("Stair Phase D prompt preflight report hash mismatch")
    if report_hash != EXPECTED_STAIR_PHASE_D_PROMPT_PREFLIGHT_REPORT_HASH:
        raise ValueError("Stair Phase D frozen report identity mismatch")
    return report


def load_stair_phase_d_prompt_preflight_report(path: Path) -> dict[str, Any]:
    """Load canonical frozen report bytes without constructing active config."""

    if not isinstance(path, Path):
        raise TypeError("Stair Phase D prompt preflight path must be a Path")
    report = load_strict_json_object(
        path,
        context="Stair Phase D prompt preflight report",
        require_canonical=True,
        trailing_newline=True,
    )
    return validate_stair_phase_d_prompt_preflight_report(report)


def _assert_report_contains_no_private_or_source_payload(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if any(forbidden in lowered for forbidden in _FORBIDDEN_REPORT_KEYS):
                raise ValueError(
                    "Stair Phase D report contains private/source payload field"
                )
            _assert_report_contains_no_private_or_source_payload(item)
        return
    if isinstance(value, list):
        for item in value:
            _assert_report_contains_no_private_or_source_payload(item)
        return
    if isinstance(value, str) and "def build_training_components" in value:
        raise ValueError("Stair Phase D report contains raw R0 source")


__all__ = [
    "EXPECTED_STAIR_PHASE_D_PROMPT_PREFLIGHT_REPORT_HASH",
    "STAIR_PHASE_D_PROMPT_PREFLIGHT_SCHEMA_VERSION",
    "load_stair_phase_d_prompt_preflight_report",
    "validate_stair_phase_d_prompt_preflight_report",
]
