"""Task-owned frozen draft contract for Stair Protocol V2."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from true_rl_agent.candidates.capability_profile import (
    CandidateCapabilityProfile,
)
from true_rl_agent.evaluation.binding import (
    canonical_json_hash,
    require_sha256_digest,
)
from true_rl_agent.evaluation.subject import validate_experiment_subject_id
from true_rl_agent.json_contract import (
    canonical_json_text,
    load_strict_json_object,
    strict_json_loads,
)
from true_rl_agent.prompting.capabilities import (
    COMPLETE_REWARD_PROGRAM_CAPABILITY_ID,
    candidate_authoring_capability_ids,
)
from true_rl_agent.tasks.stair_climbing.phase0_contract import StairProtocolV1
from true_rl_agent.training_programs import (
    CANONICAL_TRAINING_COMPONENTS_ENTRYPOINT,
    CANONICAL_TRAINING_COMPONENTS_PATH,
)

EXPECTED_STAIR_PROTOCOL_V2_DRAFT_HASH = (
    "5074190752a3fed504e2c69b9e0c45efbde1dc185ab3ed56eefe69a3f242fd8f"
)

_TOP_LEVEL_FIELDS = {
    "schema_version",
    "protocol_id",
    "task_id",
    "status",
    "base_protocol",
    "genesis_anchor_policy",
    "reference_program",
    "candidate_capability_profile",
    "pilot_training",
    "reference_capability_gate",
    "formal_protocol",
    "inheritance",
    "draft_protocol_hash",
}


@dataclass(frozen=True)
class StairProtocolV2Draft:
    """Validated V2 overlay; all Stair semantics remain task-owned."""

    payload: Mapping[str, Any]
    capability_profile: CandidateCapabilityProfile
    protocol_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.payload, Mapping):
            raise TypeError("Stair V2 draft payload must be an object")
        if not isinstance(self.capability_profile, CandidateCapabilityProfile):
            raise TypeError("Stair V2 capability profile is invalid")
        require_sha256_digest(self.protocol_hash, "Stair V2 draft protocol hash")
        object.__setattr__(
            self,
            "payload",
            _freeze_json(_json_copy(dict(self.payload))),
        )

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        base_v1_path: Path | None = None,
    ) -> "StairProtocolV2Draft":
        raw = _read_exact_json(path)
        _require_exact_fields(raw, _TOP_LEVEL_FIELDS, "Stair V2 draft")
        declared_hash = require_sha256_digest(
            raw.get("draft_protocol_hash"),
            "Stair V2 draft protocol hash",
        )
        hash_payload = {
            key: value
            for key, value in raw.items()
            if key != "draft_protocol_hash"
        }
        if canonical_json_hash(hash_payload) != declared_hash:
            raise ValueError("Stair V2 draft protocol hash mismatch")
        if declared_hash != EXPECTED_STAIR_PROTOCOL_V2_DRAFT_HASH:
            raise ValueError("Stair V2 draft does not match the trusted hash anchor")

        resolved_v1_path = base_v1_path or path.with_name(
            "stair_protocol_v1.json"
        )
        base = StairProtocolV1.load(resolved_v1_path)
        _assert_structural_contract(raw, base=base)

        profile = CandidateCapabilityProfile.from_mapping(
            raw["candidate_capability_profile"],
            known_capability_ids=candidate_authoring_capability_ids(),
        )
        if COMPLETE_REWARD_PROGRAM_CAPABILITY_ID not in profile.capability_ids:
            raise ValueError(
                "Stair V2 draft capability profile must include complete reward"
            )
        return cls(
            payload=hash_payload,
            capability_profile=profile,
            protocol_hash=declared_hash,
        )

    @property
    def allowed_pilot_budget_steps(self) -> tuple[int, ...]:
        pilot = _mapping(self.payload["pilot_training"])
        return tuple(pilot["allowed_budget_steps"])

    def to_public_dict(self) -> dict[str, Any]:
        return {
            **_thaw_json(self.payload),
            "draft_protocol_hash": self.protocol_hash,
        }


def _assert_structural_contract(
    raw: Mapping[str, Any],
    *,
    base: StairProtocolV1,
) -> None:
    for field_name in ("schema_version", "protocol_id", "task_id", "status"):
        _require_text(raw.get(field_name), f"Stair V2 draft {field_name}")
    validate_experiment_subject_id(str(raw["protocol_id"]))
    validate_experiment_subject_id(str(raw["task_id"]))
    if raw["task_id"] != base.task_id:
        raise ValueError("Stair V2 draft task_id does not match base V1")

    base_protocol = _mapping(raw["base_protocol"])
    _require_exact_fields(
        base_protocol,
        {"schema_version", "protocol_id", "task_protocol_hash"},
        "Stair V2 draft base_protocol",
    )
    if base_protocol["protocol_id"] != base.protocol_id:
        raise ValueError("Stair V2 draft base protocol_id mismatch")
    if base_protocol["task_protocol_hash"] != base.protocol_hash:
        raise ValueError("Stair V2 draft base protocol hash mismatch")

    genesis = _mapping(raw["genesis_anchor_policy"])
    if dict(genesis) != {"random": "reference_program"}:
        raise ValueError("Stair V2 draft Genesis policy is invalid")

    reference = _mapping(raw["reference_program"])
    _require_exact_fields(
        reference,
        {
            "canonical_source_path",
            "entrypoint",
            "session_role",
            "required_authorship_origin",
        },
        "Stair V2 draft reference_program",
    )
    if reference["canonical_source_path"] != CANONICAL_TRAINING_COMPONENTS_PATH:
        raise ValueError("Stair V2 draft reference source path is not canonical")
    if reference["entrypoint"] != CANONICAL_TRAINING_COMPONENTS_ENTRYPOINT:
        raise ValueError("Stair V2 draft reference entrypoint is not canonical")
    if reference["session_role"] != "reference":
        raise ValueError("Stair V2 draft reference session role is invalid")
    _require_text(
        reference["required_authorship_origin"],
        "Stair V2 draft reference authorship origin",
    )

    pilot = _mapping(raw["pilot_training"])
    _require_exact_fields(
        pilot,
        {
            "allowed_budget_steps",
            "maximum_budget_steps",
            "selection_rule",
            "pilot_seed_domain",
            "pilot_artifacts_enter_genesis",
            "pilot_results_enter_acceptance_or_prompt",
        },
        "Stair V2 draft pilot_training",
    )
    budgets = pilot["allowed_budget_steps"]
    if not isinstance(budgets, list) or not budgets:
        raise ValueError("Stair V2 draft pilot budgets must be a non-empty list")
    if any(_positive_int(item, "Stair V2 pilot budget") != item for item in budgets):
        raise ValueError("Stair V2 draft pilot budgets are invalid")
    if budgets != sorted(set(budgets)):
        raise ValueError("Stair V2 draft pilot budgets must be sorted and unique")
    if pilot["maximum_budget_steps"] != budgets[-1]:
        raise ValueError("Stair V2 draft maximum pilot budget mismatch")
    _require_text(pilot["selection_rule"], "Stair V2 draft pilot selection rule")
    _require_text(pilot["pilot_seed_domain"], "Stair V2 draft pilot seed domain")
    for key in (
        "pilot_artifacts_enter_genesis",
        "pilot_results_enter_acceptance_or_prompt",
    ):
        if pilot[key] is not False:
            raise ValueError(f"Stair V2 draft {key} must be false")

    gate = _mapping(raw["reference_capability_gate"])
    _require_exact_fields(
        gate,
        {
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
        },
        "Stair V2 draft reference_capability_gate",
    )
    _positive_int(gate["seed_count"], "Stair V2 reference seed_count")
    low_bins = gate["low_step_bin_ids"]
    if (
        not isinstance(low_bins, list)
        or not low_bins
        or any(not isinstance(item, str) or not item for item in low_bins)
        or len(low_bins) != len(set(low_bins))
    ):
        raise ValueError("Stair V2 draft low-step bins must be non-empty and unique")
    base_bin_ids = {str(item["bin_id"]) for item in base.eval_bins}
    if not set(low_bins).issubset(base_bin_ids):
        raise ValueError("Stair V2 draft low-step bins are not in base V1")
    success_floor = _finite_number(
        gate["minimum_macro_mean_success_rate"],
        "Stair V2 reference success floor",
    )
    if not 0.0 < success_floor <= 1.0:
        raise ValueError("Stair V2 draft success floor must be in (0, 1]")
    if _finite_number(
        gate["minimum_macro_mean_max_progress_m"],
        "Stair V2 reference progress floor",
    ) <= 0.0:
        raise ValueError("Stair V2 draft progress floor must be positive")
    if gate["threshold_relaxation_after_pilot_allowed"] is not False:
        raise ValueError("Stair V2 draft cannot relax the capability threshold")
    for key in ("schema_version", "source_split", "bin_coverage", "aggregation", "predicate"):
        _require_text(gate[key], f"Stair V2 draft gate {key}")

    formal = _mapping(raw["formal_protocol"])
    _require_exact_fields(
        formal,
        {
            "evaluation_protocol_version",
            "private_seed_resolver_version",
            "pilot_seed_overlap_allowed",
            "reference_candidate_training_binding_equal",
        },
        "Stair V2 draft formal_protocol",
    )
    _require_text(
        formal["evaluation_protocol_version"],
        "Stair V2 evaluation protocol version",
    )
    _require_text(
        formal["private_seed_resolver_version"],
        "Stair V2 private seed resolver version",
    )
    if formal["pilot_seed_overlap_allowed"] is not False:
        raise ValueError("Stair V2 draft pilot seed overlap must be false")
    if formal["reference_candidate_training_binding_equal"] is not True:
        raise ValueError("Stair V2 draft training binding equality must be true")

    inheritance = _mapping(raw["inheritance"])
    _require_exact_fields(
        inheritance,
        {
            "height_bins",
            "episode_outcomes",
            "evaluation_statistics",
            "episode_horizon_and_timeout",
        },
        "Stair V2 draft inheritance",
    )
    if any(
        value != "base_v1_by_task_protocol_hash"
        for value in inheritance.values()
    ):
        raise ValueError("Stair V2 draft inheritance must bind base V1 by hash")


def _read_exact_json(path: Path) -> dict[str, Any]:
    try:
        return load_strict_json_object(
            path,
            context="Stair V2 draft",
        )
    except ValueError as exc:
        raise ValueError(f"failed to load Stair V2 draft: {exc}") from exc


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: set[str],
    field_name: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{field_name} fields mismatch")


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Stair V2 draft nested value must be an object")
    return value


def _json_copy(value: Any) -> Any:
    return strict_json_loads(
        canonical_json_text(value),
        context="Stair V2 JSON value",
    )


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _finite_number(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{field_name} must be a finite number")
    return float(value)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


__all__ = [
    "EXPECTED_STAIR_PROTOCOL_V2_DRAFT_HASH",
    "StairProtocolV2Draft",
]
