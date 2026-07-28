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
    "fb00275c563d71044516ec0263c649b9b4c20cf4ad7bfe1e5b2ad988462c48c2"
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
    "training_execution",
    "complete_reward_runtime",
    "inheritance",
    "draft_protocol_hash",
}


@dataclass(frozen=True)
class StairRewardInputFieldSpec:
    name: str
    dtype: str
    trailing_shape: tuple[int, ...]
    unit: str
    description: str

    @classmethod
    def from_mapping(cls, value: Any) -> "StairRewardInputFieldSpec":
        raw = _mapping(value)
        _require_exact_fields(
            raw,
            {"name", "dtype", "trailing_shape", "unit", "description"},
            "Stair reward input field",
        )
        name = _require_text(raw["name"], "Stair reward input field name")
        if not name.isidentifier() or name.startswith("_"):
            raise ValueError("Stair reward input field name is not public")
        dtype = raw["dtype"]
        if dtype not in {"float", "bool"}:
            raise ValueError("Stair reward input field dtype is invalid")
        shape = raw["trailing_shape"]
        if not isinstance(shape, list) or any(
            _positive_int(item, "Stair reward input dimension") != item
            for item in shape
        ):
            raise ValueError("Stair reward input trailing_shape is invalid")
        return cls(
            name=name,
            dtype=dtype,
            trailing_shape=tuple(shape),
            unit=_require_text(raw["unit"], "Stair reward input unit"),
            description=_require_text(
                raw["description"],
                "Stair reward input description",
            ),
        )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "trailing_shape": list(self.trailing_shape),
            "unit": self.unit,
            "description": self.description,
        }


@dataclass(frozen=True)
class StairCompleteRewardRuntimeContract:
    payload: Mapping[str, Any]
    input_fields: tuple[StairRewardInputFieldSpec, ...]
    input_schema_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.payload, Mapping):
            raise TypeError("Stair complete reward runtime payload must be an object")
        if not self.input_fields:
            raise ValueError("Stair complete reward input fields cannot be empty")
        require_sha256_digest(
            self.input_schema_hash,
            "Stair complete reward input schema hash",
        )
        object.__setattr__(
            self,
            "payload",
            _freeze_json(_json_copy(dict(self.payload))),
        )

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        base: StairProtocolV1,
    ) -> "StairCompleteRewardRuntimeContract":
        raw = _mapping(value)
        _require_exact_fields(
            raw,
            {
                "schema_version",
                "input_schema",
                "output_contract",
                "dispatcher",
                "reference_equivalence",
                "performance_gate",
            },
            "Stair complete reward runtime",
        )
        _require_text(
            raw["schema_version"],
            "Stair complete reward runtime schema_version",
        )

        input_schema = _mapping(raw["input_schema"])
        _require_exact_fields(
            input_schema,
            {"schema_version", "fields", "source_transforms"},
            "Stair reward input schema",
        )
        _require_text(
            input_schema["schema_version"],
            "Stair reward input schema_version",
        )
        raw_fields = input_schema["fields"]
        if not isinstance(raw_fields, list) or not raw_fields:
            raise ValueError("Stair reward input fields must be a non-empty list")
        fields = tuple(
            StairRewardInputFieldSpec.from_mapping(item) for item in raw_fields
        )
        names = tuple(item.name for item in fields)
        if len(names) != len(set(names)):
            raise ValueError("Stair reward input field names must be unique")
        common_names = tuple(
            str(item)
            for item in base.to_dict()["candidate_boundary"][
                "common_reward_input_fields"
            ]
        )
        if names[: len(common_names)] != common_names:
            raise ValueError(
                "Stair reward inputs must preserve the task-neutral field prefix"
            )
        source_transforms = _mapping(input_schema["source_transforms"])
        _require_exact_fields(
            source_transforms,
            {"height_scan"},
            "Stair reward input source transforms",
        )
        height_scan_transform = _mapping(source_transforms["height_scan"])
        _require_exact_fields(
            height_scan_transform,
            {
                "source_formula",
                "offset_m",
                "corruption",
                "ray_miss_policy",
                "clip_min",
                "clip_max",
            },
            "Stair height scan source transform",
        )
        for key in ("source_formula", "corruption", "ray_miss_policy"):
            _require_text(
                height_scan_transform[key],
                f"Stair height scan transform {key}",
            )
        _finite_number(
            height_scan_transform["offset_m"],
            "Stair height scan transform offset_m",
        )
        clip_min = _finite_number(
            height_scan_transform["clip_min"],
            "Stair height scan transform clip_min",
        )
        clip_max = _finite_number(
            height_scan_transform["clip_max"],
            "Stair height scan transform clip_max",
        )
        if clip_min >= clip_max:
            raise ValueError("Stair height scan transform clip interval is invalid")

        output = _mapping(raw["output_contract"])
        _require_exact_fields(
            output,
            {
                "shape",
                "dtype",
                "device",
                "finite_required",
                "term_bound_applies_to",
                "maximum_absolute_term_rate",
                "maximum_absolute_total_rate",
            },
            "Stair reward output contract",
        )
        if output["shape"] != ["num_envs"]:
            raise ValueError("Stair reward output shape must be one vector")
        if output["dtype"] != "float" or output["device"] != "same_as_inputs":
            raise ValueError("Stair reward output dtype/device contract is invalid")
        if output["finite_required"] is not True:
            raise ValueError("Stair reward output must require finite values")
        if output["term_bound_applies_to"] != "weighted_contribution_after_clip":
            raise ValueError("Stair reward term bound authority is invalid")
        for key in (
            "maximum_absolute_term_rate",
            "maximum_absolute_total_rate",
        ):
            if _finite_number(output[key], f"Stair reward output {key}") <= 0.0:
                raise ValueError(f"Stair reward output {key} must be positive")

        dispatcher = _mapping(raw["dispatcher"])
        _require_exact_fields(
            dispatcher,
            {
                "manager_term_name",
                "manager_term_weight",
                "program_calls_per_control_step",
                "term_calls_per_program_call",
                "determinism_preflight_repeats",
                "evidence_sample_interval_steps",
                "input_isolation",
                "rng_guard",
            },
            "Stair reward dispatcher",
        )
        if dispatcher["manager_term_name"] != "complete_reward":
            raise ValueError("Stair dispatcher manager term is not authoritative")
        if _finite_number(
            dispatcher["manager_term_weight"],
            "Stair dispatcher manager term weight",
        ) != 1.0:
            raise ValueError("Stair dispatcher manager term weight must be one")
        for key in (
            "program_calls_per_control_step",
            "term_calls_per_program_call",
        ):
            if _positive_int(dispatcher[key], f"Stair dispatcher {key}") != 1:
                raise ValueError(f"Stair dispatcher {key} must be one")
        if _positive_int(
            dispatcher["determinism_preflight_repeats"],
            "Stair dispatcher determinism repeats",
        ) < 2:
            raise ValueError("Stair dispatcher determinism preflight is too weak")
        _positive_int(
            dispatcher["evidence_sample_interval_steps"],
            "Stair dispatcher evidence interval",
        )
        _require_text(
            dispatcher["input_isolation"],
            "Stair dispatcher input isolation",
        )
        _require_text(dispatcher["rng_guard"], "Stair dispatcher RNG guard")

        equivalence = _mapping(raw["reference_equivalence"])
        _require_exact_fields(
            equivalence,
            {
                "upstream_project",
                "upstream_revision",
                "upstream_source_path",
                "absolute_tolerance",
                "relative_tolerance",
                "active_term_names",
                "zero_weight_term_names",
                "disabled_term_names",
            },
            "Stair reference equivalence",
        )
        for key in (
            "upstream_project",
            "upstream_revision",
            "upstream_source_path",
        ):
            _require_text(equivalence[key], f"Stair equivalence {key}")
        for key in ("absolute_tolerance", "relative_tolerance"):
            if _finite_number(equivalence[key], f"Stair equivalence {key}") < 0.0:
                raise ValueError(f"Stair equivalence {key} must be non-negative")
        term_groups = []
        for key in (
            "active_term_names",
            "zero_weight_term_names",
            "disabled_term_names",
        ):
            names_in_group = equivalence[key]
            if (
                not isinstance(names_in_group, list)
                or any(not isinstance(item, str) or not item for item in names_in_group)
                or len(names_in_group) != len(set(names_in_group))
            ):
                raise ValueError(f"Stair equivalence {key} is invalid")
            term_groups.append(set(names_in_group))
        if not term_groups[0]:
            raise ValueError("Stair equivalence active terms cannot be empty")
        if any(
            left & right
            for index, left in enumerate(term_groups)
            for right in term_groups[index + 1 :]
        ):
            raise ValueError("Stair equivalence term groups must be disjoint")

        performance = _mapping(raw["performance_gate"])
        _require_exact_fields(
            performance,
            {
                "num_envs",
                "warmup_control_steps",
                "measured_control_steps",
                "minimum_dispatcher_throughput_ratio",
            },
            "Stair reward performance gate",
        )
        for key in ("num_envs", "warmup_control_steps", "measured_control_steps"):
            _positive_int(performance[key], f"Stair performance {key}")
        ratio = _finite_number(
            performance["minimum_dispatcher_throughput_ratio"],
            "Stair performance throughput ratio",
        )
        if not 0.0 < ratio <= 1.0:
            raise ValueError("Stair performance throughput ratio must be in (0, 1]")

        return cls(
            payload=raw,
            input_fields=fields,
            input_schema_hash=canonical_json_hash(input_schema),
        )

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.input_fields)

    def field(self, name: str) -> StairRewardInputFieldSpec:
        for item in self.input_fields:
            if item.name == name:
                return item
        raise KeyError(name)

    def source_transform(self, name: str) -> Mapping[str, Any]:
        transforms = _mapping(self.payload["input_schema"])["source_transforms"]
        raw = _mapping(_mapping(transforms)[name])
        return MappingProxyType(_json_copy(dict(raw)))

    @property
    def output_contract(self) -> Mapping[str, Any]:
        return _mapping(self.payload["output_contract"])

    @property
    def dispatcher_contract(self) -> Mapping[str, Any]:
        return _mapping(self.payload["dispatcher"])

    def to_public_dict(self) -> dict[str, Any]:
        return _thaw_json(self.payload)


@dataclass(frozen=True)
class StairProtocolV2Draft:
    """Validated V2 overlay; all Stair semantics remain task-owned."""

    payload: Mapping[str, Any]
    capability_profile: CandidateCapabilityProfile
    complete_reward_runtime: StairCompleteRewardRuntimeContract
    protocol_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.payload, Mapping):
            raise TypeError("Stair V2 draft payload must be an object")
        if not isinstance(self.capability_profile, CandidateCapabilityProfile):
            raise TypeError("Stair V2 capability profile is invalid")
        if not isinstance(
            self.complete_reward_runtime,
            StairCompleteRewardRuntimeContract,
        ):
            raise TypeError("Stair V2 complete reward runtime is invalid")
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
        reward_runtime = StairCompleteRewardRuntimeContract.from_mapping(
            raw["complete_reward_runtime"],
            base=base,
        )
        return cls(
            payload=hash_payload,
            capability_profile=profile,
            complete_reward_runtime=reward_runtime,
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
            "reference_id",
            "repository_source_path",
            "canonical_source_path",
            "entrypoint",
            "session_role",
            "required_authorship_origin",
        },
        "Stair V2 draft reference_program",
    )
    validate_experiment_subject_id(str(reference["reference_id"]))
    repository_source_path = _require_text(
        reference["repository_source_path"],
        "Stair V2 draft reference repository source path",
    )
    repository_path = Path(repository_source_path)
    if repository_path.is_absolute() or ".." in repository_path.parts:
        raise ValueError("Stair V2 draft reference repository path is unsafe")
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

    training_execution = _mapping(raw["training_execution"])
    _require_exact_fields(
        training_execution,
        {
            "schema_version",
            "runner",
            "init_at_random_episode_length",
            "checkpoint_selector",
            "checkpoint_save_interval_iterations",
            "torch_matmul_allow_tf32",
            "cudnn_allow_tf32",
            "cudnn_deterministic",
            "cudnn_benchmark",
        },
        "Stair V2 draft training_execution",
    )
    for key in ("schema_version", "runner", "checkpoint_selector"):
        _require_text(
            training_execution[key],
            f"Stair V2 training execution {key}",
        )
    _positive_int(
        training_execution["checkpoint_save_interval_iterations"],
        "Stair V2 checkpoint save interval",
    )
    for key in (
        "init_at_random_episode_length",
        "torch_matmul_allow_tf32",
        "cudnn_allow_tf32",
        "cudnn_deterministic",
        "cudnn_benchmark",
    ):
        if not isinstance(training_execution[key], bool):
            raise ValueError(f"Stair V2 training execution {key} must be boolean")

    _mapping(raw["complete_reward_runtime"])

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
    "StairCompleteRewardRuntimeContract",
    "StairProtocolV2Draft",
    "StairRewardInputFieldSpec",
]
