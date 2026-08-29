"""Task-owned historical drafts and frozen production Stair Protocol V2."""

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
from true_rl_agent.evaluation.protocol import EvaluationProtocol
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
from true_rl_agent.tasks.stair_climbing.training_distribution import (
    STAIR_PREVIOUS_TRAINING_PROTOCOL_HASH,
    validate_stair_training_distribution_contract,
)
from true_rl_agent.training_programs import (
    CANONICAL_TRAINING_COMPONENTS_ENTRYPOINT,
    CANONICAL_TRAINING_COMPONENTS_PATH,
)

EXPECTED_STAIR_PROTOCOL_V2_DRAFT_HASH = (
    "fb00275c563d71044516ec0263c649b9b4c20cf4ad7bfe1e5b2ad988462c48c2"
)
EXPECTED_STAIR_PROTOCOL_V2_PILOT_DRAFT_HASH = (
    "b7ff0280faf98b59473088768b55b242caae03b6783f41bfe51b4db07b94f7be"
)
PREVIOUS_STAIR_PROTOCOL_V2_HASH = STAIR_PREVIOUS_TRAINING_PROTOCOL_HASH
EXPECTED_STAIR_PROTOCOL_V2_HASH = (
    "97d9473e4ba6ad5b9829fab26ad47d6d4bb89e9ad711d198629b17ccc8e153f5"
)
STAIR_PROTOCOL_V2_PILOT_ARTIFACT_MANIFEST_RELATIVE_PATH = (
    "configs/tasks/stair_protocol_v2_pilot_artifact_manifest.json"
)

_DRAFT_V1_SCHEMA_VERSION = "true_rl_agent.task.stair_climbing.protocol_v2_draft.v1"
_DRAFT_V4_SCHEMA_VERSION = "true_rl_agent.task.stair_climbing.protocol_v2_draft.v4"
_FINAL_SCHEMA_VERSION = "true_rl_agent.task.stair_climbing.protocol_v2.v3"
_TRUSTED_DRAFT_HASHES = {
    _DRAFT_V1_SCHEMA_VERSION: EXPECTED_STAIR_PROTOCOL_V2_DRAFT_HASH,
    _DRAFT_V4_SCHEMA_VERSION: EXPECTED_STAIR_PROTOCOL_V2_PILOT_DRAFT_HASH,
}

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
_TOP_LEVEL_FIELDS_V2 = _TOP_LEVEL_FIELDS | {"runner_owned_training_distribution"}
_FINAL_TOP_LEVEL_FIELDS = {
    "schema_version",
    "protocol_id",
    "task_id",
    "status",
    "base_protocol",
    "genesis_anchor_policy",
    "reference_program",
    "candidate_capability_profile",
    "production_training",
    "reference_capability_gate",
    "formal_protocol",
    "runner_owned_training_distribution",
    "training_execution",
    "complete_reward_runtime",
    "inheritance",
    "pilot_lineage",
    "protocol_hash",
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
        base: StairProtocolV1 | None = None,
        common_input_fields: tuple[str, ...] | None = None,
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
        if (base is None) == (common_input_fields is None):
            raise ValueError(
                "Stair reward contract requires exactly one common-input authority"
            )
        common_names = (
            tuple(str(item) for item in common_input_fields)
            if common_input_fields is not None
            else tuple(
                str(item)
                for item in base.to_dict()["candidate_boundary"][
                    "common_reward_input_fields"
                ]
            )
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
        schema_version = raw.get("schema_version")
        expected_fields = (
            _TOP_LEVEL_FIELDS_V2
            if schema_version == _DRAFT_V4_SCHEMA_VERSION
            else _TOP_LEVEL_FIELDS
        )
        _require_exact_fields(raw, expected_fields, "Stair V2 draft")
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
        trusted_hash = _TRUSTED_DRAFT_HASHES.get(schema_version)
        if trusted_hash is None or declared_hash != trusted_hash:
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
    def pilot_max_iterations(self) -> int:
        return int(self.pilot_training_contract["fixed_max_iterations"])

    @property
    def pilot_training_contract(self) -> Mapping[str, Any]:
        return _mapping(self.payload["pilot_training"])

    @property
    def pilot_training_seed_count(self) -> int:
        return int(self.pilot_training_contract["training_seed_count"])

    @property
    def runner_owned_training_distribution(self) -> Mapping[str, Any] | None:
        value = self.payload.get("runner_owned_training_distribution")
        if value is None:
            return None
        return _mapping(value)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            **_thaw_json(self.payload),
            "draft_protocol_hash": self.protocol_hash,
        }


@dataclass(frozen=True)
class StairProtocolV2:
    """Trusted production protocol with historical pilot and revision lineage."""

    payload: Mapping[str, Any]
    capability_profile: CandidateCapabilityProfile
    complete_reward_runtime: StairCompleteRewardRuntimeContract
    protocol_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.payload, Mapping):
            raise TypeError("Stair V2 payload must be an object")
        if not isinstance(self.capability_profile, CandidateCapabilityProfile):
            raise TypeError("Stair V2 capability profile is invalid")
        if not isinstance(
            self.complete_reward_runtime,
            StairCompleteRewardRuntimeContract,
        ):
            raise TypeError("Stair V2 complete reward runtime is invalid")
        require_sha256_digest(self.protocol_hash, "Stair V2 protocol hash")
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
    ) -> "StairProtocolV2":
        (
            hash_payload,
            profile,
            reward_runtime,
            declared_hash,
        ) = _parse_final_protocol(
            path,
            base_v1_path=base_v1_path,
            require_trusted_anchor=True,
        )
        return cls(
            payload=hash_payload,
            capability_profile=profile,
            complete_reward_runtime=reward_runtime,
            protocol_hash=declared_hash,
        )

    @property
    def fixed_budget_steps(self) -> int:
        return int(_mapping(self.payload["production_training"])["budget_steps"])

    @property
    def seed_anchor_hash(self) -> str:
        return str(_mapping(self.payload["pilot_lineage"])["draft_protocol_hash"])

    @property
    def runner_owned_training_distribution(self) -> Mapping[str, Any]:
        return _mapping(self.payload["runner_owned_training_distribution"])

    def to_public_dict(self) -> dict[str, Any]:
        return {**_thaw_json(self.payload), "protocol_hash": self.protocol_hash}


def validate_stair_protocol_v2_freeze_candidate(
    path: Path,
    *,
    base_v1_path: Path | None = None,
) -> str:
    """Validate generated final bytes before their hash becomes the trust anchor."""

    *_, declared_hash = _parse_final_protocol(
        path,
        base_v1_path=base_v1_path,
        require_trusted_anchor=False,
    )
    return declared_hash


def _parse_final_protocol(
    path: Path,
    *,
    base_v1_path: Path | None,
    require_trusted_anchor: bool,
) -> tuple[
    dict[str, Any],
    CandidateCapabilityProfile,
    StairCompleteRewardRuntimeContract,
    str,
]:
    raw = _read_exact_json(path, context="Stair V2")
    _require_exact_fields(raw, _FINAL_TOP_LEVEL_FIELDS, "Stair V2")
    declared_hash = require_sha256_digest(
        raw.get("protocol_hash"),
        "Stair V2 protocol hash",
    )
    hash_payload = {
        key: value for key, value in raw.items() if key != "protocol_hash"
    }
    if canonical_json_hash(hash_payload) != declared_hash:
        raise ValueError("Stair V2 protocol hash mismatch")
    if require_trusted_anchor and declared_hash != EXPECTED_STAIR_PROTOCOL_V2_HASH:
        raise ValueError("Stair V2 does not match the trusted hash anchor")
    resolved_v1_path = base_v1_path or path.with_name("stair_protocol_v1.json")
    base = StairProtocolV1.load(resolved_v1_path)
    _assert_final_structural_contract(raw, base=base)
    _validate_final_pilot_artifact_manifest(
        raw,
        base_v1_path=resolved_v1_path,
    )
    profile = CandidateCapabilityProfile.from_mapping(
        raw["candidate_capability_profile"],
        known_capability_ids=candidate_authoring_capability_ids(),
    )
    if profile.capability_ids != (COMPLETE_REWARD_PROGRAM_CAPABILITY_ID,):
        raise ValueError("Stair V2 must expose only complete reward authoring")
    reward_runtime = StairCompleteRewardRuntimeContract.from_mapping(
        raw["complete_reward_runtime"],
        base=base,
    )
    return hash_payload, profile, reward_runtime, declared_hash


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
    schema_version = raw["schema_version"]
    if schema_version not in _TRUSTED_DRAFT_HASHES:
        raise ValueError("Stair V2 draft schema_version is unsupported")

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
    if schema_version == _DRAFT_V4_SCHEMA_VERSION:
        _validate_fixed_pilot_contract(pilot)
    else:
        _validate_legacy_pilot_contract(pilot)

    gate = _mapping(raw["reference_capability_gate"])
    _validate_remaining_draft_contract(
        raw=raw,
        base=base,
        schema_version=schema_version,
        gate=gate,
    )


def _validate_fixed_pilot_contract(pilot: Mapping[str, Any]) -> None:
    _require_exact_fields(
        pilot,
        {
            "fixed_max_iterations",
            "budget_policy",
            "pilot_seed_domain",
            "training_seed_count",
            "curve_diagnostics",
            "pilot_artifacts_enter_genesis",
            "pilot_results_enter_acceptance_or_prompt",
        },
        "Stair V2 draft pilot_training",
    )
    _positive_int(
        pilot["fixed_max_iterations"],
        "Stair V2 fixed max iterations",
    )
    if pilot["budget_policy"] != (
        "operator_selected_fixed_budget_without_optimality_or_convergence_claim"
    ):
        raise ValueError("Stair V2 fixed pilot budget policy mismatch")
    if pilot["pilot_seed_domain"] != (
        "single_seed_disjoint_from_formal_reference_and_candidate_training"
    ):
        raise ValueError("Stair V2 pilot seed domain mismatch")
    if pilot["training_seed_count"] != 1:
        raise ValueError("Stair V2 capability confirmation uses one training seed")
    _validate_pilot_curve_diagnostics(pilot["curve_diagnostics"])
    for key in (
        "pilot_artifacts_enter_genesis",
        "pilot_results_enter_acceptance_or_prompt",
    ):
        if pilot[key] is not False:
            raise ValueError(f"Stair V2 draft {key} must be false")


def _validate_legacy_pilot_contract(pilot: Mapping[str, Any]) -> None:
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
        "legacy Stair V2 draft pilot_training",
    )
    budgets = pilot["allowed_budget_steps"]
    if (
        not isinstance(budgets, list)
        or not budgets
        or budgets != sorted(set(budgets))
        or any(_positive_int(item, "legacy Stair pilot budget") != item for item in budgets)
        or pilot["maximum_budget_steps"] != budgets[-1]
    ):
        raise ValueError("legacy Stair V2 draft pilot budgets are invalid")
    _require_text(pilot["selection_rule"], "legacy Stair pilot selection rule")
    _require_text(pilot["pilot_seed_domain"], "legacy Stair pilot seed domain")
    if pilot["pilot_artifacts_enter_genesis"] is not False or pilot[
        "pilot_results_enter_acceptance_or_prompt"
    ] is not False:
        raise ValueError("legacy Stair pilot boundary flags must be false")


def _validate_remaining_draft_contract(
    *,
    raw: Mapping[str, Any],
    schema_version: str,
    gate: Mapping[str, Any],
    base: StairProtocolV1,
) -> None:
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
    execution_fields = {
        "schema_version",
        "runner",
        "init_at_random_episode_length",
        "checkpoint_selector",
        "checkpoint_save_interval_iterations",
        "torch_matmul_allow_tf32",
        "cudnn_allow_tf32",
        "cudnn_deterministic",
        "cudnn_benchmark",
    }
    if schema_version == _DRAFT_V4_SCHEMA_VERSION:
        execution_fields |= {
            "observation_groups",
            "clip_actions",
            "empirical_normalization",
            "logger",
            "resume",
        }
    _require_exact_fields(
        training_execution,
        execution_fields,
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
    if schema_version == _DRAFT_V4_SCHEMA_VERSION:
        if training_execution["observation_groups"] != {
            "policy": ["policy"],
            "critic": ["policy"],
        }:
            raise ValueError("Stair V2 observation groups mismatch")
        if training_execution["clip_actions"] is not None:
            raise ValueError("Stair V2 action clipping must remain disabled")
        if training_execution["empirical_normalization"] is not None:
            raise ValueError("Stair V2 empirical normalization must remain unset")
        if training_execution["logger"] != "tensorboard":
            raise ValueError("Stair V2 logger must be tensorboard")
        if training_execution["resume"] is not False:
            raise ValueError("Stair V2 training cannot resume checkpoints")
        validate_stair_training_distribution_contract(
            raw["runner_owned_training_distribution"],
            base_protocol=base.to_dict(),
        )
    elif "runner_owned_training_distribution" in raw:
        raise ValueError("legacy Stair V2 draft cannot carry a training overlay")

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


def _assert_final_structural_contract(
    raw: Mapping[str, Any],
    *,
    base: StairProtocolV1,
) -> None:
    if raw["schema_version"] != _FINAL_SCHEMA_VERSION:
        raise ValueError("Stair V2 schema_version is unsupported")
    if raw["protocol_id"] != "unitree_stair_climbing_v2":
        raise ValueError("Stair V2 protocol_id mismatch")
    if raw["task_id"] != base.task_id:
        raise ValueError("Stair V2 task_id does not match base V1")
    if raw["status"] != "production_revised_for_formal_target_training_distribution":
        raise ValueError("Stair V2 production status mismatch")

    base_protocol = _mapping(raw["base_protocol"])
    _require_exact_fields(
        base_protocol,
        {"schema_version", "protocol_id", "task_protocol_hash"},
        "Stair V2 base_protocol",
    )
    expected_base = {
        "schema_version": base.to_dict()["schema_version"],
        "protocol_id": base.protocol_id,
        "task_protocol_hash": base.protocol_hash,
    }
    if dict(base_protocol) != expected_base:
        raise ValueError("Stair V2 base protocol binding mismatch")
    if dict(_mapping(raw["genesis_anchor_policy"])) != {
        "random": "reference_program"
    }:
        raise ValueError("Stair V2 Genesis policy is invalid")

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
        "Stair V2 reference_program",
    )
    validate_experiment_subject_id(str(reference["reference_id"]))
    repository_path = Path(
        _require_text(
            reference["repository_source_path"],
            "Stair V2 reference repository source path",
        )
    )
    if repository_path.is_absolute() or ".." in repository_path.parts:
        raise ValueError("Stair V2 reference repository path is unsafe")
    if reference["canonical_source_path"] != CANONICAL_TRAINING_COMPONENTS_PATH:
        raise ValueError("Stair V2 reference source path is not canonical")
    if reference["entrypoint"] != CANONICAL_TRAINING_COMPONENTS_ENTRYPOINT:
        raise ValueError("Stair V2 reference entrypoint is not canonical")
    if reference["session_role"] != "reference":
        raise ValueError("Stair V2 reference session role is invalid")
    _require_text(
        reference["required_authorship_origin"],
        "Stair V2 reference authorship origin",
    )

    production = _mapping(raw["production_training"])
    _require_exact_fields(
        production,
        {
            "schema_version",
            "budget_steps",
            "num_envs",
            "steps_per_env",
            "max_iterations",
            "runner_timeout_s",
            "train_profile",
            "eval_profile",
            "policy_initialization",
            "warm_start_allowed",
            "checkpoint_selector",
            "policy_architecture_hash",
            "optimizer_training_profile_hash",
            "agent_configuration_hash",
            "initialization_seed_schedule_hash",
            "private_training_seed_resolver_hash",
            "formal_training_binding_hash",
            "capability_profile_hash",
        },
        "Stair V2 production_training",
    )
    _require_text(production["schema_version"], "Stair V2 production schema")
    budget_steps = _positive_int(production["budget_steps"], "Stair V2 budget")
    num_envs = _positive_int(production["num_envs"], "Stair V2 train envs")
    steps_per_env = _positive_int(
        production["steps_per_env"],
        "Stair V2 train steps per env",
    )
    max_iterations = _positive_int(
        production["max_iterations"],
        "Stair V2 train max iterations",
    )
    if budget_steps != num_envs * steps_per_env * max_iterations:
        raise ValueError("Stair V2 production training budget equation mismatch")
    base_training = _mapping(base.to_dict()["training"])
    if num_envs != base_training["num_envs"]:
        raise ValueError("Stair V2 production train env count differs from V1")
    if steps_per_env != base_training["steps_per_env"]:
        raise ValueError("Stair V2 production rollout length differs from V1")
    if production["runner_timeout_s"] != base_training["runner_timeout_s"]:
        raise ValueError("Stair V2 production runner timeout differs from V1")
    if production["train_profile"] != "stair_v2_train":
        raise ValueError("Stair V2 production train profile mismatch")
    if production["eval_profile"] != "stair_v2_eval":
        raise ValueError("Stair V2 production eval profile mismatch")
    if production["policy_initialization"] != "random":
        raise ValueError("Stair V2 production training must start at random")
    if production["warm_start_allowed"] is not False:
        raise ValueError("Stair V2 production training cannot warm-start")
    if production["checkpoint_selector"] != "final_iteration":
        raise ValueError("Stair V2 production checkpoint selector mismatch")
    for field_name in (
        "policy_architecture_hash",
        "optimizer_training_profile_hash",
        "agent_configuration_hash",
        "initialization_seed_schedule_hash",
        "private_training_seed_resolver_hash",
        "formal_training_binding_hash",
        "capability_profile_hash",
    ):
        require_sha256_digest(
            production[field_name],
            f"Stair V2 production {field_name}",
        )

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
        "Stair V2 reference_capability_gate",
    )
    if gate["source_split"] != "public_accept":
        raise ValueError("Stair V2 reference capability split mismatch")
    if gate["seed_count"] != base.evaluation_protocol.public_accept_seed_count:
        raise ValueError("Stair V2 reference capability seed count mismatch")
    low_bins = gate["low_step_bin_ids"]
    if (
        not isinstance(low_bins, list)
        or not low_bins
        or len(low_bins) != len(set(low_bins))
        or any(not isinstance(item, str) or not item for item in low_bins)
    ):
        raise ValueError("Stair V2 low-step bins must be non-empty and unique")
    if not set(low_bins).issubset(
        {str(item["bin_id"]) for item in base.eval_bins}
    ):
        raise ValueError("Stair V2 low-step bins are not in base V1")
    success_floor = _finite_number(
        gate["minimum_macro_mean_success_rate"],
        "Stair V2 reference success floor",
    )
    progress_floor = _finite_number(
        gate["minimum_macro_mean_max_progress_m"],
        "Stair V2 reference progress floor",
    )
    if not 0.0 < success_floor <= 1.0 or progress_floor <= 0.0:
        raise ValueError("Stair V2 reference capability floors are invalid")
    if gate["threshold_relaxation_after_pilot_allowed"] is not False:
        raise ValueError("Stair V2 cannot relax the reference capability gate")

    formal = _mapping(raw["formal_protocol"])
    _require_exact_fields(
        formal,
        {
            "evaluation_protocol_version",
            "private_seed_resolver_version",
            "pilot_seed_overlap_allowed",
            "reference_candidate_training_binding_equal",
            "evaluation_protocol_hash",
            "acceptance_protocol_hash",
            "formal_evaluation_seed_cohort_hash",
            "pilot_evaluation_seed_cohort_hash",
            "task_protocol_binding",
        },
        "Stair V2 formal_protocol",
    )
    base_evaluation = base.evaluation_protocol
    expected_evaluation_hash = EvaluationProtocol(
        eval_protocol_version=formal["evaluation_protocol_version"],
        public_seed_count=base_evaluation.public_seed_count,
        public_accept_seed_count=base_evaluation.public_accept_seed_count,
        sealed_validation_seed_count=base_evaluation.sealed_validation_seed_count,
        eval_num_envs=base_evaluation.eval_num_envs,
    ).protocol_hash
    if formal["evaluation_protocol_hash"] != expected_evaluation_hash:
        raise ValueError("Stair V2 evaluation protocol hash mismatch")
    if formal["acceptance_protocol_hash"] != base.acceptance_protocol.protocol_hash:
        raise ValueError("Stair V2 acceptance protocol hash mismatch")
    for field_name in (
        "formal_evaluation_seed_cohort_hash",
        "pilot_evaluation_seed_cohort_hash",
    ):
        require_sha256_digest(formal[field_name], f"Stair V2 {field_name}")
    if formal["formal_evaluation_seed_cohort_hash"] == formal[
        "pilot_evaluation_seed_cohort_hash"
    ]:
        raise ValueError("Stair V2 formal and pilot evaluation cohorts overlap")
    if formal["pilot_seed_overlap_allowed"] is not False:
        raise ValueError("Stair V2 pilot seed overlap must remain false")
    if formal["reference_candidate_training_binding_equal"] is not True:
        raise ValueError("Stair V2 reference/candidate training bindings must match")
    if formal["task_protocol_binding"] != "final_protocol_hash_at_materialization":
        raise ValueError("Stair V2 formal task protocol binding is invalid")

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
            "observation_groups",
            "clip_actions",
            "empirical_normalization",
            "logger",
            "resume",
        },
        "Stair V2 training_execution",
    )
    if training_execution["observation_groups"] != {
        "policy": ["policy"],
        "critic": ["policy"],
    }:
        raise ValueError("Stair V2 observation groups mismatch")
    if training_execution["clip_actions"] is not None:
        raise ValueError("Stair V2 action clipping must remain disabled")
    if training_execution["empirical_normalization"] is not None:
        raise ValueError("Stair V2 empirical normalization must remain unset")
    if training_execution["logger"] != "tensorboard":
        raise ValueError("Stair V2 logger must be tensorboard")
    if training_execution["resume"] is not False:
        raise ValueError("Stair V2 training cannot resume checkpoints")
    if training_execution["checkpoint_selector"] != production[
        "checkpoint_selector"
    ]:
        raise ValueError("Stair V2 checkpoint selector has multiple values")
    validate_stair_training_distribution_contract(
        raw["runner_owned_training_distribution"],
        base_protocol=base.to_dict(),
    )
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
        "Stair V2 inheritance",
    )
    if any(
        value != "base_v1_by_task_protocol_hash"
        for value in inheritance.values()
    ):
        raise ValueError("Stair V2 inheritance must bind base V1 by hash")

    lineage = _mapping(raw["pilot_lineage"])
    _require_exact_fields(
        lineage,
        {
            "schema_version",
            "draft_protocol_hash",
            "phase_d_report_hash",
            "pilot_evidence_hash",
            "fixed_budget_policy_hash",
            "pilot_training_binding_hash",
            "pilot_training_seed_count",
            "pilot_training_seed_domain_hash",
            "pilot_evaluation_binding_hash",
            "pilot_artifact_uri",
            "pilot_artifact_content_hash",
            "pilot_artifact_manifest_hash",
            "pilot_artifact_manifest_repository_path",
            "boundaries",
        },
        "Stair V2 pilot_lineage",
    )
    if lineage["schema_version"] != (
        "true_rl_agent.task.stair_climbing.phase_d_freeze_lineage.v4"
    ):
        raise ValueError("Stair V2 pilot lineage schema mismatch")
    if lineage["draft_protocol_hash"] != EXPECTED_STAIR_PROTOCOL_V2_PILOT_DRAFT_HASH:
        raise ValueError("Stair V2 pilot lineage does not bind the trusted pilot draft")
    for field_name in (
        "phase_d_report_hash",
        "pilot_evidence_hash",
        "fixed_budget_policy_hash",
        "pilot_training_binding_hash",
        "pilot_training_seed_domain_hash",
        "pilot_evaluation_binding_hash",
        "pilot_artifact_content_hash",
        "pilot_artifact_manifest_hash",
    ):
        require_sha256_digest(lineage[field_name], f"Stair V2 {field_name}")
    expected_budget_policy_hash = canonical_json_hash(
        {
            "kind": (
                "operator_selected_fixed_budget_without_optimality_or_"
                "convergence_claim"
            ),
            "max_iterations": max_iterations,
            "budget_steps": budget_steps,
            "budget_optimality_claimed": False,
            "curve_convergence_claimed": False,
            "warm_start_allowed": False,
            "capability_confirmation_checkpoint": "final_iteration",
        }
    )
    if lineage["fixed_budget_policy_hash"] != expected_budget_policy_hash:
        raise ValueError("Stair V2 fixed budget policy lineage mismatch")
    if lineage["pilot_training_seed_count"] != 1:
        raise ValueError("Stair V2 pilot training seed count mismatch")
    if lineage["pilot_training_binding_hash"] == production[
        "formal_training_binding_hash"
    ]:
        raise ValueError("Stair V2 formal and pilot training bindings overlap")
    if lineage["pilot_artifact_uri"] != (
        "artifact://stair-phase-d/sha256/"
        + lineage["pilot_artifact_content_hash"]
    ):
        raise ValueError("Stair V2 pilot artifact URI mismatch")
    if lineage["pilot_artifact_manifest_repository_path"] != (
        STAIR_PROTOCOL_V2_PILOT_ARTIFACT_MANIFEST_RELATIVE_PATH
    ):
        raise ValueError("Stair V2 pilot artifact manifest path mismatch")
    if lineage["boundaries"] != {
        "enters_acceptance": False,
        "enters_genesis": False,
        "enters_production_prompt": False,
    }:
        raise ValueError("Stair V2 pilot boundary flags mismatch")


def _validate_final_pilot_artifact_manifest(
    raw: Mapping[str, Any],
    *,
    base_v1_path: Path,
) -> None:
    from true_rl_agent.tasks.stair_climbing.pilot_artifact_store import (
        validate_stair_phase_d_pilot_artifact_manifest,
    )

    lineage = _mapping(raw["pilot_lineage"])
    project_root = base_v1_path.parents[2]
    manifest_path = project_root / str(
        lineage["pilot_artifact_manifest_repository_path"]
    )
    manifest = load_strict_json_object(
        manifest_path,
        context="Stair V2 pilot artifact manifest",
        require_canonical=True,
        trailing_newline=True,
    )
    validated = validate_stair_phase_d_pilot_artifact_manifest(
        manifest=manifest
    )
    expected = {
        "draft_protocol_hash": lineage["draft_protocol_hash"],
        "phase_d_report_hash": lineage["phase_d_report_hash"],
        "artifact_uri": lineage["pilot_artifact_uri"],
        "content_hash": lineage["pilot_artifact_content_hash"],
        "manifest_hash": lineage["pilot_artifact_manifest_hash"],
    }
    if any(validated.get(key) != value for key, value in expected.items()):
        raise ValueError("Stair V2 pilot artifact manifest lineage mismatch")


def _validate_pilot_curve_diagnostics(value: Any) -> None:
    diagnostics = _mapping(value)
    _require_exact_fields(
        diagnostics,
        {
            "schema_version",
            "tail_window_iterations",
            "normalization",
            "required_scalar_tags",
        },
        "Stair pilot curve diagnostics",
    )
    if diagnostics["schema_version"] != (
        "true_rl_agent.task.stair_climbing.pilot_curve_diagnostics.v1"
    ):
        raise ValueError("Stair pilot curve diagnostics schema mismatch")
    if _positive_int(
        diagnostics["tail_window_iterations"],
        "Stair pilot curve diagnostic window",
    ) < 20:
        raise ValueError("Stair pilot curve diagnostic window is invalid")
    if diagnostics["normalization"] != (
        "absolute_linear_tail_change_over_population_stddev_with_1e-6_floor"
    ):
        raise ValueError("Stair pilot curve diagnostic normalization mismatch")
    if diagnostics["required_scalar_tags"] != [
        "Train/mean_reward",
        "Curriculum/terrain_levels",
    ]:
        raise ValueError("Stair pilot curve diagnostic scalar tags mismatch")


def _read_exact_json(path: Path, *, context: str = "Stair V2 draft") -> dict[str, Any]:
    try:
        return load_strict_json_object(
            path,
            context=context,
        )
    except ValueError as exc:
        raise ValueError(f"failed to load {context}: {exc}") from exc


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
    "EXPECTED_STAIR_PROTOCOL_V2_PILOT_DRAFT_HASH",
    "EXPECTED_STAIR_PROTOCOL_V2_HASH",
    "PREVIOUS_STAIR_PROTOCOL_V2_HASH",
    "STAIR_PROTOCOL_V2_PILOT_ARTIFACT_MANIFEST_RELATIVE_PATH",
    "StairCompleteRewardRuntimeContract",
    "StairProtocolV2",
    "StairProtocolV2Draft",
    "StairRewardInputFieldSpec",
    "validate_stair_protocol_v2_freeze_candidate",
]
