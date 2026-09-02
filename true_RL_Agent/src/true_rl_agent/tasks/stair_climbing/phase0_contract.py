"""Strict, inert Phase 0 contracts for stair climbing."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from true_rl_agent.candidates.components import RewardInputs
from true_rl_agent.evaluation.acceptance_protocol import AcceptanceProtocol
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.protocol import EvaluationProtocol
from true_rl_agent.json_contract import (
    canonical_json_bytes,
    load_strict_json_object,
    strict_json_object_loads,
)
from true_rl_agent.runner.checkpoint import validate_checkpoint_safety_report

STAIR_EXTENSION_DESCRIPTOR_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.extension_descriptor.v1"
)
STAIR_PROTOCOL_SCHEMA_VERSION = "true_rl_agent.task.stair_climbing.protocol.v1"
STAIR_BASELINE_MANIFEST_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.baseline_manifest.v1"
)
STAIR_PHASE0_REPORT_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.phase0_report.v1"
)
STAIR_TASK_ID = "unitree_stair_climbing"
PHASE0_STAIR_GYM_ID = "TrueRLAgent-Phase0-Stair-G1-v0"
PHASE0_TRANSITION_EXTRAS_KEY = "stair_phase0_terminal_transition"
DEFAULT_STAIR_EXTENSION_DESCRIPTOR_REF = Path(
    "configs/tasks/stair_extension_schema.json"
)
FROZEN_STAIR_EXTENSION_DESCRIPTOR_V1_HASH = (
    "230529071a873e853cf16991dc28a7e5d1b82aba6ed840f5eef9618972a089f8"
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PLACEHOLDER_VALUES = {
    "ACTION_SCHEMA_HASH",
    "ACCEPTANCE_PROTOCOL_HASH",
    "BASELINE_MANIFEST_HASH",
    "EXTENSION_DESCRIPTOR_HASH",
    "EVALUATION_PROTOCOL_HASH",
    "OBSERVATION_SCHEMA_HASH",
    "ROBOT_SCHEMA_HASH",
    "TASK_PROTOCOL_HASH",
}
PHASE0_SOURCE_ARTIFACT_REFS = (
    "scripts/isaaclab_stair_phase0_probe.py",
    "src/true_rl_agent/tasks/stair_climbing/isaaclab_env_cfg.py",
    "src/true_rl_agent/tasks/stair_climbing/isaaclab_phase0_probe.py",
    "src/true_rl_agent/tasks/stair_climbing/isaaclab_terrain.py",
    "src/true_rl_agent/tasks/stair_climbing/phase0_contract.py",
    "src/true_rl_agent/runner/checkpoint.py",
)


@dataclass(frozen=True)
class StairExtensionDescriptorV1:
    """Single task-owned entrypoint for the frozen Phase 0 resource bundle."""

    _payload: Mapping[str, Any] = field(repr=False)

    @classmethod
    def load(cls, path: str | Path) -> "StairExtensionDescriptorV1":
        return cls.from_mapping(_load_json_object(Path(path)))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StairExtensionDescriptorV1":
        payload = _copy_json_object(value, "stair extension descriptor")
        _assert_resolved_json(payload, "stair extension descriptor")
        _validate_extension_descriptor(payload)
        return cls(_payload=_freeze_json(payload))

    @property
    def descriptor_hash(self) -> str:
        return str(self._payload["descriptor_hash"])

    def to_dict(self) -> dict[str, Any]:
        return _thaw_json(self._payload)


@dataclass(frozen=True)
class StairProtocolV1:
    """Validated immutable snapshot of the complete task-owned protocol."""

    _payload: Mapping[str, Any] = field(repr=False)
    evaluation_protocol: EvaluationProtocol
    acceptance_protocol: AcceptanceProtocol

    @classmethod
    def load(cls, path: str | Path) -> "StairProtocolV1":
        return cls.from_mapping(_load_json_object(Path(path)))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StairProtocolV1":
        payload = _copy_json_object(value, "stair protocol")
        _assert_resolved_json(payload, "stair protocol")
        _validate_stair_protocol(payload)
        evaluation_protocol = EvaluationProtocol.from_mapping(
            _object(payload["evaluation"], "evaluation")["protocol"]
        )
        acceptance_protocol = AcceptanceProtocol.from_mapping(
            _object(payload["acceptance"], "acceptance")["protocol"]
        )
        return cls(
            _payload=_freeze_json(payload),
            evaluation_protocol=evaluation_protocol,
            acceptance_protocol=acceptance_protocol,
        )

    @property
    def task_id(self) -> str:
        return str(self._payload["task_id"])

    @property
    def protocol_id(self) -> str:
        return str(self._payload["protocol_id"])

    @property
    def protocol_hash(self) -> str:
        return str(self._payload["task_protocol_hash"])

    @property
    def eval_bins(self) -> tuple[Mapping[str, Any], ...]:
        terrain = _object(self._payload["terrain"], "terrain")
        return tuple(_object(item, "evaluation bin") for item in terrain["evaluation_bins"])

    def to_dict(self) -> dict[str, Any]:
        return _thaw_json(self._payload)


@dataclass(frozen=True)
class StairBaselineManifestV1:
    """Validated immutable identity for the official G1 rough warm-start candidate."""

    _payload: Mapping[str, Any] = field(repr=False)

    @classmethod
    def load(cls, path: str | Path) -> "StairBaselineManifestV1":
        return cls.from_mapping(_load_json_object(Path(path)))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StairBaselineManifestV1":
        payload = _copy_json_object(value, "stair baseline manifest")
        _assert_resolved_json(payload, "stair baseline manifest")
        _validate_baseline_manifest(payload)
        return cls(_payload=_freeze_json(payload))

    @property
    def baseline_id(self) -> str:
        return str(self._payload["baseline_id"])

    @property
    def manifest_hash(self) -> str:
        return str(self._payload["manifest_hash"])

    @property
    def checkpoint_sha256(self) -> str:
        checkpoint = _object(self._payload["checkpoint"], "checkpoint")
        return str(checkpoint["sha256"])

    @property
    def checkpoint_size_bytes(self) -> int:
        checkpoint = _object(self._payload["checkpoint"], "checkpoint")
        return int(checkpoint["size_bytes"])

    def to_dict(self) -> dict[str, Any]:
        return _thaw_json(self._payload)


@dataclass(frozen=True)
class CheckpointFileVerification:
    """Hash-only verification performed before any checkpoint deserialization."""

    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class StairPhase0ReportV1:
    """Validated evidence that the pinned checkpoint runs on the frozen overlay."""

    _payload: Mapping[str, Any] = field(repr=False)

    @classmethod
    def load(cls, path: str | Path) -> "StairPhase0ReportV1":
        return cls.from_mapping(_load_json_object(Path(path)))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StairPhase0ReportV1":
        payload = _copy_json_object(value, "stair Phase 0 report")
        _assert_resolved_json(
            payload,
            "stair Phase 0 report",
            allowed_null_contexts=frozenset(
                {
                    "stair Phase 0 report.checkpoint_safety.loader_error_class",
                    "stair Phase 0 report.checkpoint_safety.loader_error_summary",
                }
            ),
        )
        _validate_phase0_report(payload)
        return cls(_payload=_freeze_json(payload))

    @property
    def report_hash(self) -> str:
        return str(self._payload["report_hash"])

    def to_dict(self) -> dict[str, Any]:
        return _thaw_json(self._payload)

    def assert_warm_start_admitted(self) -> None:
        payload = self.to_dict()
        admission = _object(payload["admission"], "Phase 0 admission")
        if payload["status"] != "pass" or admission["warm_start_candidate_admitted"] is not True:
            raise ValueError("Phase 0 warm-start candidate was not admitted")

    def assert_matches(
        self,
        protocol: StairProtocolV1,
        baseline: StairBaselineManifestV1,
        *,
        project_root: str | Path | None = None,
    ) -> None:
        _validate_report_bundle_compatibility(self.to_dict(), protocol, baseline)
        if project_root is not None:
            _validate_report_source_artifacts(self.to_dict(), Path(project_root))


@dataclass(frozen=True)
class StairPhase0Bundle:
    """Resolved Phase 0 resources, validated before any runtime activation."""

    project_root: Path
    descriptor_ref: Path
    descriptor: StairExtensionDescriptorV1
    protocol_ref: Path
    baseline_manifest_ref: Path
    phase0_report_ref: Path
    protocol: StairProtocolV1
    baseline: StairBaselineManifestV1

    @classmethod
    def load(
        cls,
        project_root: str | Path,
        *,
        descriptor_ref: str | Path = DEFAULT_STAIR_EXTENSION_DESCRIPTOR_REF,
    ) -> "StairPhase0Bundle":
        root = Path(project_root).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("Phase 0 project_root must be a directory")
        descriptor_path = _resolve_project_resource(root, descriptor_ref)
        descriptor = StairExtensionDescriptorV1.load(descriptor_path)
        descriptor_data = descriptor.to_dict()
        protocol_binding = _object(
            descriptor_data["task_protocol"],
            "stair extension descriptor task_protocol",
        )
        baseline_binding = _object(
            descriptor_data["baseline_manifest"],
            "stair extension descriptor baseline_manifest",
        )
        report_binding = _object(
            descriptor_data["phase0_report"],
            "stair extension descriptor phase0_report",
        )
        protocol_path = _resolve_project_resource(root, protocol_binding["ref"])
        manifest_path = _resolve_project_resource(root, baseline_binding["ref"])
        report_path = _resolve_project_output(root, report_binding["ref"])
        protocol = StairProtocolV1.load(protocol_path)
        baseline = StairBaselineManifestV1.load(manifest_path)
        _validate_descriptor_bundle_compatibility(descriptor, protocol, baseline)
        _validate_bundle_compatibility(protocol, baseline)
        return cls(
            project_root=root,
            descriptor_ref=descriptor_path.relative_to(root),
            descriptor=descriptor,
            protocol_ref=protocol_path.relative_to(root),
            baseline_manifest_ref=manifest_path.relative_to(root),
            phase0_report_ref=report_path.relative_to(root),
            protocol=protocol,
            baseline=baseline,
        )

    def load_report(
        self,
        report_ref: str | Path | None = None,
        *,
        require_admitted: bool = True,
    ) -> StairPhase0ReportV1:
        report_path = _resolve_project_resource(
            self.project_root,
            self.phase0_report_ref if report_ref is None else report_ref,
        )
        report = StairPhase0ReportV1.load(report_path)
        report.assert_matches(
            self.protocol,
            self.baseline,
            project_root=self.project_root,
        )
        if require_admitted:
            report.assert_warm_start_admitted()
        return report


def verify_checkpoint_file(
    path: str | Path,
    manifest: StairBaselineManifestV1,
) -> CheckpointFileVerification:
    """Verify the pinned checkpoint bytes before a caller invokes torch.load."""

    checkpoint_path = Path(path).resolve(strict=True)
    if not checkpoint_path.is_file():
        raise ValueError(f"checkpoint is not a file: {checkpoint_path}")
    size_bytes = checkpoint_path.stat().st_size
    if size_bytes != manifest.checkpoint_size_bytes:
        raise ValueError(
            "checkpoint size mismatch: "
            f"expected {manifest.checkpoint_size_bytes}, observed {size_bytes}"
        )
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    observed = digest.hexdigest()
    if observed != manifest.checkpoint_sha256:
        raise ValueError(
            "checkpoint sha256 mismatch: "
            f"expected {manifest.checkpoint_sha256}, observed {observed}"
        )
    return CheckpointFileVerification(
        path=checkpoint_path,
        sha256=observed,
        size_bytes=size_bytes,
    )


def canonical_json_sha256(value: Any) -> str:
    return canonical_json_hash(value)


def phase0_manager_term_groups(
    protocol: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return timeout and non-timeout manager terms declared by the task protocol."""

    probe = _object(protocol["phase0_probe"], "phase0_probe")
    termination_schema = _object(
        probe["termination_schema"],
        "phase0_probe termination_schema",
    )
    timeout_terms: list[str] = []
    terminated_terms: list[str] = []
    for index, raw_term in enumerate(
        _sequence(termination_schema["terms"], "phase0_probe termination terms")
    ):
        term = _object(raw_term, f"phase0_probe termination term {index}")
        name = _require_non_empty_text(
            term["manager_term"],
            f"phase0_probe termination term {index} manager_term",
        )
        target = (
            timeout_terms
            if _require_bool(term["time_out"], f"{name} time_out")
            else terminated_terms
        )
        target.append(name)
    return tuple(timeout_terms), tuple(terminated_terms)


def phase0_terminal_reason_names(protocol: Mapping[str, Any]) -> tuple[str, ...]:
    """Return declared Phase 0 terminal reasons in manager-term order."""

    probe = _object(protocol["phase0_probe"], "phase0_probe")
    reason_by_term = _object(
        probe["terminal_reason_by_manager_term"],
        "phase0_probe terminal_reason_by_manager_term",
    )
    termination_schema = _object(
        probe["termination_schema"],
        "phase0_probe termination_schema",
    )
    manager_terms = tuple(
        _require_non_empty_text(
            _object(raw_term, f"phase0_probe termination term {index}")[
                "manager_term"
            ],
            f"phase0_probe termination term {index} manager_term",
        )
        for index, raw_term in enumerate(
            _sequence(
                termination_schema["terms"],
                "phase0_probe termination terms",
            )
        )
    )
    return tuple(
        dict.fromkeys(
            _require_non_empty_text(
                reason_by_term[name],
                f"phase0_probe terminal reason for {name}",
            )
            for name in manager_terms
        )
    )


def resolve_phase0_terminal_reason(
    protocol: Mapping[str, Any],
    terminal_terms: Sequence[str],
) -> str:
    """Resolve the Phase 0 primary failure reason from native manager terms."""

    if not terminal_terms:
        raise ValueError("done transition has no active termination term")
    probe = _object(protocol["phase0_probe"], "phase0_probe")
    reason_by_term = _object(
        probe["terminal_reason_by_manager_term"],
        "phase0_probe terminal_reason_by_manager_term",
    )
    try:
        active_reasons = {reason_by_term[name] for name in terminal_terms}
    except KeyError as exc:
        raise ValueError(f"unmapped termination term: {exc.args[0]}") from exc
    failure_priority = _sequence(
        _object(protocol["episode_outcomes"], "episode_outcomes")[
            "failure_priority"
        ],
        "failure_priority",
    )
    for reason in failure_priority:
        if reason in active_reasons:
            return str(reason)
    raise ValueError("active termination terms have no primary failure reason")


def _validate_extension_descriptor(payload: Mapping[str, Any]) -> None:
    _expect_exact_keys(
        payload,
        {
            "schema_version",
            "task_id",
            "task_protocol",
            "baseline_manifest",
            "phase0_report",
            "descriptor_hash",
        },
        "stair extension descriptor",
    )
    _expect_equal(
        payload["schema_version"],
        STAIR_EXTENSION_DESCRIPTOR_SCHEMA_VERSION,
        "stair extension descriptor schema_version",
    )
    _expect_equal(payload["task_id"], STAIR_TASK_ID, "stair extension descriptor task_id")
    protocol = _object(payload["task_protocol"], "descriptor task_protocol")
    _expect_exact_keys(
        protocol,
        {"ref", "protocol_id", "protocol_hash"},
        "descriptor task_protocol",
    )
    _require_non_empty_text(protocol["ref"], "descriptor task_protocol ref")
    _require_non_empty_text(protocol["protocol_id"], "descriptor protocol_id")
    _require_sha256(protocol["protocol_hash"], "descriptor protocol_hash")
    baseline = _object(payload["baseline_manifest"], "descriptor baseline_manifest")
    _expect_exact_keys(
        baseline,
        {"ref", "baseline_id", "manifest_hash"},
        "descriptor baseline_manifest",
    )
    _require_non_empty_text(baseline["ref"], "descriptor baseline_manifest ref")
    _require_non_empty_text(baseline["baseline_id"], "descriptor baseline_id")
    _require_sha256(baseline["manifest_hash"], "descriptor manifest_hash")
    report = _object(payload["phase0_report"], "descriptor phase0_report")
    _expect_exact_keys(report, {"ref"}, "descriptor phase0_report")
    _require_non_empty_text(report["ref"], "descriptor phase0_report ref")
    declared_hash = _require_sha256(payload["descriptor_hash"], "descriptor_hash")
    if declared_hash != _hash_without_key(payload, "descriptor_hash"):
        raise ValueError("descriptor_hash mismatch")
    if declared_hash != FROZEN_STAIR_EXTENSION_DESCRIPTOR_V1_HASH:
        raise ValueError("descriptor is not the frozen stair extension descriptor V1")


def _validate_stair_protocol(payload: Mapping[str, Any]) -> None:
    _expect_exact_keys(
        payload,
        {
            "schema_version",
            "protocol_id",
            "task_id",
            "activation",
            "policy_compatibility",
            "terrain",
            "runtime_controls",
            "training",
            "evaluation",
            "episode_outcomes",
            "phase0_probe",
            "acceptance",
            "candidate_boundary",
            "task_protocol_hash",
        },
        "stair protocol",
    )
    _expect_equal(
        payload["schema_version"],
        STAIR_PROTOCOL_SCHEMA_VERSION,
        "stair protocol schema_version",
    )
    _expect_equal(payload["task_id"], STAIR_TASK_ID, "stair protocol task_id")
    _require_non_empty_text(payload["protocol_id"], "protocol_id")

    activation = _object(payload["activation"], "activation")
    _expect_exact_keys(
        activation,
        {
            "phase",
            "runtime_plugin_enabled",
            "training_command_enabled",
            "llm_stair_height_progression_enabled",
            "requires_baseline_genesis",
        },
        "activation",
    )
    _expect_equal(activation["phase"], "phase0_contract_frozen", "activation phase")
    for key in (
        "runtime_plugin_enabled",
        "training_command_enabled",
        "llm_stair_height_progression_enabled",
    ):
        _expect_equal(activation[key], False, f"activation {key}")
    _expect_equal(
        activation["requires_baseline_genesis"],
        True,
        "activation requires_baseline_genesis",
    )

    _validate_policy_compatibility(payload["policy_compatibility"])
    eval_bin_count = _validate_terrain(payload["terrain"])
    _validate_runtime_controls(payload["runtime_controls"])
    _validate_training(payload["training"])
    evaluation_protocol = _validate_evaluation(
        payload["evaluation"],
        eval_bin_count=eval_bin_count,
    )
    _validate_episode_outcomes(payload["episode_outcomes"])
    _validate_phase0_probe(payload["phase0_probe"], payload)
    acceptance_protocol = _validate_acceptance(payload["acceptance"])
    _validate_candidate_boundary(payload["candidate_boundary"])

    if acceptance_protocol.min_score_count != evaluation_protocol.public_accept_seed_count:
        raise ValueError("acceptance min_score_count must equal public_accept seed count")
    if acceptance_protocol.score_source != "public_accept.score_stats.mean":
        raise ValueError("acceptance score source must bind public_accept score_stats mean")

    declared_hash = _require_sha256(payload["task_protocol_hash"], "task_protocol_hash")
    expected_hash = _hash_without_key(payload, "task_protocol_hash")
    if declared_hash != expected_hash:
        raise ValueError("task_protocol_hash mismatch")


def _validate_policy_compatibility(value: Any) -> None:
    policy = _object(value, "policy_compatibility")
    _expect_exact_keys(
        policy,
        {
            "backend_id",
            "source_task_id",
            "robot",
            "robot_asset_id",
            "policy_architecture",
            "simulation_dt_s",
            "decimation",
            "control_dt_s",
            "robot_schema",
            "policy_observation",
            "action",
        },
        "policy_compatibility",
    )
    _expect_equal(
        policy["backend_id"],
        "isaac_lab_rsl_rl",
        "backend_id",
    )
    _expect_equal(
        policy["source_task_id"],
        "Isaac-Velocity-Rough-G1-v0",
        "source_task_id",
    )
    _expect_equal(
        policy["robot"],
        "unitree_g1",
        "robot",
    )
    _expect_equal(
        policy["robot_asset_id"],
        "isaaclab_assets.G1_MINIMAL_CFG",
        "robot_asset_id",
    )
    _expect_equal(
        policy["policy_architecture"],
        "rsl_rl_ppo_mlp",
        "policy_architecture",
    )
    simulation_dt = _positive_number(policy["simulation_dt_s"], "simulation_dt_s")
    decimation = _positive_int(policy["decimation"], "decimation")
    control_dt = _positive_number(policy["control_dt_s"], "control_dt_s")
    if not math.isclose(simulation_dt * decimation, control_dt, abs_tol=1e-12):
        raise ValueError("control_dt_s must equal simulation_dt_s * decimation")

    robot_schema = _object(policy["robot_schema"], "robot_schema")
    _expect_exact_keys(
        robot_schema,
        {"body_names", "joint_names", "schema_hash"},
        "robot_schema",
    )
    body_names = tuple(_sequence(robot_schema["body_names"], "robot body_names"))
    joint_names = tuple(_sequence(robot_schema["joint_names"], "robot joint_names"))
    for context, names in (("robot body_names", body_names), ("robot joint_names", joint_names)):
        if not names or len(names) != len(set(names)):
            raise ValueError(f"{context} must be non-empty and unique")
        for index, name in enumerate(names):
            _require_non_empty_text(name, f"{context} {index}")
    if len(joint_names) != 37:
        raise ValueError("robot joint_names must contain the frozen 37-joint order")
    required_bodies = {
        "torso_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    }
    if not required_bodies.issubset(body_names):
        raise ValueError("robot body_names are missing required G1 contact bodies")
    robot_schema_hash = _require_sha256(
        robot_schema["schema_hash"],
        "robot_schema schema_hash",
    )
    if robot_schema_hash != _hash_without_key(robot_schema, "schema_hash"):
        raise ValueError("robot_schema schema_hash mismatch")

    observation = _object(policy["policy_observation"], "policy_observation")
    _expect_exact_keys(
        observation,
        {"group", "concatenate_terms", "terms", "total_dim", "schema_hash"},
        "policy_observation",
    )
    _expect_equal(observation["group"], "policy", "policy_observation group")
    _expect_equal(
        observation["concatenate_terms"],
        True,
        "policy_observation concatenate_terms",
    )
    terms = _sequence(observation["terms"], "policy_observation terms")
    if not terms:
        raise ValueError("policy_observation terms must not be empty")
    term_payloads: list[dict[str, Any]] = []
    term_names: list[str] = []
    for index, raw in enumerate(terms):
        term = _object(raw, f"policy_observation term {index}")
        _expect_exact_keys(
            term,
            {"manager_term", "callable", "shape", "parameters"},
            f"policy_observation term {index}",
        )
        name = _require_non_empty_text(
            term["manager_term"],
            f"observation term {index} manager_term",
        )
        _require_non_empty_text(term["callable"], f"observation term {name} callable")
        shape = tuple(_sequence(term["shape"], f"observation term {name} shape"))
        if not shape:
            raise ValueError(f"observation term {name} shape must not be empty")
        for dimension in shape:
            _positive_int(dimension, f"observation term {name} shape dimension")
        parameters = _object(
            term["parameters"],
            f"observation term {name} parameters",
        )
        term_names.append(name)
        term_payloads.append(
            {
                "manager_term": name,
                "callable": str(term["callable"]),
                "shape": list(shape),
                "parameters": dict(parameters),
            }
        )
    if len(set(term_names)) != len(term_names):
        raise ValueError("policy_observation term names must be unique")
    expected_term_names = (
        "base_lin_vel",
        "base_ang_vel",
        "projected_gravity",
        "velocity_commands",
        "joint_pos",
        "joint_vel",
        "actions",
        "height_scan",
    )
    if tuple(term_names) != expected_term_names:
        raise ValueError("policy_observation runtime manager term order mismatch")
    if term_payloads[6]["callable"] != "isaaclab.envs.mdp.observations.last_action":
        raise ValueError("policy_observation actions term must call mdp.last_action")
    total_dim = _positive_int(observation["total_dim"], "policy_observation total_dim")
    if sum(math.prod(item["shape"]) for item in term_payloads) != total_dim:
        raise ValueError("policy_observation total_dim mismatch")
    schema_hash = _require_sha256(
        observation["schema_hash"],
        "policy_observation schema_hash",
    )
    if schema_hash != _hash_without_key(observation, "schema_hash"):
        raise ValueError("policy_observation schema_hash mismatch")

    action = _object(policy["action"], "action")
    _expect_exact_keys(
        action,
        {"terms", "total_dim", "schema_hash"},
        "action",
    )
    action_terms = _sequence(action["terms"], "action terms")
    if len(action_terms) != 1:
        raise ValueError("stair V1 action schema must contain exactly one manager term")
    action_term = _object(action_terms[0], "action term 0")
    _expect_exact_keys(
        action_term,
        {
            "manager_term",
            "implementation",
            "dim",
            "asset_name",
            "joint_selector",
            "preserve_order",
            "scale",
            "use_default_offset",
            "resolved_joint_names",
        },
        "action term 0",
    )
    _expect_equal(action_term["manager_term"], "joint_pos", "action manager_term")
    _expect_equal(
        action_term["implementation"],
        "isaaclab.envs.mdp.actions.joint_actions.JointPositionAction",
        "action implementation",
    )
    _expect_equal(action_term["asset_name"], "robot", "action asset_name")
    action_dim = _positive_int(action_term["dim"], "action term dim")
    selectors = tuple(_sequence(action_term["joint_selector"], "action joint_selector"))
    if selectors != (".*",):
        raise ValueError("action joint_selector must remain the frozen full-joint selector")
    _expect_equal(action_term["preserve_order"], False, "action preserve_order")
    if _positive_number(action_term["scale"], "action scale") != 0.5:
        raise ValueError("action scale must remain 0.5")
    _expect_equal(
        action_term["use_default_offset"],
        True,
        "action use_default_offset",
    )
    resolved_joint_names = tuple(
        _sequence(action_term["resolved_joint_names"], "action resolved_joint_names")
    )
    if resolved_joint_names != joint_names:
        raise ValueError("action resolved_joint_names must equal robot joint_names")
    total_action_dim = _positive_int(action["total_dim"], "action total_dim")
    if action_dim != total_action_dim or total_action_dim != len(joint_names):
        raise ValueError("action dimensions must equal the frozen robot joint order")
    action_hash = _require_sha256(action["schema_hash"], "action schema_hash")
    if action_hash != _hash_without_key(action, "schema_hash"):
        raise ValueError("action schema_hash mismatch")


def _validate_terrain(value: Any) -> int:
    terrain = _object(value, "terrain")
    _expect_exact_keys(
        terrain,
        {
            "generator_id",
            "family",
            "ascending_axis",
            "origin_semantics",
            "size_m",
            "floor_depth_m",
            "start_platform_length_m",
            "step_count",
            "tread_depth_m",
            "top_platform_length_m",
            "flat_fallback_allowed",
            "evaluation_grid_origins_m",
            "spawn",
            "goal",
            "training_distribution",
            "evaluation_bins",
        },
        "terrain",
    )
    _expect_equal(terrain["ascending_axis"], "+x", "terrain ascending_axis")
    _expect_equal(
        terrain["origin_semantics"],
        "bottom_start_platform_center",
        "terrain origin_semantics",
    )
    _expect_equal(
        terrain["flat_fallback_allowed"],
        False,
        "terrain flat_fallback_allowed",
    )
    bins = _sequence(terrain["evaluation_bins"], "evaluation_bins")
    if not bins:
        raise ValueError("evaluation_bins must not be empty")
    seen_ids: set[str] = set()
    observed_heights: list[float] = []
    for index, raw in enumerate(bins):
        item = _object(raw, f"evaluation bin {index}")
        _expect_exact_keys(
            item,
            {"bin_id", "height_m", "terrain_column"},
            f"evaluation bin {index}",
        )
        bin_id = _require_non_empty_text(item["bin_id"], f"evaluation bin {index} id")
        if bin_id in seen_ids:
            raise ValueError("evaluation bin ids must be unique")
        seen_ids.add(bin_id)
        observed_heights.append(_positive_number(item["height_m"], f"{bin_id} height"))
        if _non_negative_int(item["terrain_column"], f"{bin_id} terrain_column") != index:
            raise ValueError("evaluation terrain columns must be contiguous from zero")
    if any(
        current <= previous
        for previous, current in zip(
            observed_heights,
            observed_heights[1:],
            strict=False,
        )
    ):
        raise ValueError("evaluation bin heights must be strictly increasing")
    eval_bin_count = len(bins)

    size = _number_pair(terrain["size_m"], "terrain size_m", positive=True)
    floor_depth = _positive_number(terrain["floor_depth_m"], "floor_depth_m")
    start_length = _positive_number(
        terrain["start_platform_length_m"],
        "start_platform_length_m",
    )
    step_count = _positive_int(terrain["step_count"], "step_count")
    tread_depth = _positive_number(terrain["tread_depth_m"], "tread_depth_m")
    top_length = _positive_number(
        terrain["top_platform_length_m"],
        "top_platform_length_m",
    )
    _ = floor_depth
    occupied = start_length + step_count * tread_depth + top_length
    if not math.isclose(occupied, size[0], abs_tol=1e-9):
        raise ValueError("stair geometry lengths must exactly fill terrain x size")
    local_origin_x = start_length / 2.0
    local_origin_y = size[1] / 2.0
    grid_origins = _sequence(
        terrain["evaluation_grid_origins_m"],
        "terrain evaluation_grid_origins_m",
    )
    if len(grid_origins) != eval_bin_count:
        raise ValueError("evaluation_grid_origins_m must match evaluation bin count")
    expected_grid_origins = [
        [
            local_origin_x - size[0] / 2.0,
            local_origin_y + (column - eval_bin_count / 2.0) * size[1],
            0.0,
        ]
        for column in range(eval_bin_count)
    ]
    observed_grid_origins: list[list[float]] = []
    for index, origin in enumerate(grid_origins):
        values = _sequence(origin, f"terrain evaluation grid origin {index}")
        if len(values) != 3:
            raise ValueError("each evaluation grid origin must be xyz")
        observed_grid_origins.append(
            [_finite_number(value, f"evaluation grid origin {index}") for value in values]
        )
    if observed_grid_origins != expected_grid_origins:
        raise ValueError("evaluation_grid_origins_m mismatch frozen terrain placement")

    spawn = _object(terrain["spawn"], "terrain spawn")
    _expect_exact_keys(
        spawn,
        {
            "x_range_m_from_origin",
            "y_range_m_from_origin",
            "yaw_range_rad",
        },
        "terrain spawn",
    )
    spawn_x = _ordered_number_pair(spawn["x_range_m_from_origin"], "spawn x range")
    spawn_y = _ordered_number_pair(spawn["y_range_m_from_origin"], "spawn y range")
    if spawn_x[0] < -local_origin_x or spawn_x[1] > start_length - local_origin_x:
        raise ValueError("stair spawn x range must stay on the bottom start platform")
    if spawn_y[0] < -local_origin_y or spawn_y[1] > size[1] - local_origin_y:
        raise ValueError("stair spawn y range must stay inside the terrain width")
    yaw_range = _ordered_number_pair(spawn["yaw_range_rad"], "spawn yaw range")
    if yaw_range != (0.0, 0.0):
        raise ValueError("stair spawn yaw must be fixed to zero")

    goal = _object(terrain["goal"], "terrain goal")
    _expect_exact_keys(
        goal,
        {
            "progress_start_m_from_origin",
            "progress_goal_m_from_origin",
            "top_height_multiplier",
        },
        "terrain goal",
    )
    start = _non_negative_number(
        goal["progress_start_m_from_origin"],
        "goal progress_start_m_from_origin",
    )
    finish = _positive_number(
        goal["progress_goal_m_from_origin"],
        "goal progress_goal_m_from_origin",
    )
    if finish <= start or finish >= size[0]:
        raise ValueError("goal progress range must be positive and inside terrain")
    top_start_from_origin = start_length + step_count * tread_depth - local_origin_x
    top_end_from_origin = size[0] - local_origin_x
    if not top_start_from_origin < finish < top_end_from_origin:
        raise ValueError("goal must lie strictly inside the top platform")
    if _positive_int(goal["top_height_multiplier"], "top_height_multiplier") != step_count:
        raise ValueError("top_height_multiplier must equal step_count")

    training = _object(terrain["training_distribution"], "training_distribution")
    _expect_exact_keys(
        training,
        {
            "step_height_range_m",
            "terrain_rows",
            "terrain_cols",
            "generation_mode",
            "runtime_terrain_curriculum_enabled",
        },
        "training_distribution",
    )
    height_range = _ordered_number_pair(
        training["step_height_range_m"],
        "training step_height_range_m",
    )
    if height_range[0] <= 0:
        raise ValueError("training stair heights must be positive")
    _positive_int(training["terrain_rows"], "training terrain_rows")
    _positive_int(training["terrain_cols"], "training terrain_cols")
    _expect_equal(
        training["generation_mode"],
        "seeded_uniform_height",
        "training generation_mode",
    )
    _expect_equal(
        training["runtime_terrain_curriculum_enabled"],
        False,
        "runtime_terrain_curriculum_enabled",
    )

    return eval_bin_count


def _validate_runtime_controls(value: Any) -> None:
    controls = _object(value, "runtime_controls")
    _expect_exact_keys(
        controls,
        {"reset_velocity_ranges", "command", "events"},
        "runtime_controls",
    )
    velocity_ranges = _object(
        controls["reset_velocity_ranges"],
        "runtime_controls reset_velocity_ranges",
    )
    _expect_exact_keys(
        velocity_ranges,
        {"x", "y", "z", "roll", "pitch", "yaw"},
        "runtime_controls reset_velocity_ranges",
    )
    for axis, raw_range in velocity_ranges.items():
        if _ordered_number_pair(
            raw_range,
            f"runtime_controls reset velocity {axis}",
        ) != (0.0, 0.0):
            raise ValueError("runtime_controls reset velocities must be zero")

    command = _object(controls["command"], "runtime_controls command")
    _expect_exact_keys(
        command,
        {
            "resampling",
            "standing_env_fraction",
            "heading_env_fraction",
            "heading_command_enabled",
        },
        "runtime_controls command",
    )
    _expect_equal(
        command["resampling"],
        "episode_length",
        "runtime_controls command resampling",
    )
    for key in ("standing_env_fraction", "heading_env_fraction"):
        if _finite_number(command[key], f"runtime_controls command {key}") != 0.0:
            raise ValueError(f"runtime_controls command {key} must be zero")
    _expect_equal(
        command["heading_command_enabled"],
        False,
        "runtime_controls heading_command_enabled",
    )

    events = _object(controls["events"], "runtime_controls events")
    _expect_exact_keys(
        events,
        {"base_external_force_torque_enabled", "push_robot_enabled"},
        "runtime_controls events",
    )
    for key in events:
        _expect_equal(events[key], False, f"runtime_controls events {key}")


def _validate_training(value: Any) -> None:
    training = _object(value, "training")
    _expect_exact_keys(
        training,
        {
            "policy_initialization",
            "num_envs",
            "steps_per_env",
            "iterations",
            "budget_steps",
            "episode_length_s",
            "runner_timeout_s",
            "checkpoint_selector",
            "forward_command_mps",
            "lateral_command_mps",
            "yaw_command_radps",
            "observation_corruption_enabled",
        },
        "training",
    )
    _expect_equal(
        training["policy_initialization"],
        "fixed_session_genesis_baseline",
        "training policy_initialization",
    )
    num_envs = _positive_int(training["num_envs"], "training num_envs")
    steps_per_env = _positive_int(training["steps_per_env"], "training steps_per_env")
    iterations = _positive_int(training["iterations"], "training iterations")
    budget_steps = _positive_int(training["budget_steps"], "training budget_steps")
    if num_envs * steps_per_env * iterations != budget_steps:
        raise ValueError("training budget_steps arithmetic mismatch")
    _positive_number(training["episode_length_s"], "training episode_length_s")
    _positive_int(training["runner_timeout_s"], "training runner_timeout_s")
    _expect_equal(
        training["checkpoint_selector"],
        "final_iteration",
        "training checkpoint_selector",
    )
    _positive_number(training["forward_command_mps"], "training forward command")
    if _finite_number(training["lateral_command_mps"], "training lateral command") != 0:
        raise ValueError("training lateral command must be zero")
    if _finite_number(training["yaw_command_radps"], "training yaw command") != 0:
        raise ValueError("training yaw command must be zero")
    _require_bool(
        training["observation_corruption_enabled"],
        "training observation_corruption_enabled",
    )


def _validate_evaluation(
    value: Any,
    *,
    eval_bin_count: int,
) -> EvaluationProtocol:
    evaluation = _object(value, "evaluation")
    _expect_exact_keys(
        evaluation,
        {
            "protocol",
            "splits",
            "seed_policy_version",
            "env_to_bin_mapping",
            "episode_length_s",
            "forward_command_mps",
            "lateral_command_mps",
            "yaw_command_radps",
            "observation_corruption_enabled",
            "score_statistics",
        },
        "evaluation",
    )
    protocol = EvaluationProtocol.from_mapping(evaluation["protocol"])
    splits = _object(evaluation["splits"], "evaluation splits")
    _expect_exact_keys(
        splits,
        {"public_dev", "public_accept", "sealed_validation"},
        "evaluation splits",
    )
    expected_counts = {
        "public_dev": protocol.public_seed_count,
        "public_accept": protocol.public_accept_seed_count,
        "sealed_validation": protocol.sealed_validation_seed_count,
    }
    episodes_per_bin: set[int] = set()
    for name, expected_count in expected_counts.items():
        split = _object(splits[name], f"evaluation split {name}")
        _expect_exact_keys(
            split,
            {
                "seed_count",
                "episodes_per_seed_per_bin",
                "complete_seven_bins_required",
            },
            f"evaluation split {name}",
        )
        if _positive_int(split["seed_count"], f"{name} seed_count") != expected_count:
            raise ValueError(f"{name} seed_count does not match evaluation protocol")
        episodes_per_bin.add(
            _positive_int(
                split["episodes_per_seed_per_bin"],
                f"{name} episodes_per_seed_per_bin",
            )
        )
        _expect_equal(
            split["complete_seven_bins_required"],
            True,
            f"{name} complete_seven_bins_required",
        )
    if len(episodes_per_bin) != 1:
        raise ValueError("all evaluation splits must use the same episodes per bin")

    _require_non_empty_text(
        evaluation["seed_policy_version"],
        "evaluation seed_policy_version",
    )
    mapping = _object(evaluation["env_to_bin_mapping"], "env_to_bin_mapping")
    _expect_exact_keys(
        mapping,
        {"rule", "envs_per_bin", "first_env_index", "last_env_index"},
        "env_to_bin_mapping",
    )
    _expect_equal(
        mapping["rule"],
        "contiguous_equal_blocks_by_terrain_column",
        "env_to_bin_mapping rule",
    )
    envs_per_bin = _positive_int(mapping["envs_per_bin"], "envs_per_bin")
    if envs_per_bin != next(iter(episodes_per_bin)):
        raise ValueError("envs_per_bin must equal episodes_per_seed_per_bin")
    if protocol.eval_num_envs != envs_per_bin * eval_bin_count:
        raise ValueError("eval_num_envs must equal bin count times envs_per_bin")
    if _non_negative_int(mapping["first_env_index"], "first_env_index") != 0:
        raise ValueError("first_env_index must be zero")
    if (
        _non_negative_int(mapping["last_env_index"], "last_env_index")
        != protocol.eval_num_envs - 1
    ):
        raise ValueError("last_env_index must match eval_num_envs")

    _positive_number(evaluation["episode_length_s"], "evaluation episode_length_s")
    _positive_number(evaluation["forward_command_mps"], "evaluation forward command")
    if _finite_number(evaluation["lateral_command_mps"], "evaluation lateral command") != 0:
        raise ValueError("evaluation lateral command must be zero")
    if _finite_number(evaluation["yaw_command_radps"], "evaluation yaw command") != 0:
        raise ValueError("evaluation yaw command must be zero")
    _expect_equal(
        evaluation["observation_corruption_enabled"],
        False,
        "evaluation observation_corruption_enabled",
    )
    stats = _object(evaluation["score_statistics"], "score_statistics")
    _expect_exact_keys(
        stats,
        {
            "per_bin",
            "per_seed",
            "across_seeds",
            "score_stats_values_unit",
            "public_score_source",
        },
        "score_statistics",
    )
    _expect_equal(stats["per_bin"], "episode_success_rate", "score per_bin")
    _expect_equal(
        stats["per_seed"],
        "equal_weight_macro_mean_over_seven_bins",
        "score per_seed",
    )
    _expect_equal(
        stats["across_seeds"],
        "mean_sample_std_standard_error",
        "score across_seeds",
    )
    _expect_equal(
        stats["score_stats_values_unit"],
        "one_macro_score_per_seed",
        "score_stats_values_unit",
    )
    _expect_equal(
        stats["public_score_source"],
        "public_accept.score_stats.mean",
        "public_score_source",
    )
    return protocol


def _validate_episode_outcomes(value: Any) -> None:
    outcomes = _object(value, "episode_outcomes")
    _expect_exact_keys(
        outcomes,
        {
            "primary_reasons",
            "failure_priority",
            "success",
            "signals",
            "diagnostic_flags",
            "reserved_not_admitted",
            "counting_invariant",
            "classification_warning_blocks_pass",
        },
        "episode_outcomes",
    )
    primary_reasons = _unique_text_tuple(
        outcomes["primary_reasons"],
        "primary_reasons",
    )
    if not primary_reasons or primary_reasons[0] != "success":
        raise ValueError("primary_reasons must start with success")
    failure_priority = _unique_text_tuple(
        outcomes["failure_priority"],
        "failure_priority",
    )
    if failure_priority != primary_reasons[1:]:
        raise ValueError(
            "failure_priority must contain every non-success primary reason in order"
        )
    for required_reason in ("fall", "timeout"):
        if required_reason not in failure_priority:
            raise ValueError(
                f"failure_priority must include probe-required reason: {required_reason}"
            )

    success = _object(outcomes["success"], "success outcome")
    _expect_exact_keys(
        success,
        {
            "minimum_progress_fraction",
            "top_hold_control_steps",
            "requires_no_base_contact",
            "requires_not_fallen",
        },
        "success outcome",
    )
    if _positive_number(success["minimum_progress_fraction"], "success progress") != 1.0:
        raise ValueError("success minimum_progress_fraction must be 1.0")
    _positive_int(success["top_hold_control_steps"], "success top_hold_control_steps")
    _expect_equal(
        success["requires_no_base_contact"],
        True,
        "success requires_no_base_contact",
    )
    _expect_equal(success["requires_not_fallen"], True, "success requires_not_fallen")

    signals = _object(outcomes["signals"], "outcome signals")
    _expect_exact_keys(signals, set(failure_priority), "outcome signals")
    for name in failure_priority:
        signal = _object(signals[name], f"outcome signal {name}")
        if not signal:
            raise ValueError(f"outcome signal {name} must not be empty")
        if "source" not in signal:
            raise ValueError(f"outcome signal {name} must define source")
        _require_non_empty_text(signal["source"], f"{name} source")
        for key, raw in signal.items():
            if key == "source":
                continue
            if key == "consecutive_control_steps":
                _positive_int(raw, f"{name} {key}")
            elif key == "threshold" or key.endswith("_threshold"):
                _positive_number(raw, f"{name} {key}")
            else:
                _require_non_empty_text(raw, f"{name} {key}")
    fall_signal = _object(signals["fall"], "outcome signal fall")
    for required_key in ("threshold", "consecutive_control_steps"):
        if required_key not in fall_signal:
            raise ValueError(f"fall signal must define {required_key}")
    timeout_signal = _object(signals["timeout"], "outcome signal timeout")
    if "threshold" not in timeout_signal:
        raise ValueError("timeout signal must define threshold")

    flags = _sequence(outcomes["diagnostic_flags"], "diagnostic_flags")
    if len(flags) != len(set(flags)):
        raise ValueError("diagnostic_flags must be unique")
    for index, flag in enumerate(flags):
        _require_non_empty_text(flag, f"diagnostic flag {index}")
    reserved = _object(outcomes["reserved_not_admitted"], "reserved_not_admitted")
    for name, explanation in reserved.items():
        _require_non_empty_text(name, "reserved_not_admitted reason")
        _require_non_empty_text(
            explanation,
            f"reserved_not_admitted {name}",
        )
    _expect_equal(
        outcomes["counting_invariant"],
        "success_count_plus_primary_failure_counts_equals_total_episode_count",
        "episode counting_invariant",
    )
    _expect_equal(
        outcomes["classification_warning_blocks_pass"],
        True,
        "classification_warning_blocks_pass",
    )


def _validate_phase0_probe(value: Any, protocol: Mapping[str, Any]) -> None:
    probe = _object(value, "phase0_probe")
    _expect_exact_keys(
        probe,
        {
            "seed",
            "control_steps",
            "first_episode_only",
            "first_step_progress_m_from_origin",
            "in_distribution_bin_ids",
            "minimum_reached_first_step_fraction_per_bin",
            "requires_all_first_episodes_complete",
            "success_required_for_warm_start_admission",
            "termination_schema",
            "terminal_reason_by_manager_term",
            "diagnostic_milestones",
            "checkpoint_safety_report_required",
        },
        "phase0_probe",
    )
    _positive_int(probe["seed"], "phase0_probe seed")
    policy = _object(protocol["policy_compatibility"], "policy_compatibility")
    training = _object(protocol["training"], "training")
    evaluation = _object(protocol["evaluation"], "evaluation")
    control_dt = _positive_number(policy["control_dt_s"], "control_dt_s")
    training_episode_s = _positive_number(
        training["episode_length_s"],
        "training episode_length_s",
    )
    evaluation_episode_s = _positive_number(
        evaluation["episode_length_s"],
        "evaluation episode_length_s",
    )
    if not math.isclose(training_episode_s, evaluation_episode_s, abs_tol=1e-12):
        raise ValueError("training and evaluation episode lengths must match in stair V1")
    expected_control_steps = int(round(evaluation_episode_s / control_dt))
    if not math.isclose(
        expected_control_steps * control_dt,
        evaluation_episode_s,
        abs_tol=1e-12,
    ):
        raise ValueError("episode length must be an integer number of control steps")
    if (
        _positive_int(probe["control_steps"], "phase0_probe control_steps")
        != expected_control_steps
    ):
        raise ValueError("phase0_probe control_steps must cover one complete episode")
    for key in (
        "first_episode_only",
        "requires_all_first_episodes_complete",
        "checkpoint_safety_report_required",
    ):
        _expect_equal(probe[key], True, f"phase0_probe {key}")
    _expect_equal(
        probe["success_required_for_warm_start_admission"],
        False,
        "phase0_probe success_required_for_warm_start_admission",
    )

    terrain = _object(protocol["terrain"], "terrain")
    expected_first_step_progress = float(terrain["start_platform_length_m"]) / 2.0
    observed_first_step_progress = _positive_number(
        probe["first_step_progress_m_from_origin"],
        "phase0_probe first_step_progress_m_from_origin",
    )
    if not math.isclose(
        observed_first_step_progress,
        expected_first_step_progress,
        abs_tol=1e-12,
    ):
        raise ValueError("phase0_probe first-step progress must bind the first riser")
    bins = _sequence(terrain["evaluation_bins"], "evaluation_bins")
    train_upper = float(terrain["training_distribution"]["step_height_range_m"][1])
    expected_in_distribution_bins = tuple(
        item["bin_id"] for item in bins if float(item["height_m"]) <= train_upper
    )
    observed_in_distribution_bins = tuple(
        _sequence(
            probe["in_distribution_bin_ids"],
            "phase0_probe in_distribution_bin_ids",
        )
    )
    if observed_in_distribution_bins != expected_in_distribution_bins:
        raise ValueError("phase0_probe in-distribution bins mismatch training heights")
    minimum_fraction = _positive_number(
        probe["minimum_reached_first_step_fraction_per_bin"],
        "phase0_probe minimum reached-first-step fraction",
    )
    if minimum_fraction > 1.0:
        raise ValueError(
            "phase0_probe minimum reached-first-step fraction must not exceed one"
        )
    termination_schema = _object(
        probe["termination_schema"],
        "phase0_probe termination_schema",
    )
    _expect_exact_keys(
        termination_schema,
        {"terms", "schema_hash"},
        "phase0_probe termination_schema",
    )
    raw_termination_terms = list(
        _sequence(termination_schema["terms"], "phase0_probe termination terms")
    )
    if not raw_termination_terms:
        raise ValueError("phase0_probe termination schema must not be empty")
    manager_terms: list[str] = []
    timeout_terms: list[str] = []
    terminated_terms: list[str] = []
    for index, raw_term in enumerate(raw_termination_terms):
        term = _object(raw_term, f"phase0_probe termination term {index}")
        _expect_exact_keys(
            term,
            {"manager_term", "callable", "time_out", "parameters"},
            f"phase0_probe termination term {index}",
        )
        manager_term = _require_non_empty_text(
            term["manager_term"],
            f"phase0_probe termination term {index} manager_term",
        )
        _require_non_empty_text(
            term["callable"],
            f"phase0_probe termination term {manager_term} callable",
        )
        _object(
            term["parameters"],
            f"phase0_probe termination term {manager_term} parameters",
        )
        target = (
            timeout_terms
            if _require_bool(term["time_out"], f"{manager_term} time_out")
            else terminated_terms
        )
        target.append(manager_term)
        manager_terms.append(manager_term)
    if len(manager_terms) != len(set(manager_terms)):
        raise ValueError("phase0_probe termination manager terms must be unique")
    if len(timeout_terms) != 1 or len(terminated_terms) != 1:
        raise ValueError(
            "phase0_probe termination schema requires exactly one timeout "
            "and one non-timeout term"
        )
    declared_termination_hash = _require_sha256(
        termination_schema["schema_hash"],
        "phase0_probe termination schema_hash",
    )
    if declared_termination_hash != _hash_without_key(
        termination_schema,
        "schema_hash",
    ):
        raise ValueError("phase0_probe termination schema_hash mismatch")
    reason_by_term = _object(
        probe["terminal_reason_by_manager_term"],
        "phase0_probe terminal_reason_by_manager_term",
    )
    _expect_exact_keys(
        reason_by_term,
        set(manager_terms),
        "phase0_probe terminal reason mapping",
    )
    failure_priority = _unique_text_tuple(
        _object(protocol["episode_outcomes"], "episode_outcomes")[
            "failure_priority"
        ],
        "failure_priority",
    )
    mapped_reasons = tuple(
        _require_non_empty_text(
            reason_by_term[name],
            f"phase0_probe terminal reason for {name}",
        )
        for name in manager_terms
    )
    if len(mapped_reasons) != len(set(mapped_reasons)):
        raise ValueError("phase0_probe terminal reason mapping must be one-to-one")
    if any(reason not in failure_priority for reason in mapped_reasons):
        raise ValueError(
            "phase0_probe terminal reason mapping must target primary failure reasons"
        )
    diagnostic_milestones = _unique_text_tuple(
        probe["diagnostic_milestones"],
        "diagnostic_milestones",
    )
    if not diagnostic_milestones:
        raise ValueError("diagnostic_milestones must not be empty")

    outcomes = _object(protocol["episode_outcomes"], "episode_outcomes")
    timeout = _object(
        _object(outcomes["signals"], "outcome signals")["timeout"],
        "timeout outcome signal",
    )
    if not math.isclose(
        float(timeout["threshold"]),
        evaluation_episode_s,
        abs_tol=1e-12,
    ):
        raise ValueError("timeout threshold must equal the frozen episode length")
    top_hold_steps = _positive_int(
        _object(outcomes["success"], "success outcome")["top_hold_control_steps"],
        "success top_hold_control_steps",
    )
    if top_hold_steps >= expected_control_steps:
        raise ValueError("success top hold must fit inside one episode")


def _validate_acceptance(value: Any) -> AcceptanceProtocol:
    acceptance = _object(value, "acceptance")
    _expect_exact_keys(
        acceptance,
        {"protocol", "requires_baseline_genesis", "required_pass_gates"},
        "acceptance",
    )
    protocol = AcceptanceProtocol.from_mapping(acceptance["protocol"])
    _expect_equal(
        acceptance["requires_baseline_genesis"],
        True,
        "acceptance requires_baseline_genesis",
    )
    gates = _sequence(acceptance["required_pass_gates"], "required_pass_gates")
    gate_names = _unique_text_tuple(gates, "required_pass_gates")
    if not gate_names:
        raise ValueError("required_pass_gates must not be empty")
    return protocol


def _validate_candidate_boundary(value: Any) -> None:
    boundary = _object(value, "candidate_boundary")
    _expect_exact_keys(
        boundary,
        {
            "common_reward_input_type",
            "common_reward_input_fields",
            "planned_task_reward_input_fields",
            "prompt_effective_task_specific_fields",
            "parameter_only_components",
            "diagnostic_components",
            "engineering_components",
            "code_level_components",
            "forbidden_components",
            "forbidden_prompt_fields",
        },
        "candidate_boundary",
    )
    _expect_equal(
        boundary["common_reward_input_type"],
        "true_rl_agent.candidates.components.RewardInputs",
        "common_reward_input_type",
    )
    common_fields = tuple(
        _sequence(
            boundary["common_reward_input_fields"],
            "common_reward_input_fields",
        )
    )
    if common_fields != RewardInputs.field_names():
        raise ValueError("common_reward_input_fields must match RewardInputs exactly")
    fields = _sequence(
        boundary["planned_task_reward_input_fields"],
        "planned_task_reward_input_fields",
    )
    names: list[str] = []
    for index, raw in enumerate(fields):
        item = _object(raw, f"reward input field {index}")
        _expect_exact_keys(
            item,
            {
                "name",
                "shape",
                "unit",
                "provenance",
                "source",
                "device",
                "reset_semantics",
                "activation_gate",
            },
            f"reward input field {index}",
        )
        names.append(_require_non_empty_text(item["name"], f"reward input {index} name"))
        shape = _sequence(item["shape"], f"reward input {index} shape")
        if not shape:
            raise ValueError("reward input shape must not be empty")
        for dimension in shape:
            if isinstance(dimension, bool) or not isinstance(dimension, int | str):
                raise ValueError("reward input shape dimensions must be integers or names")
        _require_non_empty_text(item["unit"], f"reward input {index} unit")
        if item["provenance"] not in {"observed", "runner_derived"}:
            raise ValueError("reward input provenance must be observed or runner_derived")
        for key in ("source", "device", "reset_semantics", "activation_gate"):
            _require_non_empty_text(item[key], f"reward input {index} {key}")
    if len(names) != len(set(names)):
        raise ValueError("planned reward input field names must be unique")
    if set(names) & set(common_fields):
        raise ValueError("planned task reward inputs must not duplicate common fields")
    prompt_effective = _sequence(
        boundary["prompt_effective_task_specific_fields"],
        "prompt_effective_task_specific_fields",
    )
    if prompt_effective:
        raise ValueError(
            "Phase 0 must not declare task-specific prompt-effective reward inputs"
        )

    for key in (
        "parameter_only_components",
        "diagnostic_components",
        "engineering_components",
        "code_level_components",
        "forbidden_components",
        "forbidden_prompt_fields",
    ):
        values = _sequence(boundary[key], key)
        if not values or len(values) != len(set(values)):
            raise ValueError(f"{key} must be a non-empty unique sequence")
        for index, item in enumerate(values):
            _require_non_empty_text(item, f"{key} item {index}")
    if "reward_callable" not in boundary["code_level_components"]:
        raise ValueError("reward_callable must remain code-level")
    classified_groups = (
        set(boundary["parameter_only_components"]),
        set(boundary["diagnostic_components"]),
        set(boundary["engineering_components"]),
        set(boundary["code_level_components"]),
    )
    for index, group in enumerate(classified_groups):
        if any(group & other for other in classified_groups[index + 1 :]):
            raise ValueError("candidate boundary component groups must be disjoint")


def _validate_baseline_manifest(payload: Mapping[str, Any]) -> None:
    _expect_exact_keys(
        payload,
        {
            "schema_version",
            "baseline_id",
            "target_task_id",
            "role",
            "accepted_current_best",
            "source",
            "checkpoint",
            "compatibility",
            "version_lock",
            "admission",
            "manifest_hash",
        },
        "stair baseline manifest",
    )
    _expect_equal(
        payload["schema_version"],
        STAIR_BASELINE_MANIFEST_SCHEMA_VERSION,
        "baseline schema_version",
    )
    _require_non_empty_text(payload["baseline_id"], "baseline_id")
    _expect_equal(payload["target_task_id"], STAIR_TASK_ID, "baseline target_task_id")
    _expect_equal(payload["role"], "warm_start", "baseline role")
    _expect_equal(
        payload["accepted_current_best"],
        False,
        "baseline accepted_current_best",
    )

    source = _object(payload["source"], "baseline source")
    _expect_exact_keys(
        source,
        {"provider", "workflow", "source_task_id", "artifact_ref"},
        "baseline source",
    )
    for key in source:
        _require_non_empty_text(source[key], f"baseline source {key}")
    artifact_ref = str(source["artifact_ref"])
    if "://" not in artifact_ref or artifact_ref.startswith(("file://", "/", "\\")):
        raise ValueError("baseline artifact_ref must be a logical non-file URI")
    _expect_equal(
        source["source_task_id"],
        "Isaac-Velocity-Rough-G1-v0",
        "baseline source_task_id",
    )

    checkpoint = _object(payload["checkpoint"], "baseline checkpoint")
    _expect_exact_keys(
        checkpoint,
        {
            "sha256",
            "size_bytes",
            "iteration",
            "container",
            "required_top_level_keys",
        },
        "baseline checkpoint",
    )
    _require_sha256(checkpoint["sha256"], "baseline checkpoint sha256")
    _positive_int(checkpoint["size_bytes"], "baseline checkpoint size_bytes")
    _non_negative_int(checkpoint["iteration"], "baseline checkpoint iteration")
    _require_non_empty_text(checkpoint["container"], "baseline checkpoint container")
    required_keys = _sequence(
        checkpoint["required_top_level_keys"],
        "baseline required_top_level_keys",
    )
    if "model_state_dict" not in required_keys:
        raise ValueError("baseline checkpoint must require model_state_dict")

    compatibility = _object(payload["compatibility"], "baseline compatibility")
    _expect_exact_keys(
        compatibility,
        {
            "backend_id",
            "robot",
            "robot_asset_id",
            "robot_schema_hash",
            "policy_architecture",
            "policy_observation_dim",
            "policy_observation_schema_hash",
            "action_dim",
            "action_schema_hash",
            "actor_hidden_dims",
            "critic_hidden_dims",
            "activation",
            "actor_output_dim",
        },
        "baseline compatibility",
    )
    _expect_equal(
        compatibility["backend_id"],
        "isaac_lab_rsl_rl",
        "baseline backend_id",
    )
    _expect_equal(
        compatibility["robot"],
        "unitree_g1",
        "baseline robot",
    )
    _require_non_empty_text(compatibility["robot_asset_id"], "baseline robot_asset_id")
    _require_sha256(
        compatibility["robot_schema_hash"],
        "baseline robot_schema_hash",
    )
    _expect_equal(
        compatibility["policy_architecture"],
        "rsl_rl_ppo_mlp",
        "baseline policy_architecture",
    )
    _positive_int(
        compatibility["policy_observation_dim"],
        "baseline policy_observation_dim",
    )
    _require_sha256(
        compatibility["policy_observation_schema_hash"],
        "baseline policy_observation_schema_hash",
    )
    action_dim = _positive_int(compatibility["action_dim"], "baseline action_dim")
    _require_sha256(
        compatibility["action_schema_hash"],
        "baseline action_schema_hash",
    )
    for key in ("actor_hidden_dims", "critic_hidden_dims"):
        dims = _sequence(compatibility[key], f"baseline {key}")
        if not dims:
            raise ValueError(f"baseline {key} must not be empty")
        for index, dim in enumerate(dims):
            _positive_int(dim, f"baseline {key} {index}")
    _require_non_empty_text(compatibility["activation"], "baseline activation")
    if (
        _positive_int(compatibility["actor_output_dim"], "baseline actor_output_dim")
        != action_dim
    ):
        raise ValueError("baseline actor_output_dim must equal action_dim")

    version_lock = _object(payload["version_lock"], "baseline version_lock")
    _expect_exact_keys(
        version_lock,
        {
            "isaac_lab",
            "isaaclab_extension",
            "isaaclab_tasks_extension",
            "isaaclab_rl_extension",
            "rsl_rl",
            "torch",
        },
        "baseline version_lock",
    )
    for key, raw in version_lock.items():
        _require_non_empty_text(raw, f"baseline version_lock {key}")

    admission = _object(payload["admission"], "baseline admission")
    _expect_exact_keys(
        admission,
        {
            "manifest_role",
            "phase0_warm_start_admission_required",
            "baseline_eval_only_required",
            "session_genesis_required",
        },
        "baseline admission",
    )
    _expect_equal(
        admission["manifest_role"],
        "warm_start_candidate",
        "baseline admission manifest_role",
    )
    for key in (
        "phase0_warm_start_admission_required",
        "baseline_eval_only_required",
        "session_genesis_required",
    ):
        _expect_equal(admission[key], True, f"baseline admission {key}")

    declared_hash = _require_sha256(payload["manifest_hash"], "manifest_hash")
    expected_hash = _hash_without_key(payload, "manifest_hash")
    if declared_hash != expected_hash:
        raise ValueError("manifest_hash mismatch")


def _validate_descriptor_bundle_compatibility(
    descriptor: StairExtensionDescriptorV1,
    protocol: StairProtocolV1,
    baseline: StairBaselineManifestV1,
) -> None:
    payload = descriptor.to_dict()
    protocol_binding = _object(payload["task_protocol"], "descriptor task_protocol")
    baseline_binding = _object(
        payload["baseline_manifest"],
        "descriptor baseline_manifest",
    )
    expected_pairs = (
        ("task_id", payload["task_id"], protocol.task_id),
        ("protocol_id", protocol_binding["protocol_id"], protocol.protocol_id),
        ("protocol_hash", protocol_binding["protocol_hash"], protocol.protocol_hash),
        ("baseline_id", baseline_binding["baseline_id"], baseline.baseline_id),
        ("manifest_hash", baseline_binding["manifest_hash"], baseline.manifest_hash),
    )
    for name, expected, observed in expected_pairs:
        if expected != observed:
            raise ValueError(f"stair extension descriptor binding mismatch: {name}")


def _validate_bundle_compatibility(
    protocol: StairProtocolV1,
    baseline: StairBaselineManifestV1,
) -> None:
    protocol_data = protocol.to_dict()
    baseline_data = baseline.to_dict()
    if baseline_data["target_task_id"] != protocol.task_id:
        raise ValueError("baseline target_task_id does not match stair protocol")
    policy = _object(protocol_data["policy_compatibility"], "policy_compatibility")
    robot_schema = _object(policy["robot_schema"], "robot_schema")
    observation = _object(policy["policy_observation"], "policy_observation")
    action = _object(policy["action"], "action")
    compatibility = _object(baseline_data["compatibility"], "baseline compatibility")
    expected_pairs = (
        ("backend_id", policy["backend_id"], compatibility["backend_id"]),
        ("robot", policy["robot"], compatibility["robot"]),
        ("robot_asset_id", policy["robot_asset_id"], compatibility["robot_asset_id"]),
        (
            "robot_schema_hash",
            robot_schema["schema_hash"],
            compatibility["robot_schema_hash"],
        ),
        (
            "policy_architecture",
            policy["policy_architecture"],
            compatibility["policy_architecture"],
        ),
        (
            "policy_observation_dim",
            observation["total_dim"],
            compatibility["policy_observation_dim"],
        ),
        (
            "policy_observation_schema_hash",
            observation["schema_hash"],
            compatibility["policy_observation_schema_hash"],
        ),
        ("action_dim", action["total_dim"], compatibility["action_dim"]),
        (
            "action_schema_hash",
            action["schema_hash"],
            compatibility["action_schema_hash"],
        ),
    )
    for name, expected, observed in expected_pairs:
        if expected != observed:
            raise ValueError(f"baseline/protocol compatibility mismatch: {name}")


def _validate_phase0_report(payload: Mapping[str, Any]) -> None:
    _expect_exact_keys(
        payload,
        {
            "schema_version",
            "task_id",
            "protocol_id",
            "task_protocol_hash",
            "baseline_id",
            "baseline_manifest_hash",
            "checkpoint",
            "checkpoint_safety",
            "source_artifacts",
            "status",
            "runtime_versions",
            "checkpoint_structure",
            "environment",
            "geometry",
            "rollout",
            "admission",
            "report_hash",
        },
        "stair Phase 0 report",
    )
    _expect_equal(
        payload["schema_version"],
        STAIR_PHASE0_REPORT_SCHEMA_VERSION,
        "Phase 0 report schema_version",
    )
    _expect_equal(payload["task_id"], STAIR_TASK_ID, "Phase 0 report task_id")
    _require_non_empty_text(payload["protocol_id"], "Phase 0 report protocol_id")
    _require_sha256(payload["task_protocol_hash"], "Phase 0 task_protocol_hash")
    _require_non_empty_text(payload["baseline_id"], "Phase 0 baseline_id")
    _require_sha256(
        payload["baseline_manifest_hash"],
        "Phase 0 baseline_manifest_hash",
    )
    if payload["status"] not in {"pass", "fail"}:
        raise ValueError("Phase 0 report status must be pass or fail")

    checkpoint = _object(payload["checkpoint"], "Phase 0 checkpoint")
    _expect_exact_keys(checkpoint, {"sha256", "size_bytes"}, "Phase 0 checkpoint")
    _require_sha256(checkpoint["sha256"], "Phase 0 checkpoint sha256")
    _positive_int(checkpoint["size_bytes"], "Phase 0 checkpoint size_bytes")
    validate_checkpoint_safety_report(payload["checkpoint_safety"])
    source_artifacts = _object(
        payload["source_artifacts"],
        "Phase 0 source_artifacts",
    )
    _expect_exact_keys(
        source_artifacts,
        set(PHASE0_SOURCE_ARTIFACT_REFS),
        "Phase 0 source_artifacts",
    )
    for ref, digest in source_artifacts.items():
        _require_non_empty_text(ref, "Phase 0 source artifact ref")
        _require_sha256(digest, f"Phase 0 source artifact {ref}")

    versions = _object(payload["runtime_versions"], "Phase 0 runtime_versions")
    _expect_exact_keys(
        versions,
        {
            "isaac_lab",
            "isaaclab_extension",
            "isaaclab_tasks_extension",
            "isaaclab_rl_extension",
            "rsl_rl",
            "torch",
        },
        "Phase 0 runtime_versions",
    )
    for key, raw in versions.items():
        _require_non_empty_text(raw, f"Phase 0 runtime version {key}")

    structure = _object(
        payload["checkpoint_structure"],
        "Phase 0 checkpoint_structure",
    )
    _expect_exact_keys(
        structure,
        {
            "iteration",
            "top_level_keys",
            "model_state_key_count",
            "actor_input_dim",
            "actor_hidden_dims",
            "actor_output_dim",
            "critic_input_dim",
            "critic_hidden_dims",
            "critic_output_dim",
        },
        "Phase 0 checkpoint_structure",
    )
    for key in (
        "iteration",
        "model_state_key_count",
        "actor_input_dim",
        "actor_output_dim",
        "critic_input_dim",
        "critic_output_dim",
    ):
        parser = _non_negative_int if key == "iteration" else _positive_int
        parser(structure[key], f"Phase 0 checkpoint_structure {key}")
    for key in ("top_level_keys", "actor_hidden_dims", "critic_hidden_dims"):
        values = _sequence(structure[key], f"Phase 0 checkpoint_structure {key}")
        if not values:
            raise ValueError(f"Phase 0 checkpoint_structure {key} must not be empty")
        for index, raw in enumerate(values):
            if key == "top_level_keys":
                _require_non_empty_text(raw, f"{key} {index}")
            else:
                _positive_int(raw, f"{key} {index}")

    environment = _object(payload["environment"], "Phase 0 environment")
    _expect_exact_keys(
        environment,
        {
            "gym_id",
            "device",
            "num_envs",
            "simulation_dt_s",
            "control_dt_s",
            "policy_observation_shape",
            "action_shape",
            "robot_schema",
            "policy_observation_schema",
            "action_schema",
            "termination_schema",
            "foot_contact_body_names",
            "terrain_types",
            "terrain_levels",
            "terrain_grid_origins_m",
            "env_origins_m",
            "observation_corruption_enabled",
        },
        "Phase 0 environment",
    )
    for key in ("gym_id", "device"):
        _require_non_empty_text(environment[key], f"Phase 0 environment {key}")
    _positive_int(environment["num_envs"], "Phase 0 environment num_envs")
    _positive_number(
        environment["simulation_dt_s"],
        "Phase 0 environment simulation_dt_s",
    )
    _positive_number(
        environment["control_dt_s"],
        "Phase 0 environment control_dt_s",
    )
    for key in ("policy_observation_shape", "action_shape"):
        shape = _sequence(environment[key], f"Phase 0 environment {key}")
        if len(shape) != 2:
            raise ValueError(f"Phase 0 environment {key} must have rank two")
        for index, raw in enumerate(shape):
            _positive_int(raw, f"Phase 0 environment {key} {index}")
    for key in ("foot_contact_body_names",):
        names = _sequence(environment[key], f"Phase 0 environment {key}")
        if not names or len(names) != len(set(names)):
            raise ValueError(f"Phase 0 environment {key} must be non-empty and unique")
        for index, raw in enumerate(names):
            _require_non_empty_text(raw, f"Phase 0 environment {key} {index}")
    for key in ("terrain_types", "terrain_levels"):
        values = _sequence(environment[key], f"Phase 0 environment {key}")
        for index, raw in enumerate(values):
            _non_negative_int(raw, f"Phase 0 environment {key} {index}")
    for key in (
        "robot_schema",
        "policy_observation_schema",
        "action_schema",
        "termination_schema",
    ):
        _object(environment[key], f"Phase 0 environment {key}")
    _assert_finite_json(
        environment["terrain_grid_origins_m"],
        "Phase 0 environment terrain_grid_origins_m",
    )
    _assert_finite_json(
        environment["env_origins_m"],
        "Phase 0 environment env_origins_m",
    )
    _require_bool(
        environment["observation_corruption_enabled"],
        "Phase 0 environment observation_corruption_enabled",
    )

    geometry = _sequence(payload["geometry"], "Phase 0 geometry")
    if not geometry:
        raise ValueError("Phase 0 geometry must not be empty")
    for index, raw in enumerate(geometry):
        item = _object(raw, f"Phase 0 geometry {index}")
        _expect_exact_keys(
            item,
            {
                "bin_id",
                "height_m",
                "mesh_count",
                "origin_m",
                "expected_top_levels_m",
                "missing_top_levels_m",
                "max_height_m",
                "is_non_flat",
            },
            f"Phase 0 geometry {index}",
        )
        _require_non_empty_text(item["bin_id"], f"Phase 0 geometry {index} bin_id")
        _positive_number(item["height_m"], f"Phase 0 geometry {index} height_m")
        _positive_int(item["mesh_count"], f"Phase 0 geometry {index} mesh_count")
        _assert_finite_json(item["origin_m"], f"Phase 0 geometry {index} origin_m")
        _assert_finite_json(
            item["expected_top_levels_m"],
            f"Phase 0 geometry {index} expected_top_levels_m",
        )
        if list(_sequence(item["missing_top_levels_m"], "missing_top_levels_m")):
            raise ValueError("Phase 0 geometry has missing stair top levels")
        _positive_number(
            item["max_height_m"],
            f"Phase 0 geometry {index} max_height_m",
        )
        _expect_equal(
            item["is_non_flat"],
            True,
            f"Phase 0 geometry {index} is_non_flat",
        )

    rollout = _object(payload["rollout"], "Phase 0 rollout")
    _expect_exact_keys(
        rollout,
        {
            "seed",
            "control_steps_limit",
            "control_steps_executed",
            "envs_per_bin",
            "first_episodes",
            "summary_per_bin",
            "elapsed_s",
            "aggregate_env_steps_per_s",
            "all_values_finite",
        },
        "Phase 0 rollout",
    )
    _positive_int(rollout["seed"], "Phase 0 rollout seed")
    control_steps_limit = _positive_int(
        rollout["control_steps_limit"],
        "Phase 0 control_steps_limit",
    )
    control_steps_executed = _positive_int(
        rollout["control_steps_executed"],
        "Phase 0 control_steps_executed",
    )
    if control_steps_executed > control_steps_limit:
        raise ValueError("Phase 0 control_steps_executed exceeds limit")
    _positive_int(rollout["envs_per_bin"], "Phase 0 envs_per_bin")
    first_episodes = _object(rollout["first_episodes"], "Phase 0 first_episodes")
    for bin_id, raw_records in first_episodes.items():
        _require_non_empty_text(bin_id, "Phase 0 first_episodes bin_id")
        records = _sequence(raw_records, f"Phase 0 first_episodes {bin_id}")
        if not records:
            raise ValueError("Phase 0 first_episodes bins must not be empty")
        for index, raw_record in enumerate(records):
            record = _object(raw_record, f"Phase 0 first episode {bin_id} {index}")
            _expect_exact_keys(
                record,
                {
                    "env_index",
                    "terminal_reason",
                    "terminal_terms",
                    "episode_steps",
                    "max_progress_m",
                    "reached_first_step",
                    "max_top_hold_steps",
                    "success",
                    "max_fall_consecutive_steps",
                    "fall_detected",
                },
                f"Phase 0 first episode {bin_id} {index}",
            )
            _non_negative_int(record["env_index"], "Phase 0 first episode env_index")
            _require_non_empty_text(
                record["terminal_reason"],
                "Phase 0 first episode terminal_reason",
            )
            terminal_terms = _sequence(
                record["terminal_terms"],
                "Phase 0 first episode terminal_terms",
            )
            if not terminal_terms or len(terminal_terms) != len(set(terminal_terms)):
                raise ValueError(
                    "Phase 0 first episode terminal_terms must be non-empty and unique"
                )
            for term_index, raw_term in enumerate(terminal_terms):
                _require_non_empty_text(
                    raw_term,
                    f"Phase 0 first episode terminal_terms {term_index}",
                )
            episode_steps = _positive_int(
                record["episode_steps"],
                "Phase 0 first episode episode_steps",
            )
            if episode_steps > control_steps_limit:
                raise ValueError("Phase 0 first episode exceeds control step limit")
            _finite_number(record["max_progress_m"], "Phase 0 first episode max_progress_m")
            for key in ("reached_first_step", "success", "fall_detected"):
                _require_bool(record[key], f"Phase 0 first episode {key}")
            max_top_hold_steps = _non_negative_int(
                record["max_top_hold_steps"],
                "Phase 0 first episode max_top_hold_steps",
            )
            if max_top_hold_steps > episode_steps:
                raise ValueError("Phase 0 top-hold steps exceed episode length")
            max_fall_consecutive_steps = _non_negative_int(
                record["max_fall_consecutive_steps"],
                "Phase 0 first episode max_fall_consecutive_steps",
            )
            if max_fall_consecutive_steps > episode_steps:
                raise ValueError("Phase 0 fall streak exceeds episode length")
    summary_per_bin = _object(
        rollout["summary_per_bin"],
        "Phase 0 summary_per_bin",
    )
    for bin_id, raw_summary in summary_per_bin.items():
        _require_non_empty_text(bin_id, "Phase 0 summary_per_bin bin_id")
        summary = _object(raw_summary, f"Phase 0 summary_per_bin {bin_id}")
        _expect_exact_keys(
            summary,
            {
                "episode_count",
                "terminal_reason_counts",
                "reached_first_step_count",
                "fall_detected_count",
                "success_count",
                "mean_max_progress_m",
            },
            f"Phase 0 summary_per_bin {bin_id}",
        )
        episode_count = _non_negative_int(
            summary["episode_count"],
            f"Phase 0 summary_per_bin {bin_id} episode_count",
        )
        terminal_reason_counts = _object(
            summary["terminal_reason_counts"],
            f"Phase 0 summary_per_bin {bin_id} terminal_reason_counts",
        )
        if not terminal_reason_counts:
            raise ValueError("Phase 0 terminal_reason_counts must not be empty")
        counted_episodes = 0
        for reason, raw_count in terminal_reason_counts.items():
            _require_non_empty_text(reason, "Phase 0 terminal reason count key")
            counted_episodes += _non_negative_int(
                raw_count,
                f"Phase 0 summary_per_bin {bin_id} terminal_reason_counts {reason}",
            )
        if counted_episodes != episode_count:
            raise ValueError(
                "Phase 0 terminal reason counts must equal episode_count"
            )
        for key in (
            "reached_first_step_count",
            "fall_detected_count",
            "success_count",
        ):
            _non_negative_int(summary[key], f"Phase 0 summary_per_bin {bin_id} {key}")
        _finite_number(
            summary["mean_max_progress_m"],
            f"Phase 0 summary_per_bin {bin_id} mean_max_progress_m",
        )
    _positive_number(rollout["elapsed_s"], "Phase 0 elapsed_s")
    _positive_number(
        rollout["aggregate_env_steps_per_s"],
        "Phase 0 aggregate_env_steps_per_s",
    )
    _expect_equal(
        rollout["all_values_finite"],
        True,
        "Phase 0 all_values_finite",
    )

    admission = _object(payload["admission"], "Phase 0 admission")
    _expect_exact_keys(
        admission,
        {
            "checkpoint_structurally_compatible",
            "runtime_schema_compatible",
            "terrain_binding_compatible",
            "first_episode_probe_complete",
            "task_viability_gate_passed",
            "warm_start_candidate_admitted",
            "accepted_current_best",
            "baseline_eval_only_still_required",
            "session_genesis_still_required",
            "report_is_acceptance_evidence",
        },
        "Phase 0 admission",
    )
    for key in (
        "checkpoint_structurally_compatible",
        "runtime_schema_compatible",
        "terrain_binding_compatible",
        "first_episode_probe_complete",
        "task_viability_gate_passed",
        "warm_start_candidate_admitted",
    ):
        _require_bool(admission[key], f"Phase 0 {key}")
    _expect_equal(
        admission["accepted_current_best"],
        False,
        "Phase 0 accepted_current_best",
    )
    _expect_equal(
        admission["baseline_eval_only_still_required"],
        True,
        "Phase 0 baseline_eval_only_still_required",
    )
    _expect_equal(
        admission["session_genesis_still_required"],
        True,
        "Phase 0 session_genesis_still_required",
    )
    _expect_equal(
        admission["report_is_acceptance_evidence"],
        False,
        "Phase 0 report_is_acceptance_evidence",
    )

    declared_hash = _require_sha256(payload["report_hash"], "Phase 0 report_hash")
    if declared_hash != _hash_without_key(payload, "report_hash"):
        raise ValueError("Phase 0 report_hash mismatch")


def _validate_report_bundle_compatibility(
    report: Mapping[str, Any],
    protocol: StairProtocolV1,
    baseline: StairBaselineManifestV1,
) -> None:
    protocol_data = protocol.to_dict()
    baseline_data = baseline.to_dict()
    expected_identity = {
        "task_id": protocol.task_id,
        "protocol_id": protocol.protocol_id,
        "task_protocol_hash": protocol.protocol_hash,
        "baseline_id": baseline.baseline_id,
        "baseline_manifest_hash": baseline.manifest_hash,
    }
    for key, expected in expected_identity.items():
        if report[key] != expected:
            raise ValueError(f"Phase 0 report binding mismatch: {key}")
    checkpoint = _object(report["checkpoint"], "Phase 0 checkpoint")
    if checkpoint["sha256"] != baseline.checkpoint_sha256:
        raise ValueError("Phase 0 report binding mismatch: checkpoint sha256")
    if checkpoint["size_bytes"] != baseline.checkpoint_size_bytes:
        raise ValueError("Phase 0 report binding mismatch: checkpoint size")
    validate_checkpoint_safety_report(
        report["checkpoint_safety"],
        expected_checkpoint_hash=f"sha256:{baseline.checkpoint_sha256}",
    )
    if report["runtime_versions"] != baseline_data["version_lock"]:
        raise ValueError("Phase 0 report binding mismatch: runtime version lock")

    compatibility = baseline_data["compatibility"]
    structure = report["checkpoint_structure"]
    required_checkpoint_keys = set(
        baseline_data["checkpoint"]["required_top_level_keys"]
    )
    if not required_checkpoint_keys.issubset(structure["top_level_keys"]):
        raise ValueError("Phase 0 report checkpoint required keys mismatch")
    structure_checks = {
        "iteration": baseline_data["checkpoint"]["iteration"],
        "actor_input_dim": compatibility["policy_observation_dim"],
        "actor_hidden_dims": compatibility["actor_hidden_dims"],
        "actor_output_dim": compatibility["actor_output_dim"],
        "critic_input_dim": compatibility["policy_observation_dim"],
        "critic_hidden_dims": compatibility["critic_hidden_dims"],
        "critic_output_dim": 1,
    }
    for key, expected in structure_checks.items():
        if structure[key] != expected:
            raise ValueError(f"Phase 0 report checkpoint mismatch: {key}")

    environment = report["environment"]
    evaluation = protocol_data["evaluation"]
    policy = protocol_data["policy_compatibility"]
    expected_num_envs = evaluation["protocol"]["eval_num_envs"]
    environment_checks = {
        "num_envs": expected_num_envs,
        "simulation_dt_s": policy["simulation_dt_s"],
        "control_dt_s": policy["control_dt_s"],
        "policy_observation_shape": [
            expected_num_envs,
            policy["policy_observation"]["total_dim"],
        ],
        "action_shape": [expected_num_envs, policy["action"]["total_dim"]],
        "robot_schema": policy["robot_schema"],
        "policy_observation_schema": policy["policy_observation"],
        "action_schema": policy["action"],
        "termination_schema": protocol_data["phase0_probe"]["termination_schema"],
        "terrain_grid_origins_m": protocol_data["terrain"]["evaluation_grid_origins_m"],
        "observation_corruption_enabled": False,
    }
    for key, expected in environment_checks.items():
        if environment[key] != expected:
            raise ValueError(f"Phase 0 report environment mismatch: {key}")
    if environment["gym_id"] != PHASE0_STAIR_GYM_ID:
        raise ValueError("Phase 0 report environment mismatch: gym_id")
    expected_types = [
        terrain_type
        for terrain_type in range(len(protocol_data["terrain"]["evaluation_bins"]))
        for _ in range(evaluation["env_to_bin_mapping"]["envs_per_bin"])
    ]
    if environment["terrain_types"] != expected_types:
        raise ValueError("Phase 0 report environment mismatch: terrain_types")
    if environment["terrain_levels"] != [0] * expected_num_envs:
        raise ValueError("Phase 0 report environment mismatch: terrain_levels")
    expected_foot_contact_bodies = [
        name
        for name in policy["robot_schema"]["body_names"]
        if name.endswith("_ankle_roll_link")
    ]
    if environment["foot_contact_body_names"] != expected_foot_contact_bodies:
        raise ValueError("Phase 0 report environment mismatch: foot contact bodies")
    expected_env_origins = [
        protocol_data["terrain"]["evaluation_grid_origins_m"][terrain_type]
        for terrain_type in expected_types
    ]
    if environment["env_origins_m"] != expected_env_origins:
        raise ValueError("Phase 0 report environment mismatch: env_origins_m")

    terrain = protocol_data["terrain"]
    expected_origin = [
        terrain["start_platform_length_m"] / 2.0,
        terrain["size_m"][1] / 2.0,
        0.0,
    ]
    for item, bin_spec in zip(report["geometry"], terrain["evaluation_bins"], strict=True):
        if item["bin_id"] != bin_spec["bin_id"] or item["height_m"] != bin_spec["height_m"]:
            raise ValueError("Phase 0 report geometry bin mismatch")
        if item["origin_m"] != expected_origin:
            raise ValueError("Phase 0 report geometry origin mismatch")
        if item["mesh_count"] != terrain["step_count"] + 2:
            raise ValueError("Phase 0 report geometry mesh count mismatch")
        expected_top = terrain["step_count"] * bin_spec["height_m"]
        if not math.isclose(item["max_height_m"], expected_top, abs_tol=1e-9):
            raise ValueError("Phase 0 report geometry top mismatch")
        expected_levels = [
            round(level * bin_spec["height_m"], 8)
            for level in range(terrain["step_count"] + 1)
        ]
        if item["expected_top_levels_m"] != expected_levels:
            raise ValueError("Phase 0 report geometry level mismatch")

    rollout = report["rollout"]
    if rollout["envs_per_bin"] != evaluation["env_to_bin_mapping"]["envs_per_bin"]:
        raise ValueError("Phase 0 report rollout envs_per_bin mismatch")
    probe = protocol_data["phase0_probe"]
    if rollout["seed"] != probe["seed"]:
        raise ValueError("Phase 0 report rollout seed mismatch")
    if rollout["control_steps_limit"] != probe["control_steps"]:
        raise ValueError("Phase 0 report control step limit mismatch")
    first_episodes = rollout["first_episodes"]
    summary_per_bin = rollout["summary_per_bin"]
    expected_bin_ids = {item["bin_id"] for item in terrain["evaluation_bins"]}
    if set(first_episodes) != expected_bin_ids or set(summary_per_bin) != expected_bin_ids:
        raise ValueError("Phase 0 report rollout bin coverage mismatch")
    envs_per_bin = int(rollout["envs_per_bin"])
    first_step_progress = float(probe["first_step_progress_m_from_origin"])
    goal_progress = float(terrain["goal"]["progress_goal_m_from_origin"])
    success_hold_steps = int(
        protocol_data["episode_outcomes"]["success"]["top_hold_control_steps"]
    )
    fall_consecutive_steps = int(
        protocol_data["episode_outcomes"]["signals"]["fall"][
            "consecutive_control_steps"
        ]
    )
    termination_term_order = [
        item["manager_term"] for item in probe["termination_schema"]["terms"]
    ]
    timeout_manager_terms, terminated_manager_terms = phase0_manager_term_groups(
        protocol_data
    )
    timeout_manager_term_set = set(timeout_manager_terms)
    terminated_manager_term_set = set(terminated_manager_terms)
    terminal_reason_names = phase0_terminal_reason_names(protocol_data)
    reached_counts: dict[str, int] = {}
    all_episode_steps: list[int] = []
    for bin_index, bin_spec in enumerate(terrain["evaluation_bins"]):
        bin_id = bin_spec["bin_id"]
        records = first_episodes[bin_id]
        if len(records) != envs_per_bin:
            raise ValueError("Phase 0 report must contain one first episode per env")
        expected_env_indices = list(
            range(bin_index * envs_per_bin, (bin_index + 1) * envs_per_bin)
        )
        if [record["env_index"] for record in records] != expected_env_indices:
            raise ValueError("Phase 0 report first episode env index mapping mismatch")
        reached_count = 0
        for record in records:
            episode_steps = int(record["episode_steps"])
            all_episode_steps.append(episode_steps)
            terminal_terms = list(record["terminal_terms"])
            if any(term not in termination_term_order for term in terminal_terms):
                raise ValueError("Phase 0 terminal term is not in the frozen manager schema")
            if terminal_terms != [
                term for term in termination_term_order if term in terminal_terms
            ]:
                raise ValueError("Phase 0 terminal term order mismatch")
            expected_terminal_reason = resolve_phase0_terminal_reason(
                protocol_data,
                terminal_terms,
            )
            if record["terminal_reason"] != expected_terminal_reason:
                raise ValueError("Phase 0 terminal reason mismatch term evidence")
            active_terminal_terms = set(terminal_terms)
            has_timeout = bool(active_terminal_terms & timeout_manager_term_set)
            has_termination = bool(
                active_terminal_terms & terminated_manager_term_set
            )
            if has_timeout and episode_steps != probe["control_steps"]:
                raise ValueError("Phase 0 timeout episode must span the full horizon")
            if episode_steps == probe["control_steps"] and not has_timeout:
                raise ValueError("Phase 0 full-horizon episode is missing timeout term")
            max_progress = float(record["max_progress_m"])
            expected_reached = max_progress >= first_step_progress
            if record["reached_first_step"] is not expected_reached:
                raise ValueError("Phase 0 reached_first_step evidence mismatch")
            reached_count += int(expected_reached)
            expected_fall = (
                int(record["max_fall_consecutive_steps"])
                >= fall_consecutive_steps
            )
            if record["fall_detected"] is not expected_fall:
                raise ValueError("Phase 0 fall_detected evidence mismatch")
            expected_success = (
                has_timeout
                and not has_termination
                and max_progress >= goal_progress
                and int(record["max_top_hold_steps"]) >= success_hold_steps
                and not expected_fall
            )
            if record["success"] is not expected_success:
                raise ValueError("Phase 0 success evidence mismatch")
        reached_counts[bin_id] = reached_count
        expected_summary = {
            "episode_count": len(records),
            "terminal_reason_counts": {
                reason: sum(
                    record["terminal_reason"] == reason for record in records
                )
                for reason in terminal_reason_names
            },
            "reached_first_step_count": sum(
                bool(record["reached_first_step"]) for record in records
            ),
            "fall_detected_count": sum(
                bool(record["fall_detected"]) for record in records
            ),
            "success_count": sum(bool(record["success"]) for record in records),
            "mean_max_progress_m": round(
                sum(float(record["max_progress_m"]) for record in records)
                / len(records),
                8,
            ),
        }
        if summary_per_bin[bin_id] != expected_summary:
            raise ValueError("Phase 0 per-bin summary mismatch first-episode evidence")
    if rollout["control_steps_executed"] != max(all_episode_steps):
        raise ValueError("Phase 0 control_steps_executed mismatch first episodes")

    minimum_fraction = float(
        probe["minimum_reached_first_step_fraction_per_bin"]
    )
    viability_passed = all(
        reached_counts[bin_id] / envs_per_bin >= minimum_fraction
        for bin_id in probe["in_distribution_bin_ids"]
    )
    admission = report["admission"]
    for key in (
        "checkpoint_structurally_compatible",
        "runtime_schema_compatible",
        "terrain_binding_compatible",
        "first_episode_probe_complete",
    ):
        if admission[key] is not True:
            raise ValueError(f"Phase 0 report admission mismatch: {key}")
    if admission["task_viability_gate_passed"] is not viability_passed:
        raise ValueError("Phase 0 report task viability gate mismatch")
    if admission["warm_start_candidate_admitted"] is not viability_passed:
        raise ValueError("Phase 0 report warm-start admission mismatch")
    expected_status = "pass" if viability_passed else "fail"
    if report["status"] != expected_status:
        raise ValueError("Phase 0 report status does not match admission evidence")


def _validate_report_source_artifacts(
    report: Mapping[str, Any],
    project_root: Path,
) -> None:
    root = project_root.resolve(strict=True)
    declared = _object(report["source_artifacts"], "Phase 0 source_artifacts")
    for ref in PHASE0_SOURCE_ARTIFACT_REFS:
        path = _resolve_project_resource(root, ref)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if declared[ref] != digest.hexdigest():
            raise ValueError(f"Phase 0 report source artifact mismatch: {ref}")


def _resolve_project_resource(root: Path, ref: str | Path) -> Path:
    resource_ref = Path(ref)
    if resource_ref.is_absolute():
        raise ValueError("Phase 0 resource refs must be project-relative")
    path = (root / resource_ref).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Phase 0 resource ref escapes project root") from exc
    if not path.is_file():
        raise ValueError(f"Phase 0 resource is not a file: {resource_ref}")
    return path


def _resolve_project_output(root: Path, ref: str | Path) -> Path:
    resource_ref = Path(ref)
    if resource_ref.is_absolute():
        raise ValueError("Phase 0 output refs must be project-relative")
    path = (root / resource_ref).resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Phase 0 output ref escapes project root") from exc
    return path


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        return load_strict_json_object(
            path,
            context=str(path),
            require_canonical=True,
            trailing_newline=True,
        )
    except ValueError as exc:
        raise ValueError(f"invalid JSON resource: {path}") from exc


def _copy_json_object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    try:
        copied = strict_json_object_loads(
            canonical_json_bytes(value),
            context=context,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must contain only finite JSON values") from exc
    return copied


def _assert_resolved_json(
    value: Any,
    context: str,
    *,
    allowed_null_contexts: frozenset[str] = frozenset(),
) -> None:
    if value is None:
        if context in allowed_null_contexts:
            return
        raise ValueError(f"{context} contains null")
    if isinstance(value, str):
        upper = value.upper()
        if value in _PLACEHOLDER_VALUES or "TODO" in upper or "TBD" in upper:
            raise ValueError(f"{context} contains unresolved placeholder")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_resolved_json(
                item,
                f"{context}.{key}",
                allowed_null_contexts=allowed_null_contexts,
            )
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _assert_resolved_json(
                item,
                f"{context}[{index}]",
                allowed_null_contexts=allowed_null_contexts,
            )
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{context} contains non-finite number")


def _assert_finite_json(value: Any, context: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        raise ValueError(f"{context} must contain only finite numbers or arrays")
    if isinstance(value, int | float):
        _finite_number(value, context)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            raise ValueError(f"{context} must not be empty")
        for index, item in enumerate(value):
            _assert_finite_json(item, f"{context}[{index}]")
        return
    raise ValueError(f"{context} must contain only finite numbers or arrays")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _hash_without_key(value: Mapping[str, Any], key: str) -> str:
    payload = {name: _thaw_json(item) for name, item in value.items() if name != key}
    return canonical_json_sha256(payload)


def _expect_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    context: str,
) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        unknown = sorted(observed - expected)
        raise ValueError(
            f"{context} keys mismatch; missing={missing}, unknown={unknown}"
        )


def _expect_equal(observed: Any, expected: Any, context: str) -> None:
    if observed != expected or type(observed) is not type(expected):
        raise ValueError(f"{context} must equal {expected!r}")


def _object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _sequence(value: Any, context: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{context} must be an array")
    return value


def _unique_text_tuple(value: Any, context: str) -> tuple[str, ...]:
    items = tuple(
        _require_non_empty_text(raw, f"{context} {index}")
        for index, raw in enumerate(_sequence(value, context))
    )
    if len(items) != len(set(items)):
        raise ValueError(f"{context} must contain unique values")
    return items


def _require_non_empty_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be non-empty text")
    return value


def _require_bool(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{context} must be boolean")
    return value


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{context} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{context} must be finite")
    return parsed


def _positive_number(value: Any, context: str) -> float:
    parsed = _finite_number(value, context)
    if parsed <= 0.0:
        raise ValueError(f"{context} must be positive")
    return parsed


def _non_negative_number(value: Any, context: str) -> float:
    parsed = _finite_number(value, context)
    if parsed < 0.0:
        raise ValueError(f"{context} must be non-negative")
    return parsed


def _positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _non_negative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def _number_pair(value: Any, context: str, *, positive: bool) -> tuple[float, float]:
    sequence = _sequence(value, context)
    if len(sequence) != 2:
        raise ValueError(f"{context} must contain exactly two numbers")
    parser = _positive_number if positive else _finite_number
    return (parser(sequence[0], context), parser(sequence[1], context))


def _ordered_number_pair(value: Any, context: str) -> tuple[float, float]:
    lower, upper = _number_pair(value, context, positive=False)
    if lower > upper:
        raise ValueError(f"{context} lower bound must not exceed upper bound")
    return (lower, upper)


def _require_sha256(value: Any, context: str) -> str:
    text = _require_non_empty_text(value, context)
    if not _SHA256_PATTERN.fullmatch(text):
        raise ValueError(f"{context} must be a lowercase SHA-256 hex digest")
    return text


__all__ = [
    "DEFAULT_STAIR_EXTENSION_DESCRIPTOR_REF",
    "FROZEN_STAIR_EXTENSION_DESCRIPTOR_V1_HASH",
    "PHASE0_SOURCE_ARTIFACT_REFS",
    "PHASE0_STAIR_GYM_ID",
    "PHASE0_TRANSITION_EXTRAS_KEY",
    "STAIR_BASELINE_MANIFEST_SCHEMA_VERSION",
    "STAIR_EXTENSION_DESCRIPTOR_SCHEMA_VERSION",
    "STAIR_PHASE0_REPORT_SCHEMA_VERSION",
    "STAIR_PROTOCOL_SCHEMA_VERSION",
    "STAIR_TASK_ID",
    "CheckpointFileVerification",
    "StairBaselineManifestV1",
    "StairExtensionDescriptorV1",
    "StairPhase0Bundle",
    "StairPhase0ReportV1",
    "StairProtocolV1",
    "canonical_json_sha256",
    "phase0_manager_term_groups",
    "phase0_terminal_reason_names",
    "resolve_phase0_terminal_reason",
    "verify_checkpoint_file",
]
