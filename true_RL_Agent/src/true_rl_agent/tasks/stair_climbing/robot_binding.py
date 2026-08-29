"""Stair-owned thin bindings for robot rows of one task."""

from __future__ import annotations

from true_rl_agent.composition.cell_manifest import (
    CELL_PROTOCOL_STATE_SCHEMA_VERSION,
    CellEnvironmentDescriptor,
    CellResolution,
    CellVariantCatalog,
    CellVariantDefinition,
    ResolvedCellManifestV1,
    compile_cell_manifest,
    grant_legacy_training_binding_v1,
)
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.robots.defaults import build_default_robot_registry
from true_rl_agent.robots.base import RobotRequirements
from true_rl_agent.robots.registry import RobotRegistry
from true_rl_agent.tasks.stair_climbing.config import (
    STAIR_TASK_ID,
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
from true_rl_agent.tasks.stair_climbing.runtime_projection import (
    stair_runtime_projection_from_execution,
    stair_runtime_projection_from_manifest,
)
from true_rl_agent.tasks.stair_climbing.row_catalog import (
    STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID,
    STAIR_G1_37DOF_UNIFIED_CELL_VARIANT_ID,
    STAIR_GO2_12DOF_UNIFIED_CELL_VARIANT_ID,
    stair_robot_row,
)

STAIR_G1_37DOF_ROBOT_TASK_CELL_ID = "unitree_stair_climbing.g1_37dof_v1"
STAIR_GO2_12DOF_ROBOT_TASK_CELL_ID = "unitree_stair_climbing.go2_12dof_v1"


def build_stair_cell_catalog(
    *,
    config: StairClimbingTaskConfig,
    plugin: StairClimbingTaskPlugin,
    robot_registry: RobotRegistry | None = None,
) -> CellVariantCatalog:
    if plugin.config is not config:
        raise ValueError("stair robot binding config/plugin mismatch")
    registry = robot_registry or build_default_robot_registry()
    if config.is_unified_protocol:
        return _build_unified_stair_cell_catalog(
            config=config,
            plugin=plugin,
            registry=registry,
        )
    assert config.protocol_v2 is not None
    phase2 = config.protocol_v2.to_public_dict()
    row = stair_robot_row(config.cell_variant_id)
    binding = row.load_binding()
    profile, adapter = registry.execution(
        robot_profile_id=binding.robot_profile_id,
        runtime_adapter_id=binding.runtime_adapter_id,
    )
    runtime_projection = stair_runtime_projection_from_execution(
        profile=profile,
        adapter=adapter,
        binding=binding,
    )
    facts = binding.task_projection["phase0_binding_facts"]
    environment_id = row.environment_id
    environment_payload = {
        "legacy_cell_key": facts["cell_key"],
        "train_runtime_dynamics_hash": (
            config.train_runtime_profile().runtime_dynamics_hash
        ),
        "eval_runtime_dynamics_hash": (
            config.eval_runtime_profile().runtime_dynamics_hash
        ),
        "terrain_protocol_hash": canonical_json_hash(
            config.protocol_v1.to_dict()["terrain"]
        ),
    }
    environment = CellEnvironmentDescriptor(
        environment_id=environment_id,
        cell_variant_id=config.cell_variant_id,
        task_id=STAIR_TASK_ID,
        payload=environment_payload,
    )
    contracts = {
        "acceptance_protocol_hash": config.acceptance_protocol.protocol_hash,
        "action_content_hash": canonical_json_hash(facts["actions"]),
        "agent_configuration_hash": (
            config.training_agent_configuration().agent_configuration_hash
        ),
        "binding_facts_hash": facts["binding_facts_hash"],
        "candidate_capability_profile_hash": (
            plugin.task_spec.candidate_capability_profile.profile_hash
        ),
        "evaluation_protocol_hash": config.evaluation_protocol().protocol_hash,
        "observation_content_hash": canonical_json_hash(facts["policy_observation"]),
        "policy_io_schema_hash": facts["policy_io_schema_hash"],
        "reference_capability_gate_hash": canonical_json_hash(
            phase2["reference_capability_gate"]
        ),
        "reward_input_schema_hash": canonical_json_hash(
            phase2["complete_reward_runtime"]["input_schema"]
        ),
        "reward_input_vnext_schema_hash": (
            runtime_projection.reward_input_schema_hash_vnext
        ),
        "reward_capability_vnext_hash": (
            runtime_projection.reward_capability_hash_vnext
        ),
        "sensor_schema_hash": facts["sensor_schema_hash"],
    }
    catalog = CellVariantCatalog()
    definition = CellVariantDefinition(
        cell_variant_id=config.cell_variant_id,
        robot_task_cell_id=row.robot_task_cell_id,
        task_id=STAIR_TASK_ID,
        robot_profile_id=binding.robot_profile_id,
        runtime_adapter_id=binding.runtime_adapter_id,
        robot_task_binding=binding,
        environment=environment,
        contract_hashes=contracts,
    )
    catalog.register(
        grant_legacy_training_binding_v1(
            definition,
            robot_registry=registry,
            requirements=plugin.robot_requirements(),
            base_task_protocol_hash=config.base_task_protocol_hash,
        )
    )
    return catalog


def resolve_stair_cell_manifest(
    *,
    cell_variant_id: str,
    config: StairClimbingTaskConfig,
    plugin: StairClimbingTaskPlugin,
) -> ResolvedCellManifestV1:
    if cell_variant_id != config.cell_variant_id:
        raise ValueError("Stair selected cell differs from task config")
    registry = build_default_robot_registry()
    definition = build_stair_cell_catalog(
        config=config,
        plugin=plugin,
        robot_registry=registry,
    ).resolve(
        cell_variant_id
    )
    return compile_stair_cell_manifest(
        definition=definition,
        robot_registry=registry,
        requirements=plugin.robot_requirements(),
        base_task_protocol_hash=config.base_task_protocol_hash,
    ).require_supported()


def compile_stair_cell_manifest(
    *,
    definition: CellVariantDefinition,
    robot_registry: RobotRegistry,
    requirements: RobotRequirements,
    base_task_protocol_hash: str,
) -> CellResolution:
    """Compile the shared matrix cell, then enforce Stair-owned runtime identity."""

    result = compile_cell_manifest(
        definition=definition,
        robot_registry=robot_registry,
        requirements=requirements,
        base_task_protocol_hash=base_task_protocol_hash,
    )
    if not result.supported or result.manifest is None:
        return result
    try:
        stair_runtime_projection_from_manifest(result.manifest)
    except (TypeError, ValueError) as exc:
        return CellResolution(
            "unsupported",
            (f"stair_runtime_projection:{exc}",),
        )
    return result


def _build_unified_stair_cell_catalog(
    *,
    config: StairClimbingTaskConfig,
    plugin: StairClimbingTaskPlugin,
    registry: RobotRegistry,
) -> CellVariantCatalog:
    row = stair_robot_row(config.cell_variant_id)
    binding = row.load_binding()
    profile, adapter = registry.execution(
        robot_profile_id=binding.robot_profile_id,
        runtime_adapter_id=binding.runtime_adapter_id,
    )
    runtime_projection = stair_runtime_projection_from_execution(
        profile=profile,
        adapter=adapter,
        binding=binding,
    )
    authority = config.vnext_reward_authority
    declared_group_inputs = tuple(
        dict(item) for item in authority.joint_group_inputs
    )
    projected_group_inputs = tuple(
        {"role_id": role_id, "source": source_name}
        for _, role_id, source_name, _, _ in (
            runtime_projection.joint_group_input_specs
        )
    )
    if projected_group_inputs != declared_group_inputs:
        raise ValueError("Stair binding/reward authority joint-group inputs diverged")
    projected_input_names = {
        item.name for item in runtime_projection.reward_input_fields_vnext
    }
    missing_common_inputs = (
        set(authority.common_input_field_names) - projected_input_names
    )
    if missing_common_inputs:
        raise ValueError(
            "Stair reward authority inputs are absent from the resolved projection: "
            + ",".join(sorted(missing_common_inputs))
        )
    facts = binding.task_projection["phase0_binding_facts"]
    suite = config.unified_protocol.evaluation_suite
    training_profile = config.unified_training_profile
    environment = CellEnvironmentDescriptor(
        environment_id=row.environment_id,
        cell_variant_id=config.cell_variant_id,
        task_id=STAIR_TASK_ID,
        payload={
            "training_profile_hash": training_profile.profile_hash,
            "evaluation_suite_hash": suite.suite_hash,
            "ordered_cell_set_hash": suite.cell_set_hash,
            "cohort_hashes": {
                cohort: suite.cohort_hash(cohort)
                for cohort in (
                    "calibration",
                    "public_dev",
                    "public_accept",
                    "sealed_validation",
                )
            },
            "train_runtime_dynamics_hash": (
                config.train_runtime_profile().runtime_dynamics_hash
            ),
            "eval_runtime_dynamics_hash": (
                config.eval_runtime_profile().runtime_dynamics_hash
            ),
        },
    )
    protocol_state_hash = canonical_json_hash(
        {
            "schema_version": CELL_PROTOCOL_STATE_SCHEMA_VERSION,
            "cell_variant_id": config.cell_variant_id,
            "protocol_state": config.protocol_state,
        }
    )
    contracts = {
        "acceptance_protocol_hash": config.acceptance_protocol.protocol_hash,
        "action_content_hash": canonical_json_hash(facts["actions"]),
        "agent_configuration_hash": (
            config.training_agent_configuration().agent_configuration_hash
        ),
        "binding_facts_hash": facts["binding_facts_hash"],
        "candidate_capability_profile_hash": (
            config.candidate_capability_profile.profile_hash
        ),
        "evaluation_protocol_hash": config.evaluation_protocol().protocol_hash,
        "evaluation_suite_hash": suite.suite_hash,
        "observation_content_hash": canonical_json_hash(
            facts["policy_observation"]
        ),
        "ordered_cell_set_hash": suite.cell_set_hash,
        "policy_io_schema_hash": facts["policy_io_schema_hash"],
        "protocol_state_hash": protocol_state_hash,
        "reference_capability_gate_hash": canonical_json_hash(
            dict(config.unified_protocol.reference_readiness)
        ),
        "reward_capability_vnext_hash": (
            runtime_projection.reward_capability_hash_vnext
        ),
        "reward_input_schema_hash": (
            runtime_projection.reward_input_schema_hash_vnext
        ),
        "reward_input_vnext_schema_hash": (
            runtime_projection.reward_input_schema_hash_vnext
        ),
        "sensor_schema_hash": facts["sensor_schema_hash"],
        "training_profile_hash": training_profile.profile_hash,
        "vnext_reward_authority_hash": (
            config.vnext_reward_authority.authority_hash
        ),
    }
    definition = CellVariantDefinition(
        cell_variant_id=config.cell_variant_id,
        robot_task_cell_id=row.robot_task_cell_id,
        task_id=STAIR_TASK_ID,
        robot_profile_id=binding.robot_profile_id,
        runtime_adapter_id=binding.runtime_adapter_id,
        robot_task_binding=binding,
        environment=environment,
        contract_hashes=contracts,
        protocol_state=config.protocol_state,
    )
    if definition.protocol_state_hash != protocol_state_hash:
        raise RuntimeError("Stair unified protocol state identity drifted")
    catalog = CellVariantCatalog()
    catalog.register(definition)
    return catalog


__all__ = [
    "STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID",
    "STAIR_G1_37DOF_UNIFIED_CELL_VARIANT_ID",
    "STAIR_G1_37DOF_ROBOT_TASK_CELL_ID",
    "STAIR_GO2_12DOF_ROBOT_TASK_CELL_ID",
    "STAIR_GO2_12DOF_UNIFIED_CELL_VARIANT_ID",
    "build_stair_cell_catalog",
    "compile_stair_cell_manifest",
    "resolve_stair_cell_manifest",
]
