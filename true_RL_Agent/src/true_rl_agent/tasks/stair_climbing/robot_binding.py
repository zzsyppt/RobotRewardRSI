"""Stair-owned binding of the legacy G1-37 execution cell."""

from __future__ import annotations

from pathlib import Path

from true_rl_agent.composition.cell_manifest import (
    CellEnvironmentDescriptor,
    CellResolution,
    CellVariantCatalog,
    CellVariantDefinition,
    ResolvedCellManifestV1,
    RobotTaskBinding,
    compile_cell_manifest,
    grant_legacy_training_binding_v1,
)
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.json_contract import load_strict_json_object
from true_rl_agent.robots.defaults import (
    G1_37DOF_ISAACLAB_ADAPTER_ID,
    G1_37DOF_ISAACLAB_PROFILE_ID,
    build_default_robot_registry,
)
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

STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID = "stair.g1_37dof.clean_legacy_v1"
STAIR_G1_37DOF_ROBOT_TASK_CELL_ID = "unitree_stair_climbing.g1_37dof_v1"

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_BINDING_PATH = (
    _PROJECT_ROOT / "configs/tasks/stair_g1_37dof_clean_legacy_binding_v1.json"
)


def build_stair_cell_catalog(
    *,
    config: StairClimbingTaskConfig,
    plugin: StairClimbingTaskPlugin,
    robot_registry: RobotRegistry | None = None,
) -> CellVariantCatalog:
    if plugin.config is not config:
        raise ValueError("stair robot binding config/plugin mismatch")
    registry = robot_registry or build_default_robot_registry()
    phase2 = config.protocol_v2.to_public_dict()
    binding = RobotTaskBinding.from_mapping(
        load_strict_json_object(
            _BINDING_PATH,
            context="Stair legacy robot task binding",
            require_canonical=True,
            trailing_newline=True,
        )
    )
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
    environment = CellEnvironmentDescriptor(
        environment_id="stair_clean_legacy_environment_v1",
        cell_variant_id=STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID,
        task_id=STAIR_TASK_ID,
        payload={
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
        },
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
    catalog.register(
        grant_legacy_training_binding_v1(
            CellVariantDefinition(
                cell_variant_id=STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID,
                robot_task_cell_id=STAIR_G1_37DOF_ROBOT_TASK_CELL_ID,
                task_id=STAIR_TASK_ID,
                robot_profile_id=G1_37DOF_ISAACLAB_PROFILE_ID,
                runtime_adapter_id=G1_37DOF_ISAACLAB_ADAPTER_ID,
                robot_task_binding=binding,
                environment=environment,
                contract_hashes=contracts,
            ),
            robot_registry=registry,
            requirements=plugin.robot_requirements(),
            base_task_protocol_hash=config.task_protocol_hash,
        ),
    )
    return catalog


def resolve_stair_cell_manifest(
    *,
    cell_variant_id: str,
    config: StairClimbingTaskConfig,
    plugin: StairClimbingTaskPlugin,
) -> ResolvedCellManifestV1:
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
        base_task_protocol_hash=config.task_protocol_hash,
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


__all__ = [
    "STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID",
    "STAIR_G1_37DOF_ROBOT_TASK_CELL_ID",
    "build_stair_cell_catalog",
    "compile_stair_cell_manifest",
    "resolve_stair_cell_manifest",
]
