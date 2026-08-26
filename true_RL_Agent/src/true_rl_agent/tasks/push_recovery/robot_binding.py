"""Push-recovery-owned binding of the legacy G1-29 execution cell."""

from __future__ import annotations

from pathlib import Path

from true_rl_agent.composition.cell_manifest import (
    CellEnvironmentDescriptor,
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
    G1_29DOF_UNITREE_RL_LAB_ADAPTER_ID,
    G1_29DOF_UNITREE_RL_LAB_PROFILE_ID,
    build_default_robot_registry,
)
from true_rl_agent.robots.registry import RobotRegistry
from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
from true_rl_agent.tasks.push_recovery.plugin import PushRecoveryTaskPlugin
from true_rl_agent.tasks.push_recovery.reward_input_contract import (
    push_recovery_reward_input_fields,
)

PUSH_RECOVERY_TASK_ID = "unitree_push_recovery"
PUSH_G1_29DOF_LEGACY_CELL_VARIANT_ID = "push.g1_29dof.legacy_v1"
PUSH_G1_29DOF_ROBOT_TASK_CELL_ID = "unitree_push_recovery.g1_29dof_v1"

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_BINDING_PATH = _PROJECT_ROOT / "configs/tasks/push_g1_29dof_legacy_binding_v1.json"


def build_push_cell_catalog(
    *,
    config: PushRecoveryTaskConfig,
    plugin: PushRecoveryTaskPlugin,
    robot_registry: RobotRegistry | None = None,
) -> CellVariantCatalog:
    if plugin.config != config:
        raise ValueError("push robot binding config/plugin mismatch")
    registry = robot_registry or build_default_robot_registry()
    binding = RobotTaskBinding.from_mapping(
        load_strict_json_object(
            _BINDING_PATH,
            context="Push legacy robot task binding",
            require_canonical=True,
            trailing_newline=True,
        )
    )
    facts = binding.task_projection["phase0_binding_facts"]
    environment = CellEnvironmentDescriptor(
        environment_id="push_recovery_legacy_environment_v1",
        cell_variant_id=PUSH_G1_29DOF_LEGACY_CELL_VARIANT_ID,
        task_id=PUSH_RECOVERY_TASK_ID,
        payload={
            "legacy_cell_key": facts["cell_key"],
            "train_runtime_dynamics_hash": config.train_runtime_dynamics()[
                "runtime_dynamics_hash"
            ],
            "eval_runtime_dynamics_hash": config.eval_runtime_dynamics()[
                "runtime_dynamics_hash"
            ],
        },
    )
    contracts = {
        "acceptance_protocol_hash": config.acceptance_protocol().protocol_hash,
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
            {
                "schema_version": "true_rl_agent.not_applicable.v1",
                "reason": "push_recovery_has_no_reference_program_genesis",
            }
        ),
        "reward_input_schema_hash": canonical_json_hash(
            {
                "schema_version": "true_rl_agent.push_reward_inputs.v1",
                "fields": list(push_recovery_reward_input_fields()),
            }
        ),
        "sensor_schema_hash": facts["sensor_schema_hash"],
    }
    catalog = CellVariantCatalog()
    catalog.register(
        grant_legacy_training_binding_v1(
            CellVariantDefinition(
                cell_variant_id=PUSH_G1_29DOF_LEGACY_CELL_VARIANT_ID,
                robot_task_cell_id=PUSH_G1_29DOF_ROBOT_TASK_CELL_ID,
                task_id=PUSH_RECOVERY_TASK_ID,
                robot_profile_id=G1_29DOF_UNITREE_RL_LAB_PROFILE_ID,
                runtime_adapter_id=G1_29DOF_UNITREE_RL_LAB_ADAPTER_ID,
                robot_task_binding=binding,
                environment=environment,
                contract_hashes=contracts,
            ),
            robot_registry=registry,
            requirements=plugin.robot_requirements(),
            base_task_protocol_hash=plugin.task_spec.resolved_task_protocol_hash(),
        ),
    )
    return catalog


def resolve_push_cell_manifest(
    *,
    cell_variant_id: str,
    config: PushRecoveryTaskConfig,
    plugin: PushRecoveryTaskPlugin,
) -> ResolvedCellManifestV1:
    registry = build_default_robot_registry()
    definition = build_push_cell_catalog(
        config=config,
        plugin=plugin,
        robot_registry=registry,
    ).resolve(
        cell_variant_id
    )
    return compile_cell_manifest(
        definition=definition,
        robot_registry=registry,
        requirements=plugin.robot_requirements(),
        base_task_protocol_hash=plugin.task_spec.resolved_task_protocol_hash(),
    ).require_supported()


__all__ = [
    "PUSH_G1_29DOF_LEGACY_CELL_VARIANT_ID",
    "PUSH_G1_29DOF_ROBOT_TASK_CELL_ID",
    "build_push_cell_catalog",
    "resolve_push_cell_manifest",
]
