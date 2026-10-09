"""Push-owned bindings for robot recovery rows and frozen G1-29 history."""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace

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
from true_rl_agent.robots.rsl_rl_runtime import rsl_rl_agent_configuration
from true_rl_agent.tasks.push_recovery.config import (
    PUSH_G1_37DOF_RECOVERY_CELL_VARIANT_ID,
    PUSH_GO2_12DOF_RECOVERY_CELL_VARIANT_ID,
    PushRecoveryTaskConfig,
)
from true_rl_agent.tasks.push_recovery.plugin import PushRecoveryTaskPlugin
from true_rl_agent.tasks.push_recovery.reward_input_contract import (
    push_recovery_reward_input_fields,
)

PUSH_RECOVERY_TASK_ID = "unitree_push_recovery"
PUSH_G1_29DOF_LEGACY_CELL_VARIANT_ID = "push.g1_29dof.legacy_v1"
PUSH_G1_29DOF_REFERENCE_CELL_VARIANT_ID = "push.g1_29dof.reference_v1"
PUSH_G1_29DOF_ROBOT_TASK_CELL_ID = "unitree_push_recovery.g1_29dof_v1"

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_BINDING_PATH = _PROJECT_ROOT / "configs/tasks/push_g1_29dof_legacy_binding_v1.json"
_RECOVERY_ABI_PATH = _PROJECT_ROOT / "configs/tasks/push_recovery_robot_policy_abi_v2.json"


def build_push_cell_catalog(
    *,
    config: PushRecoveryTaskConfig,
    plugin: PushRecoveryTaskPlugin,
    robot_registry: RobotRegistry | None = None,
) -> CellVariantCatalog:
    if plugin.config != config:
        raise ValueError("push robot binding config/plugin mismatch")
    registry = robot_registry or build_default_robot_registry()
    if config.push_protocol is not None:
        return _build_recovery_cell_catalog(config=config, plugin=plugin, registry=registry)
    binding = RobotTaskBinding.from_mapping(
        load_strict_json_object(
            _BINDING_PATH,
            context="Push legacy robot task binding",
            require_canonical=True,
            trailing_newline=True,
        )
    )
    facts = binding.task_projection["phase0_binding_facts"]
    cell_variant_id = (PUSH_G1_29DOF_REFERENCE_CELL_VARIANT_ID
                       if config.reference_program_genesis else PUSH_G1_29DOF_LEGACY_CELL_VARIANT_ID)
    if config.reference_program_genesis:
        binding = replace(binding, binding_id="push_g1_29dof_reference_binding_v1",
                          cell_variant_id=cell_variant_id, binding_hash=None)
    environment = CellEnvironmentDescriptor(
        environment_id="push_recovery_legacy_environment_v1",
        cell_variant_id=cell_variant_id,
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
        "agent_configuration_hash": rsl_rl_agent_configuration(
            binding.runtime_adapter_id
        ).agent_configuration_hash,
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
    if config.reference_program_genesis:
        from true_rl_agent.tasks.push_recovery.reference_program import reference_qualification_contract

        contracts["reference_capability_gate_hash"] = canonical_json_hash(
            reference_qualification_contract(config))
    catalog.register(
        grant_legacy_training_binding_v1(
            CellVariantDefinition(
                cell_variant_id=cell_variant_id,
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


def _build_recovery_cell_catalog(*, config, plugin, registry) -> CellVariantCatalog:
    from true_rl_agent.tasks.push_recovery.reference_program import (
        RECOVERY_REFERENCE_ID, reference_qualification_contract,
    )
    from true_rl_agent.tasks.push_recovery.reward_inputs import PushRecoveryV2RewardInputs

    adapter = registry.runtime_adapter(config.runtime_adapter_id)
    profile = registry.profile(adapter.robot_profile_id)
    if config.push_robot_scale_m != profile.runtime_facts["morphology"]["default_root_height_m"]:
        raise ValueError("Push robot scale must come from the selected profile's default root height")
    catalog_data = load_strict_json_object(
        _RECOVERY_ABI_PATH, context="Push recovery robot policy ABI",
        require_canonical=True, trailing_newline=True,
    )
    facts = dict(catalog_data["rows"][adapter.runtime_adapter_id])
    facts_hash = facts.pop("binding_facts_hash")
    if canonical_json_hash(facts) != facts_hash:
        raise ValueError("Push recovery robot policy ABI content differs from its hash")
    facts["binding_facts_hash"] = facts_hash
    physical = config.push_protocol.resolve(config.push_robot_scale_m)
    force = config.push_evaluation_protocol
    evaluation_projection = ({"push_evaluation_protocol": force.to_dict(),
        "push_resolved_evaluation_protocol": force.resolve(config.push_robot_scale_m),
        "evaluation_training_relation": "evaluation_only_successor_training_protocol_unchanged"}
        if force is not None else {})
    version = "v3" if force is not None else "v2"
    reward_contract = {
        "schema_version": "true_rl_agent.task.push_recovery.reward_contract.v2",
        "mode": "total_reward_program_replacement",
        "integration": "reward_manager_multiplies_rate_by_control_dt_once",
        "reference_program_must_be_nonempty": True,
    }
    binding = RobotTaskBinding(
        binding_id=config.cell_variant_id.replace(".", "_") + "_binding_" + version,
        cell_variant_id=config.cell_variant_id,
        task_id=PUSH_RECOVERY_TASK_ID,
        robot_profile_id=profile.robot_profile_id,
        runtime_adapter_id=adapter.runtime_adapter_id,
        role_bindings={"base": profile.semantic_roles["base"].targets},
        provided_capabilities=frozenset({
            "base_velocity_write", "joint_position_action", "policy_observation_projection",
            *({"external_body_force"} if force is not None else set()),
        }),
        task_projection={
            "push_abi_facts": facts,
            "push_protocol": config.push_protocol.to_dict(),
            "push_robot_scale_m": config.push_robot_scale_m,
            "push_resolved_protocol": physical,
            "reward_contract": reward_contract,
            **evaluation_projection,
            **({"reference_admission_policy": config.reference_admission_policy}
               if config.reference_admission_policy is not None else {}),
            **({"reference_program_id": config.reference_program_id}
               if config.reference_program_id is not None
               and config.reference_program_id != RECOVERY_REFERENCE_ID else {}),
        },
    )
    environment = CellEnvironmentDescriptor(
        environment_id="push_recovery_environment_" + version,
        cell_variant_id=config.cell_variant_id,
        task_id=PUSH_RECOVERY_TASK_ID,
        payload={
            "push_protocol_hash": config.push_protocol.protocol_hash,
            "push_resolved_protocol_hash": physical["resolved_protocol_hash"],
            "train_runtime_dynamics_hash": config.train_runtime_dynamics()["runtime_dynamics_hash"],
            "eval_runtime_dynamics_hash": config.eval_runtime_dynamics()["runtime_dynamics_hash"],
            **({"push_evaluation_protocol_hash": force.protocol_hash} if force is not None else {}),
        },
    )
    catalog = CellVariantCatalog()
    catalog.register(CellVariantDefinition(
        cell_variant_id=config.cell_variant_id,
        robot_task_cell_id="unitree_push_recovery." + config.cell_variant_id.split(".")[1] + "_" + version,
        task_id=PUSH_RECOVERY_TASK_ID,
        robot_profile_id=profile.robot_profile_id,
        runtime_adapter_id=adapter.runtime_adapter_id,
        robot_task_binding=binding,
        environment=environment,
        protocol_state=config.push_protocol.protocol_state,
        contract_hashes={
            "acceptance_protocol_hash": config.acceptance_protocol().protocol_hash,
            "action_content_hash": facts["action_content_hash"],
            "observation_content_hash": facts["observation_content_hash"],
            "agent_configuration_hash": rsl_rl_agent_configuration(adapter.runtime_adapter_id).agent_configuration_hash,
            "binding_facts_hash": facts_hash,
            "candidate_capability_profile_hash": plugin.task_spec.candidate_capability_profile.profile_hash,
            "evaluation_protocol_hash": config.evaluation_protocol().protocol_hash,
            "policy_io_schema_hash": facts["policy_io_schema_hash"],
            "reference_capability_gate_hash": canonical_json_hash(reference_qualification_contract(config)),
            "reward_input_schema_hash": canonical_json_hash({
                "schema_version": "true_rl_agent.push_reward_inputs.v2",
                "fields": list(PushRecoveryV2RewardInputs.field_names()),
            }),
            "reward_contract_hash": canonical_json_hash(reward_contract),
            "sensor_schema_hash": facts["sensor_schema_hash"],
            "push_protocol_hash": config.push_protocol.protocol_hash,
            "push_resolved_protocol_hash": physical["resolved_protocol_hash"],
            **({"push_evaluation_protocol_hash": force.protocol_hash,
                "push_resolved_evaluation_protocol_hash": evaluation_projection["push_resolved_evaluation_protocol"]["resolved_protocol_hash"]}
                if force is not None else {}),
        },
    ))
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
    "PUSH_G1_29DOF_REFERENCE_CELL_VARIANT_ID",
    "PUSH_G1_29DOF_ROBOT_TASK_CELL_ID",
    "PUSH_G1_37DOF_RECOVERY_CELL_VARIANT_ID",
    "PUSH_GO2_12DOF_RECOVERY_CELL_VARIANT_ID",
    "build_push_cell_catalog",
    "resolve_push_cell_manifest",
]
