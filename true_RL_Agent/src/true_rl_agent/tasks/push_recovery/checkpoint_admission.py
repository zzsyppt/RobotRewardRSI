"""Task-owned admission of a completed v2 reference into v3 evaluation only.

The full cell hash includes evaluation rules, so a v2 training binding must be
preserved, not relabeled as a new v3 training run. Current-cell jobs keep their
existing path; cross-version jobs must carry the original manifest and receipt.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from true_rl_agent.composition.cell_manifest import load_resolved_cell_manifest
from true_rl_agent.runner.argv_contract import unique_option_value
from true_rl_agent.runner.checkpoint import file_sha256
from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
from true_rl_agent.tasks.push_recovery.plugin import PushRecoveryTaskPlugin
from true_rl_agent.tasks.push_recovery.robot_binding import resolve_push_cell_manifest
from true_rl_agent.robots.rsl_rl_runtime import rsl_rl_agent_configuration


def validate_push_reference_evaluation_source(source: Any, current: Any, trained: Mapping[str, Any]) -> str:
    """Verify the training facts and retain the source's original binding hash."""
    source_config, current_config = PushRecoveryTaskConfig.from_manifest(source), PushRecoveryTaskConfig.from_manifest(current)
    if source_config.push_protocol is None or current_config.push_protocol is None:
        raise ValueError("Push reference replay requires physical recovery cells")
    hash_fields = {"manifest_hash", "base_task_protocol_hash", "resolved_cell_task_protocol_hash"}
    same_cell_definitions = source.cell_variant_id == current.cell_variant_id and {
        key: value for key, value in source.to_dict().items() if key not in hash_fields
    } == {key: value for key, value in current.to_dict().items() if key not in hash_fields}
    if not same_cell_definitions and not (
        source.cell_variant_id.endswith(".recovery_v2") and current.cell_variant_id == source.cell_variant_id[:-2] + "v3"
    ):
        raise ValueError("Push checkpoint successor admission only permits v2 to v3 of the same robot")
    for manifest, config in ((source, source_config), (current, current_config)):
        config = replace(config, resolved_cell_task_protocol_hash=None)
        rebuilt = resolve_push_cell_manifest(cell_variant_id=config.cell_variant_id, config=config, plugin=PushRecoveryTaskPlugin(config))
        # A cell snapshot does not contain the whole historical TaskSpec (notably
        # its eval seed cohort). Preserve its authenticated base hash; recompile
        # and compare every stored physical/ABI/task-projection field instead.
        if {key: value for key, value in rebuilt.to_dict().items() if key not in hash_fields} != {
            key: value for key, value in manifest.to_dict().items() if key not in hash_fields
        }:
            raise ValueError("Push replay manifest definitions differ from the canonical task resolver")
    for name in ("robot_profile", "runtime_adapter", "resolved_roles"):
        if getattr(source, name) != getattr(current, name):
            raise ValueError("Push evaluation-only successor changed " + name)
    for name in ("action_content_hash", "observation_content_hash", "policy_io_schema_hash", "agent_configuration_hash",
        "sensor_schema_hash", "reward_input_schema_hash", "reward_contract_hash", "candidate_capability_profile_hash"):
        if source.contract_hashes[name] != current.contract_hashes[name]:
            raise ValueError("Push evaluation-only successor changed " + name)
    if source_config.push_protocol != current_config.push_protocol or source_config.train_runtime_dynamics() != current_config.train_runtime_dynamics():
        raise ValueError("Push evaluation-only successor changed training dynamics")
    if (trained.get("status") != "pass" or trained.get("subject_kind") != "reference"
        or trained.get("task_id") != "unitree_push_recovery" or trained.get("resolved_cell_manifest_hash") != source.manifest_hash):
        raise ValueError("Push successor requires its original completed reference receipt")
    agent = rsl_rl_agent_configuration(source_config.runtime_adapter_id)
    binding = agent.training_binding(initialization_mode="random", train_seed=source_config.train_seed,
        budget_steps=source_config.budget_steps,
        capability_profile_hash=PushRecoveryTaskPlugin(source_config).task_spec.candidate_capability_profile.profile_hash,
        task_protocol_hash=source.resolved_cell_task_protocol_hash)
    if trained.get("training_binding_hash") != binding.binding_hash:
        raise ValueError("Push source training binding cannot be reconstructed from its original contract")
    for key in ("policy_architecture_hash", "agent_configuration_hash", "optimizer_training_profile_hash",
        "initialization_seed_schedule_hash", "capability_profile_hash"):
        if trained.get(key) != getattr(binding, key):
            raise ValueError("Push source training receipt differs on " + key)
    budget = trained["training"]["budget_accounting"]
    if budget.get("status") != "pass" or any(budget.get(key) != value for key, value in {
        "budget_steps": source_config.budget_steps, "planned_env_steps": source_config.budget_steps,
        "config_derived_total_timesteps": source_config.budget_steps,
        "num_envs": source_config.train_num_envs, "num_steps_per_env": source_config.train_steps_per_iteration,
        "max_iterations": source_config.train_max_iterations}.items()) or trained["training"].get("latest_checkpoint_iteration") != source_config.train_max_iterations - 1:
        raise ValueError("Push source reference did not complete the declared training budget")
    return str(binding.binding_hash)


def admit_push_evaluation_checkpoint(checkpoint: Any, current_manifest: Any, argv: tuple[str, ...]) -> None:
    """Existing runner admission hook; no shared or artifact-supplied hash allowlist."""
    if checkpoint.role != "eval":
        raise ValueError("Push successor admission is evaluation-only")
    source_hash = checkpoint.metadata.get("resolved_cell_manifest_hash")
    if source_hash == current_manifest.manifest_hash and "source_training_summary_path" not in checkpoint.metadata:
        return
    required = ("source_resolved_manifest_path", "source_training_summary_path", "source_training_summary_sha256")
    if any(not checkpoint.metadata.get(key) for key in required):
        raise ValueError("Push successor checkpoint requires its source manifest and training receipt")
    source = load_resolved_cell_manifest(Path(checkpoint.metadata["source_resolved_manifest_path"]), expected_manifest_hash=source_hash)
    summary_path = Path(checkpoint.metadata["source_training_summary_path"])
    if file_sha256(summary_path) != checkpoint.metadata["source_training_summary_sha256"]:
        raise ValueError("Push source training receipt bytes changed")
    trained = json.loads(summary_path.read_text(encoding="utf-8"))
    binding = validate_push_reference_evaluation_source(source, current_manifest, trained)
    if trained["checkpoint"]["checksum"] != checkpoint.checksum:
        raise ValueError("Push checkpoint differs from the source training receipt")
    for flag, value in (("--training-binding-hash", binding), ("--program-source-checksum", trained["program_source_checksum"]),
        ("--subject-kind", trained["subject_kind"]), ("--subject-id", trained["subject_id"]),
        ("--subject-manifest-hash", trained["subject_manifest_hash"])):
        if unique_option_value(argv, flag) != value:
            raise ValueError("Push successor evaluation relabeled the source " + flag)
