"""Nonlegacy Push rows bind existing robot execution and concrete policy IO."""

from __future__ import annotations

import sys
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from true_rl_agent.composition.base import TaskCompositionContext
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.robots.defaults import build_default_robot_registry
from true_rl_agent.robots.rsl_rl_runtime import rsl_rl_agent_configuration
from true_rl_agent.tasks.push_recovery.composition import (
    PushRecoveryCompositionProvider,
)
from true_rl_agent.tasks.push_recovery.config import (
    PUSH_G1_37DOF_RECOVERY_CELL_VARIANT_ID,
    PUSH_GO2_12DOF_RECOVERY_CELL_VARIANT_ID,
    PUSH_G1_37DOF_FORCE_CELL_VARIANT_ID, PUSH_GO2_12DOF_FORCE_CELL_VARIANT_ID,
    PUSH_FORCE_EVAL_BASE_SEED,
    PushRecoveryTaskConfig,
)
from true_rl_agent.tasks.push_recovery.plugin import PushRecoveryTaskPlugin
from true_rl_agent.tasks.push_recovery.robot_binding import resolve_push_cell_manifest
from true_rl_agent.tasks.push_recovery.protocol import PushRecoveryProtocol
from true_rl_agent.tasks.push_recovery.reference_program import RECOVERY_REFERENCE_ID, STATIONARY_REFERENCE_ID, reference_program_id
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
from true_rl_agent.tasks.stair_climbing.robot_binding import resolve_stair_cell_manifest

ROWS = (
    (PUSH_G1_37DOF_FORCE_CELL_VARIANT_ID, "stair_unified_protocol_v2.json"),
    (PUSH_GO2_12DOF_FORCE_CELL_VARIANT_ID, "stair_go2_12dof_unified_protocol_v2.json"),
)
HISTORICAL_G1_TRAINING = "PLAN/2new_world/P6/history/push_g1_37dof_velocity_training_v3.json"


class PushRecoveryCellTests(unittest.TestCase):
    def test_recovery_rows_reuse_stair_robot_identity_with_concrete_io_and_no_legacy_grant(self):
        identities = []
        for cell_id, stair_path in ROWS:
            with self.subTest(cell=cell_id):
                config = PushRecoveryTaskConfig.for_cell(cell_id)
                cell = resolve_push_cell_manifest(cell_variant_id=cell_id, config=config,
                                                  plugin=PushRecoveryTaskPlugin(config))
                stair_config = StairClimbingTaskConfig.defaults(ROOT / "configs/tasks" / stair_path)
                stair = resolve_stair_cell_manifest(cell_variant_id=stair_config.cell_variant_id,
                    config=stair_config, plugin=StairClimbingTaskPlugin(stair_config))
                self.assertEqual(cell.robot_profile, stair.robot_profile)
                self.assertEqual(cell.runtime_adapter, stair.runtime_adapter)
                self.assertFalse(cell.uses_legacy_training_binding_v1)
                projection = cell.robot_task_binding["task_projection"]
                self.assertEqual(projection["push_resolved_protocol"], config.push_protocol.resolve(config.push_robot_scale_m))
                profile = build_default_robot_registry().profile(cell.robot_profile["robot_profile_id"])
                self.assertEqual(config.push_robot_scale_m, profile.runtime_facts["morphology"]["default_root_height_m"])
                facts = projection["push_abi_facts"]
                self.assertIsInstance(facts["actions"], list)
                self.assertEqual(facts["sensors"], {})
                self.assertEqual(facts["sensor_schema_hash"], canonical_json_hash({}))
                self.assertEqual(facts["joint_names"], list(profile.runtime_facts["joint_names"]))
                self.assertNotIn("height_scan", str(projection))
                self.assertNotIn("stair_runtime_projection", projection)
                self.assertEqual(cell.contract_hashes["action_content_hash"], canonical_json_hash({
                    "terms": facts["actions"], "joint_names": facts["joint_names"], "dimension": facts["action_dimension"],
                }))
                self.assertEqual(cell.contract_hashes["observation_content_hash"], canonical_json_hash({
                    "policy_observation": facts["policy_observation"], "joint_names": facts["joint_names"],
                    "term_dimensions": facts["policy_observation_term_dimensions"], "dimension": facts["policy_observation_dimension"],
                }))
                self.assertEqual(cell.contract_hashes["agent_configuration_hash"],
                    rsl_rl_agent_configuration(cell.runtime_adapter["runtime_adapter_id"]).agent_configuration_hash)
                identities.append((cell.contract_hashes["action_content_hash"], cell.contract_hashes["observation_content_hash"]))
        self.assertNotEqual(*identities)

    def test_new_defaults_materialize_shared_force_evaluation_and_frozen_training(self):
        provider = PushRecoveryCompositionProvider()
        self.assertEqual(provider.cli_defaults()["cell_variant_id"], PUSH_G1_37DOF_FORCE_CELL_VARIANT_ID)
        self.assertTrue(provider.cli_defaults()["reference_program_genesis"])
        for cell_id, _ in ROWS:
            with self.subTest(cell=cell_id), tempfile.TemporaryDirectory() as directory:
                composition = provider.resolve(values={**provider.cli_defaults(), "cell_variant_id": cell_id},
                    context=TaskCompositionContext(project_root=ROOT, output_dir=Path(directory)))
                config = composition.task_plugin.config
                cell = composition.resolved_cell_manifest
                self.assertEqual(composition.genesis_anchor_kind, "reference_program")
                self.assertEqual(composition.task_plugin.task_spec.candidate_capability_profile.profile_id, "push_complete_recovery_v2")
                binding = composition.backend_adapter._training_binding(
                    task_spec=composition.task_plugin.task_spec, policy_initialization=PolicyInitialization.random())
                self.assertEqual(binding.task_protocol_hash, cell.resolved_cell_task_protocol_hash)
                self.assertEqual(composition.backend_adapter.config.evaluation_binding.task_protocol_hash, cell.resolved_cell_task_protocol_hash)
                self.assertEqual(PushRecoveryTaskConfig.from_manifest(cell, eval_base_seed=config.eval_base_seed), config)
                self.assertEqual(config.budget_steps, 500 * 24 * 4096)
                self.assertEqual(config.train_seed, 42)
                self.assertEqual((config.public_seed_count, config.public_accept_seed_count, config.sealed_validation_seed_count, config.eval_num_envs), (3, 3, 7, 64))
                self.assertEqual(config.acceptance_protocol().absolute_accept_floor, 0.0)
                self.assertEqual(config.eval_base_seed, PUSH_FORCE_EVAL_BASE_SEED)
                self.assertIsNone(config.push_evaluation_protocol.resolve(config.push_robot_scale_m)["max_displacement_m"])
                if cell_id == PUSH_G1_37DOF_FORCE_CELL_VARIANT_ID:
                    self.assertEqual(config.push_protocol.to_dict(), json.loads((ROOT / HISTORICAL_G1_TRAINING).read_text()))
                    self.assertEqual(reference_program_id(config), STATIONARY_REFERENCE_ID)
                else:
                    self.assertEqual(config.push_protocol.high_velocity_scale, 1.0)
                    self.assertEqual(cell.robot_task_binding["task_projection"]["reference_program_id"], STATIONARY_REFERENCE_ID)

    def test_g1_default_matches_explicit_selection_and_original_protocol_remains_readable(self):
        provider = PushRecoveryCompositionProvider()
        selected_path = ROOT / HISTORICAL_G1_TRAINING
        self.assertEqual(selected_path.read_bytes(),
                         (ROOT / "PLAN/2new_world/P6/calibration_protocols_01/scale_1p5.json").read_bytes())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def compose(name, **overrides):
                return provider.resolve(values={**provider.cli_defaults(), **overrides},
                    context=TaskCompositionContext(project_root=ROOT, output_dir=root / name))
            default = compose("d")
            explicit = compose("e", eval_base_seed=PUSH_FORCE_EVAL_BASE_SEED, reference_program_id=STATIONARY_REFERENCE_ID,
                               push_protocol_path=selected_path)
            self.assertEqual(default.genesis_identity_snapshot()["scientific"], explicit.genesis_identity_snapshot()["scientific"])
            old = compose("o", cell_variant_id=PUSH_G1_37DOF_RECOVERY_CELL_VARIANT_ID,
                          push_evaluation_protocol=None, eval_base_seed=42, reference_program_id=RECOVERY_REFERENCE_ID,
                          push_protocol=PushRecoveryProtocol.defaults().to_dict())
            self.assertEqual(old.resolved_cell_manifest.manifest_hash,
                             "7fca96669153de0679c7cdd52911fecd407dc48e9f9c6e56adfb588a34061ace")
            decoded = PushRecoveryTaskConfig.from_manifest(old.resolved_cell_manifest)
            self.assertIsNone(decoded.reference_program_id)
            self.assertEqual(reference_program_id(decoded), RECOVERY_REFERENCE_ID)
            self.assertEqual(decoded.eval_base_seed, 42)
            self.assertEqual(decoded.push_protocol, PushRecoveryProtocol.defaults())

    def test_explicit_seed_reference_and_protocol_materialize_actual_jobs(self):
        provider = PushRecoveryCompositionProvider()
        protocol = replace(PushRecoveryProtocol.defaults(), high_velocity_scale=0.6)
        for cell_id, _ in ROWS:
            with self.subTest(cell=cell_id), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                protocol_path = root / "protocol.json"
                protocol_path.write_text(json.dumps(protocol.to_dict()), encoding="utf-8")
                values = {**provider.cli_defaults(), "cell_variant_id": cell_id,
                          "eval_base_seed": 1000000000, "reference_program_id": STATIONARY_REFERENCE_ID,
                          "push_protocol_path": protocol_path}
                composition = provider.resolve(values=values,
                    context=TaskCompositionContext(project_root=ROOT, output_dir=root / "a"))
                config = composition.task_plugin.config
                reference = composition.reference_program_artifact
                backend = composition.backend_adapter
                self.assertEqual(config.eval_base_seed, 1000000000)
                self.assertEqual(backend.config.eval_base_seed, config.eval_base_seed)
                self.assertEqual(config.push_protocol, protocol)
                self.assertEqual(composition.task_defaults["push_protocol"], protocol.to_dict())
                self.assertEqual(PushRecoveryTaskConfig.from_manifest(
                    composition.resolved_cell_manifest, eval_base_seed=config.eval_base_seed), config)
                self.assertEqual(composition.task_plugin.task_spec.fixed_train_protocol["reference_program_id"],
                                 reference.manifest.reference_id)
                self.assertEqual(reference.manifest.reference_id, STATIONARY_REFERENCE_ID)
                self.assertEqual(reference.manifest.authorship_origin, "llm_generated")
                self.assertEqual(reference.source_bytes, (ROOT / reference.manifest.upstream_source_path).read_bytes())
                self.assertEqual(reference.manifest.manifest_hash,
                                 "3ba402e052a8fce86a0f5a83a7ac5e39889a717011b083b602e1351aa04c3dc6")
                jobs = backend.materialize_reference_jobs(session_id="selected", task_spec=composition.task_plugin.task_spec,
                    reference=reference, artifact_root=root / "r", policy_initialization=PolicyInitialization.random())
                for spec in (jobs.train_spec, jobs.eval_spec):
                    self.assertEqual(spec.subject.subject_id, STATIONARY_REFERENCE_ID)
                    self.assertEqual(spec.program_source_checksum, reference.manifest.source_checksum)
                    self.assertEqual(spec.subject_manifest_hash, reference.manifest.manifest_hash)
                argv = list(jobs.eval_spec.metadata["cli_argv"])
                self.assertEqual(argv[argv.index("--seed") + 1], "1000000000")
                self.assertEqual(argv[argv.index("--seed-cohort-hash") + 1], config.seed_cohort_hash())
                self.assertIsNone(jobs.train_spec.training_start_checkpoint())
                train_argv = list(jobs.train_spec.metadata["cli_argv"])
                self.assertEqual(train_argv[train_argv.index("--seed") + 1], "42")
                # Loading identical bytes at a different path is placement, not a new task.
                other_path = root / "same.json"
                other_path.write_bytes(protocol_path.read_bytes())
                other = provider.resolve(values={**values, "push_protocol_path": other_path},
                    context=TaskCompositionContext(project_root=ROOT, output_dir=root / "b"))
                self.assertEqual(composition.resolved_cell_manifest.manifest_hash, other.resolved_cell_manifest.manifest_hash)

    def test_each_selected_condition_changes_scientific_identity(self):
        base = PushRecoveryTaskConfig.for_cell(PUSH_GO2_12DOF_RECOVERY_CELL_VARIANT_ID)
        original = PushRecoveryTaskPlugin(base).task_spec.resolved_task_protocol_hash()
        for selected in (replace(base, eval_base_seed=1000000000),
                         replace(base, reference_program_id=STATIONARY_REFERENCE_ID),
                         replace(base, push_protocol=replace(base.push_protocol, high_velocity_scale=0.6))):
            with self.subTest(selected=selected):
                self.assertNotEqual(PushRecoveryTaskPlugin(selected).task_spec.resolved_task_protocol_hash(), original)
        with self.assertRaisesRegex(ValueError, "reference_program_id"):
            PushRecoveryTaskConfig(reference_program_id=STATIONARY_REFERENCE_ID)

    def test_explicit_legacy_rows_keep_original_config_and_protocol(self):
        provider = PushRecoveryCompositionProvider()
        for cell_id, reference in (("push.g1_29dof.legacy_v1", False), ("push.g1_29dof.reference_v1", True)):
            with self.subTest(cell=cell_id), tempfile.TemporaryDirectory() as directory:
                composition = provider.resolve(values={**provider.cli_defaults(), "cell_variant_id": cell_id},
                    context=TaskCompositionContext(project_root=ROOT, output_dir=Path(directory)))
                self.assertEqual(composition.task_plugin.config, PushRecoveryTaskConfig(reference_program_genesis=reference))
                original = PushRecoveryTaskConfig(reference_program_genesis=reference)
                original_cell = resolve_push_cell_manifest(cell_variant_id=cell_id, config=original,
                    plugin=PushRecoveryTaskPlugin(original))
                self.assertEqual(composition.resolved_cell_manifest.manifest_hash, original_cell.manifest_hash)
                self.assertTrue(composition.resolved_cell_manifest.uses_legacy_training_binding_v1)
                self.assertEqual(composition.backend_adapter.config.observation_schema_hash, "push_recovery_obs_v1")
                self.assertEqual(composition.task_plugin.config.acceptance_protocol().absolute_accept_floor, 0.05)

    def test_protocol_changes_scientific_dynamics_and_robot_scale_cannot_be_relabelled(self):
        config = PushRecoveryTaskConfig.for_cell(PUSH_GO2_12DOF_RECOVERY_CELL_VARIANT_ID)
        with self.assertRaisesRegex(ValueError, "runtime_config_path is legacy-only"):
            replace(config, runtime_config_path="configs/push_recovery.yaml")
        changed = replace(config, push_protocol=replace(config.push_protocol, low_velocity_scale=0.16))
        self.assertNotEqual(config.train_runtime_dynamics()["runtime_dynamics_hash"], changed.train_runtime_dynamics()["runtime_dynamics_hash"])
        self.assertNotEqual(PushRecoveryTaskPlugin(config).task_spec.resolved_task_protocol_hash(),
                            PushRecoveryTaskPlugin(changed).task_spec.resolved_task_protocol_hash())
        wrong_scale = replace(config, push_robot_scale_m=config.push_robot_scale_m * 2)
        with self.assertRaisesRegex(ValueError, "default root height"):
            resolve_push_cell_manifest(cell_variant_id=wrong_scale.cell_variant_id, config=wrong_scale,
                                       plugin=PushRecoveryTaskPlugin(wrong_scale))


if __name__ == "__main__":
    unittest.main()
