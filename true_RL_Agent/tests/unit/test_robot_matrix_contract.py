from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.fakes.evaluation import make_verified_candidate_artifact
from true_rl_agent.backends.isaac_lab_rsl_rl import IsaacLabRslRlBackendAdapter
from true_rl_agent.composition.base import TaskCompositionContext
from true_rl_agent.composition.cell_manifest import (
    CellEnvironmentDescriptor,
    CellVariantCatalog,
    RobotTaskBinding,
    compile_cell_manifest,
    grant_legacy_training_binding_v1,
    load_optional_resolved_cell_manifest_from_args,
    load_resolved_cell_manifest,
    materialize_resolved_cell_manifest,
)
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.json_contract import load_strict_json_object
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.robots.base import RobotProfile, RobotRequirements, SemanticRole
from true_rl_agent.robots.defaults import (
    G1_29DOF_UNITREE_RL_LAB_ADAPTER_ID,
    G1_29DOF_UNITREE_RL_LAB_PROFILE_ID,
    G1_37DOF_ISAACLAB_PROFILE_ID,
    build_default_robot_registry,
)
from true_rl_agent.robots.registry import RobotRegistry
from true_rl_agent.robots.rsl_rl_runtime import rsl_rl_agent_configuration
from true_rl_agent.runner.specs import CheckpointRef
from true_rl_agent.runner.subprocess_runner import SubprocessRunner
from true_rl_agent.tasks.push_recovery.composition import (
    PushRecoveryCompositionProvider,
)
from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
from true_rl_agent.tasks.push_recovery.plugin import PushRecoveryTaskPlugin
from true_rl_agent.tasks.push_recovery.robot_binding import (
    PUSH_G1_29DOF_LEGACY_CELL_VARIANT_ID,
    build_push_cell_catalog,
    resolve_push_cell_manifest,
)
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
from true_rl_agent.tasks.stair_climbing.robot_binding import (
    STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID,
    build_stair_cell_catalog,
    resolve_stair_cell_manifest,
)
from true_rl_agent.tasks.stair_climbing.row_catalog import (
    STAIR_G1_37DOF_UNIFIED_V1_CELL_VARIANT_ID,
    STAIR_GO2_12DOF_UNIFIED_V1_CELL_VARIANT_ID,
    stair_robot_row,
)
from true_rl_agent.training_binding import (
    TRAINING_BINDING_V1_SCHEMA_VERSION,
    TRAINING_BINDING_V2_SCHEMA_VERSION,
    TrainingBinding,
)


class RobotMatrixContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.receipt = load_strict_json_object(
            PROJECT_ROOT
            / "PLAN/Variant_construction/P0/PHASE0_RUNTIME_PROBE_RECEIPT.json",
            context="Phase 0 runtime receipt",
        )
        cls.stair_config = StairClimbingTaskConfig.defaults(
            PROJECT_ROOT / "configs/tasks/stair_protocol_v2.json",
            cell_variant_id=STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID,
        )
        cls.stair_plugin = StairClimbingTaskPlugin(cls.stair_config)
        cls.push_config = PushRecoveryTaskConfig()
        cls.push_plugin = PushRecoveryTaskPlugin(cls.push_config)

    def test_two_profiles_recompute_complete_phase0_runtime_facts(self) -> None:
        registry = build_default_robot_registry()
        by_key = {
            item["profile_key"]: item for item in self.receipt["robot_runtime_facts"]
        }
        expected = {
            G1_37DOF_ISAACLAB_PROFILE_ID: "p0.g1_37_stair_execution",
            G1_29DOF_UNITREE_RL_LAB_PROFILE_ID: "p0.g1_29_push_execution",
        }
        for profile_id, receipt_key in expected.items():
            profile = registry.profile(profile_id)
            self.assertEqual(profile.runtime_facts, by_key[receipt_key])
            facts_without_hash = {
                key: value
                for key, value in profile.runtime_facts.items()
                if key != "runtime_facts_hash"
            }
            self.assertEqual(
                canonical_json_hash(facts_without_hash),
                profile.runtime_facts_hash,
            )
        g1_37 = registry.profile(G1_37DOF_ISAACLAB_PROFILE_ID)
        g1_29 = registry.profile(G1_29DOF_UNITREE_RL_LAB_PROFILE_ID)
        self.assertEqual(len(g1_37.runtime_facts["joint_names"]), 37)
        self.assertEqual(len(g1_29.runtime_facts["joint_names"]), 29)
        self.assertNotEqual(g1_37.profile_hash, g1_29.profile_hash)
        self.assertNotEqual(
            g1_37.runtime_facts["asset_closure"],
            g1_29.runtime_facts["asset_closure"],
        )
        self.assertNotEqual(
            registry.runtime_adapter_ids()[0],
            registry.runtime_adapter_ids()[1],
        )
        g1_29_adapter = registry.runtime_adapter(
            G1_29DOF_UNITREE_RL_LAB_ADAPTER_ID
        )
        self.assertEqual(
            g1_29_adapter.articulation_source,
            "unitree_rl_lab.assets.robots.unitree:UNITREE_G1_29DOF_CFG",
        )
        self.assertNotIn(".tasks.", g1_29_adapter.articulation_source)

    def test_archived_stair_training_predecessors_are_identity_frozen(self) -> None:
        for cell_variant_id in (
            STAIR_G1_37DOF_UNIFIED_V1_CELL_VARIANT_ID,
            STAIR_GO2_12DOF_UNIFIED_V1_CELL_VARIANT_ID,
        ):
            row = stair_robot_row(cell_variant_id)
            config = StairClimbingTaskConfig.defaults(
                PROJECT_ROOT / row.protocol_relative_path,
                cell_variant_id=cell_variant_id,
            )
            self.assertFalse(row.active)
            self.assertEqual(
                config.base_task_protocol_hash,
                row.frozen_task_protocol_hash,
            )
            self.assertEqual(
                row.load_binding(project_root=PROJECT_ROOT).binding_hash,
                row.frozen_binding_hash,
            )
            with self.assertRaisesRegex(ValueError, "predecessor protocol identity"):
                row.require_frozen_protocol_identity("0" * 64)

    def test_task_owned_bindings_preserve_phase0_projection(self) -> None:
        facts_by_cell = {
            item["cell_key"]: item for item in self.receipt["legacy_cell_binding_facts"]
        }
        stair_definition = build_stair_cell_catalog(
            config=self.stair_config,
            plugin=self.stair_plugin,
        ).resolve(STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID)
        push_definition = build_push_cell_catalog(
            config=self.push_config,
            plugin=self.push_plugin,
        ).resolve(PUSH_G1_29DOF_LEGACY_CELL_VARIANT_ID)
        for definition in (stair_definition, push_definition):
            facts = definition.robot_task_binding.task_projection[
                "phase0_binding_facts"
            ]
            self.assertEqual(facts, facts_by_cell[definition.cell_variant_id])
            self.assertEqual(
                canonical_json_hash(
                    {
                        key: value
                        for key, value in facts.items()
                        if key != "binding_facts_hash"
                    }
                ),
                facts["binding_facts_hash"],
            )
        self.assertEqual(
            tuple(
                requirement.role_id
                for requirement in self.push_plugin.robot_requirements().roles
            ),
            ("base",),
        )
        self.assertEqual(
            tuple(push_definition.robot_task_binding.role_bindings),
            ("base",),
        )
        self.assertNotIn(
            "support_contact",
            self.push_plugin.robot_requirements().capabilities,
        )
        self.assertEqual(
            stair_definition.robot_task_binding.role_bindings["base"],
            stair_definition.robot_task_binding.role_bindings[
                "forbidden_contact"
            ],
        )
        for path in (PROJECT_ROOT / "configs/robots").glob("unitree_g1_*v1.json"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn('"height_scanner"', text)
            self.assertNotIn('"policy_observation"', text)

    def test_push_agent_configuration_is_robot_row_owned(self) -> None:
        manifest = resolve_push_cell_manifest(
            cell_variant_id=PUSH_G1_29DOF_LEGACY_CELL_VARIANT_ID,
            config=self.push_config,
            plugin=self.push_plugin,
        )
        agent = rsl_rl_agent_configuration(
            manifest.runtime_adapter["runtime_adapter_id"]
        )
        self.assertEqual(
            manifest.contract_hashes["agent_configuration_hash"],
            agent.agent_configuration_hash,
        )
        self.assertTrue(
            {
                "policy_architecture_hash",
                "optimizer_training_profile_hash",
                "agent_configuration_hash",
            }.isdisjoint(self.push_plugin.task_spec.fixed_train_protocol)
        )

    def test_duplicate_capability_declarations_are_rejected_before_compilation(
        self,
    ) -> None:
        profile_path = PROJECT_ROOT / "configs/robots/unitree_g1_37dof_isaaclab_v1.json"
        profile_payload = load_strict_json_object(profile_path)
        profile_payload["provided_capabilities"].append(
            profile_payload["provided_capabilities"][0]
        )
        with self.assertRaisesRegex(ValueError, "contain duplicates"):
            RobotProfile.from_mapping(profile_payload)

        binding_path = (
            PROJECT_ROOT / "configs/tasks/stair_g1_37dof_clean_legacy_binding_v1.json"
        )
        binding_payload = load_strict_json_object(binding_path)
        binding_payload["provided_capabilities"].append(
            binding_payload["provided_capabilities"][0]
        )
        with self.assertRaisesRegex(ValueError, "contain duplicates"):
            RobotTaskBinding.from_mapping(binding_payload)

    def test_fresh_process_manifests_are_byte_identical(self) -> None:
        code = r"""
import sys
from pathlib import Path
from true_rl_agent.json_contract import canonical_json_bytes
if sys.argv[1] == "stair":
    from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
    from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
    from true_rl_agent.tasks.stair_climbing.robot_binding import STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID, resolve_stair_cell_manifest
    root=Path.cwd(); config=StairClimbingTaskConfig.defaults(root / "configs/tasks/stair_protocol_v2.json", cell_variant_id=STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID); plugin=StairClimbingTaskPlugin(config)
    manifest=resolve_stair_cell_manifest(cell_variant_id=STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID, config=config, plugin=plugin)
else:
    from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
    from true_rl_agent.tasks.push_recovery.plugin import PushRecoveryTaskPlugin
    from true_rl_agent.tasks.push_recovery.robot_binding import PUSH_G1_29DOF_LEGACY_CELL_VARIANT_ID, resolve_push_cell_manifest
    config=PushRecoveryTaskConfig(); plugin=PushRecoveryTaskPlugin(config)
    manifest=resolve_push_cell_manifest(cell_variant_id=PUSH_G1_29DOF_LEGACY_CELL_VARIANT_ID, config=config, plugin=plugin)
sys.stdout.buffer.write(canonical_json_bytes(manifest.to_dict(), trailing_newline=True))
"""
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(SRC_ROOT)
        for cell in ("stair", "push"):
            outputs = [
                subprocess.check_output(
                    [sys.executable, "-c", code, cell],
                    cwd=PROJECT_ROOT,
                    env=environment,
                )
                for _ in range(2)
            ]
            self.assertEqual(outputs[0], outputs[1])
            payload = json.loads(outputs[0])
            self.assertEqual(
                canonical_json_hash(
                    {
                        key: value
                        for key, value in payload.items()
                        if key
                        not in {
                            "manifest_hash",
                            "resolved_cell_task_protocol_hash",
                        }
                    }
                ),
                payload["manifest_hash"],
            )

    def test_each_composition_input_changes_effective_protocol(self) -> None:
        catalog = build_stair_cell_catalog(
            config=self.stair_config,
            plugin=self.stair_plugin,
        )
        definition = catalog.resolve(STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID)
        registry = build_default_robot_registry()
        requirements = self.stair_plugin.robot_requirements()
        base_task_protocol_hash = self.stair_config.task_protocol_hash

        def regrant(
            candidate,
            *,
            candidate_registry=registry,
            candidate_requirements=requirements,
            candidate_base_task_protocol_hash=base_task_protocol_hash,
        ):
            return grant_legacy_training_binding_v1(
                candidate,
                robot_registry=candidate_registry,
                requirements=candidate_requirements,
                base_task_protocol_hash=candidate_base_task_protocol_hash,
            )

        baseline = compile_cell_manifest(
            definition=definition,
            robot_registry=registry,
            requirements=requirements,
            base_task_protocol_hash=base_task_protocol_hash,
        ).require_supported()

        changed_task_definition = regrant(
            definition,
            candidate_base_task_protocol_hash="0" * 64,
        )
        changed_task = compile_cell_manifest(
            definition=changed_task_definition,
            robot_registry=registry,
            requirements=requirements,
            base_task_protocol_hash="0" * 64,
        ).require_supported()
        changed_environment = replace(
            definition,
            environment=CellEnvironmentDescriptor(
                environment_id=definition.environment.environment_id,
                cell_variant_id=definition.cell_variant_id,
                task_id=definition.task_id,
                payload={
                    **definition.environment.payload,
                    "variant_revision": "changed",
                },
            ),
        )
        changed_binding_value = replace(
            definition.robot_task_binding,
            task_projection={
                **definition.robot_task_binding.task_projection,
                "projection_revision": "changed",
            },
            binding_hash=None,
        )
        changed_binding = replace(
            definition,
            robot_task_binding=changed_binding_value,
        )
        changed_schema = replace(
            definition,
            contract_hashes={
                **definition.contract_hashes,
                "observation_content_hash": "1" * 64,
            },
        )

        original_profile = registry.profile(G1_37DOF_ISAACLAB_PROFILE_ID)
        changed_profile = replace(
            original_profile,
            display_name=original_profile.display_name + " changed",
            profile_hash=None,
        )
        changed_registry = RobotRegistry()
        changed_registry.register_profile(changed_profile)
        changed_registry.register_runtime_adapter(
            registry.runtime_adapter(definition.runtime_adapter_id)
        )

        variants = [
            changed_task,
            compile_cell_manifest(
                definition=regrant(changed_environment),
                robot_registry=registry,
                requirements=requirements,
                base_task_protocol_hash=base_task_protocol_hash,
            ).require_supported(),
            compile_cell_manifest(
                definition=regrant(changed_binding),
                robot_registry=registry,
                requirements=requirements,
                base_task_protocol_hash=base_task_protocol_hash,
            ).require_supported(),
            compile_cell_manifest(
                definition=regrant(changed_schema),
                robot_registry=registry,
                requirements=requirements,
                base_task_protocol_hash=base_task_protocol_hash,
            ).require_supported(),
            compile_cell_manifest(
                definition=regrant(
                    definition,
                    candidate_registry=changed_registry,
                ),
                robot_registry=changed_registry,
                requirements=requirements,
                base_task_protocol_hash=base_task_protocol_hash,
            ).require_supported(),
        ]
        for variant in variants:
            self.assertNotEqual(variant.manifest_hash, baseline.manifest_hash)
            self.assertNotEqual(
                variant.resolved_cell_task_protocol_hash,
                baseline.resolved_cell_task_protocol_hash,
            )

    def test_compiler_returns_deterministic_unsupported_reasons(self) -> None:
        definition = build_stair_cell_catalog(
            config=self.stair_config,
            plugin=self.stair_plugin,
        ).resolve(STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID)
        registry = build_default_robot_registry()
        swap = grant_legacy_training_binding_v1(
            replace(
                definition,
                robot_profile_id=G1_29DOF_UNITREE_RL_LAB_PROFILE_ID,
                runtime_adapter_id=G1_29DOF_UNITREE_RL_LAB_ADAPTER_ID,
            ),
            robot_registry=registry,
            requirements=self.stair_plugin.robot_requirements(),
            base_task_protocol_hash=self.stair_config.task_protocol_hash,
        )
        swapped = compile_cell_manifest(
            definition=swap,
            robot_registry=registry,
            requirements=self.stair_plugin.robot_requirements(),
            base_task_protocol_hash=self.stair_config.task_protocol_hash,
        )
        self.assertEqual(swapped.status, "unsupported")
        self.assertEqual(
            swapped.reasons,
            (
                "adapter_binding_mismatch",
                "missing_profile_role:forbidden_contact",
                "profile_binding_mismatch",
            ),
        )

        profile = registry.profile(G1_37DOF_ISAACLAB_PROFILE_ID)
        missing_role_profile = replace(
            profile,
            semantic_roles={
                key: role
                for key, role in profile.semantic_roles.items()
                if key != "support_effector"
            },
            profile_hash=None,
        )
        missing_registry = RobotRegistry()
        missing_registry.register_profile(missing_role_profile)
        missing_registry.register_runtime_adapter(
            registry.runtime_adapter(definition.runtime_adapter_id)
        )
        missing_role_definition = grant_legacy_training_binding_v1(
            definition,
            robot_registry=missing_registry,
            requirements=self.stair_plugin.robot_requirements(),
            base_task_protocol_hash=self.stair_config.task_protocol_hash,
        )
        missing_role = compile_cell_manifest(
            definition=missing_role_definition,
            robot_registry=missing_registry,
            requirements=self.stair_plugin.robot_requirements(),
            base_task_protocol_hash=self.stair_config.task_protocol_hash,
        )
        self.assertIn("missing_profile_role:support_effector", missing_role.reasons)

        missing_capability_requirements = RobotRequirements(
            roles=self.stair_plugin.robot_requirements().roles,
            capabilities=(
                self.stair_plugin.robot_requirements().capabilities
                | {"unavailable_sensor"}
            ),
        )
        missing_capability_definition = grant_legacy_training_binding_v1(
            definition,
            robot_registry=registry,
            requirements=missing_capability_requirements,
            base_task_protocol_hash=self.stair_config.task_protocol_hash,
        )
        missing_capability = compile_cell_manifest(
            definition=missing_capability_definition,
            robot_registry=registry,
            requirements=missing_capability_requirements,
            base_task_protocol_hash=self.stair_config.task_protocol_hash,
        )
        self.assertEqual(
            missing_capability.reasons,
            ("missing_capability:unavailable_sensor",),
        )

        baseline_manifest = compile_cell_manifest(
            definition=definition,
            robot_registry=registry,
            requirements=self.stair_plugin.robot_requirements(),
            base_task_protocol_hash=self.stair_config.task_protocol_hash,
        ).require_supported()
        self.assertEqual(
            baseline_manifest.resolved_roles["base"],
            baseline_manifest.resolved_roles["forbidden_contact"],
        )

        conflicting_roles = dict(profile.semantic_roles)
        conflicting_roles["support_effector"] = SemanticRole(
            "support_effector",
            "body",
            ("torso_link", "left_ankle_roll_link"),
        )
        conflicting_profile = replace(
            profile,
            semantic_roles=conflicting_roles,
            profile_hash=None,
        )
        conflicting_binding = replace(
            definition.robot_task_binding,
            role_bindings={
                **definition.robot_task_binding.role_bindings,
                "support_effector": (
                    "torso_link",
                    "left_ankle_roll_link",
                ),
            },
            binding_hash=None,
        )
        conflicting_registry = RobotRegistry()
        conflicting_registry.register_profile(conflicting_profile)
        conflicting_registry.register_runtime_adapter(
            registry.runtime_adapter(definition.runtime_adapter_id)
        )
        conflicting = compile_cell_manifest(
            definition=grant_legacy_training_binding_v1(
                replace(
                    definition,
                    robot_task_binding=conflicting_binding,
                ),
                robot_registry=conflicting_registry,
                requirements=self.stair_plugin.robot_requirements(),
                base_task_protocol_hash=self.stair_config.task_protocol_hash,
            ),
            robot_registry=conflicting_registry,
            requirements=self.stair_plugin.robot_requirements(),
            base_task_protocol_hash=self.stair_config.task_protocol_hash,
        )
        self.assertIn(
            "conflicting_role_target:forbidden_contact:support_effector:torso_link",
            conflicting.reasons,
        )

        duplicate_catalog = CellVariantCatalog()
        duplicate_catalog.register(definition)
        with self.assertRaisesRegex(ValueError, "already registered"):
            duplicate_catalog.register(definition)

    def test_legacy_training_binding_v1_grant_is_content_bound(self) -> None:
        definition = build_stair_cell_catalog(
            config=self.stair_config,
            plugin=self.stair_plugin,
        ).resolve(STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID)
        self.assertIsNotNone(
            definition.legacy_training_binding_v1_grant_hash
        )
        changed_environment = replace(
            definition,
            environment=CellEnvironmentDescriptor(
                environment_id=definition.environment.environment_id,
                cell_variant_id=definition.cell_variant_id,
                task_id=definition.task_id,
                payload={
                    **definition.environment.payload,
                    "variant_revision": "new-cell",
                },
            ),
        )
        stale_grant = compile_cell_manifest(
            definition=changed_environment,
            robot_registry=build_default_robot_registry(),
            requirements=self.stair_plugin.robot_requirements(),
            base_task_protocol_hash=self.stair_config.task_protocol_hash,
        )
        self.assertEqual(
            stale_grant.reasons,
            ("legacy_training_binding_v1_grant_mismatch",),
        )

        future_cell = compile_cell_manifest(
            definition=replace(
                changed_environment,
                legacy_training_binding_v1_grant_hash=None,
            ),
            robot_registry=build_default_robot_registry(),
            requirements=self.stair_plugin.robot_requirements(),
            base_task_protocol_hash=self.stair_config.task_protocol_hash,
        ).require_supported()
        self.assertFalse(future_cell.uses_legacy_training_binding_v1)
        self.assertNotIn(
            "legacy_training_binding_v1",
            future_cell.to_dict(),
        )

    def test_manifest_reload_rejects_hash_and_task_tamper(self) -> None:
        manifest = resolve_stair_cell_manifest(
            cell_variant_id=STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID,
            config=self.stair_config,
            plugin=self.stair_plugin,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = materialize_resolved_cell_manifest(
                manifest,
                output_dir=Path(tmp),
            )
            loaded = load_resolved_cell_manifest(
                path,
                expected_manifest_hash=str(manifest.manifest_hash),
                expected_task_id=self.stair_plugin.task_spec.task_id,
                expected_base_task_protocol_hash=self.stair_config.task_protocol_hash,
            )
            self.assertEqual(loaded.to_dict(), manifest.to_dict())
            wrong_grant = manifest.to_dict()
            wrong_grant["legacy_training_binding_v1_grant_hash"] = "0" * 64
            with self.assertRaisesRegex(
                ValueError,
                "legacy TrainingBindingV1 grant hash mismatch",
            ):
                type(manifest).from_mapping(wrong_grant)
            with self.assertRaisesRegex(ValueError, "argv hash mismatch"):
                load_resolved_cell_manifest(
                    path,
                    expected_manifest_hash="f" * 64,
                )
            with self.assertRaisesRegex(ValueError, "task_id mismatch"):
                load_resolved_cell_manifest(
                    path,
                    expected_task_id="wrong_task",
                )
            with self.assertRaisesRegex(ValueError, "provided together"):
                load_optional_resolved_cell_manifest_from_args(
                    argparse.Namespace(
                        resolved_cell_manifest_path=None,
                        resolved_cell_manifest_hash="0" * 64,
                    ),
                    expected_task_id=self.stair_plugin.task_spec.task_id,
                    expected_base_task_protocol_hash=self.stair_config.task_protocol_hash,
                )

    def test_training_binding_v2_adds_task_protocol_without_rewriting_v1(self) -> None:
        fields = {
            "initialization_mode": "random",
            "policy_architecture_hash": "1" * 64,
            "agent_configuration_hash": "2" * 64,
            "initialization_seed_schedule_hash": "3" * 64,
            "optimizer_training_profile_hash": "4" * 64,
            "budget_steps": 128,
            "private_training_seed_resolver_hash": "5" * 64,
            "capability_profile_hash": "6" * 64,
        }
        legacy = TrainingBinding(**fields)
        self.assertEqual(legacy.schema_version, TRAINING_BINDING_V1_SCHEMA_VERSION)
        self.assertNotIn("task_protocol_hash", legacy.to_private_dict())
        self.assertEqual(
            TrainingBinding.from_mapping(legacy.to_private_dict()),
            legacy,
        )

        task_hash = "7" * 64
        v2 = TrainingBinding(**fields, task_protocol_hash=task_hash)
        self.assertEqual(v2.schema_version, TRAINING_BINDING_V2_SCHEMA_VERSION)
        self.assertNotEqual(v2.binding_hash, legacy.binding_hash)
        self.assertEqual(v2.to_private_dict()["task_protocol_hash"], task_hash)
        self.assertEqual(TrainingBinding.from_mapping(v2.to_private_dict()), v2)
        invalid_v2 = {**v2.to_private_dict(), "task_protocol_hash": None}
        with self.assertRaisesRegex(ValueError, "V2 task_protocol_hash"):
            TrainingBinding.from_mapping(invalid_v2)
        v2.require_task_protocol(task_hash)
        with self.assertRaisesRegex(ValueError, "task protocol mismatch"):
            v2.require_task_protocol("8" * 64)
        with self.assertRaisesRegex(ValueError, "V2"):
            legacy.require_task_protocol(task_hash)

    def test_source_ownership_boundaries_are_small_and_task_neutral(self) -> None:
        robots_root = SRC_ROOT / "true_rl_agent/robots"
        for path in robots_root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.append(node.module)
            self.assertFalse(
                any(name.startswith("true_rl_agent.tasks") for name in imported),
                path,
            )

        compiler_source = (
            (SRC_ROOT / "true_rl_agent/composition/cell_manifest.py")
            .read_text(encoding="utf-8")
            .lower()
        )
        for marker in ("phase0_binding_facts", "stair", "push_recovery", "roughness"):
            self.assertNotIn(marker, compiler_source)

        concrete_markers = (
            "torso_link",
            "ankle_roll_link",
            ".usd",
            "isaaclab_assets",
            "unitree_rl_lab",
        )
        for path in (
            SRC_ROOT / "true_rl_agent/tasks/stair_climbing/plugin.py",
            SRC_ROOT / "true_rl_agent/tasks/push_recovery/plugin.py",
        ):
            source = path.read_text(encoding="utf-8").lower()
            for marker in concrete_markers:
                self.assertNotIn(marker, source, path)

        forbidden_keys = {
            "budget_steps",
            "train_seed",
            "eval_seed",
            "seed_cohort_hash",
            "acceptance_protocol",
            "acceptance_protocol_hash",
        }
        for path in (PROJECT_ROOT / "configs/tasks").glob("*g1*legacy*binding_v1.json"):
            payload = load_strict_json_object(path)
            observed_keys = _recursive_keys(payload)
            self.assertFalse(forbidden_keys & observed_keys, path)

    def test_compositions_materialize_absolute_manifest_identity(self) -> None:
        provider = PushRecoveryCompositionProvider()
        with tempfile.TemporaryDirectory() as tmp:
            composition = provider.resolve(
                values=dict(provider.cli_defaults()),
                context=TaskCompositionContext(
                    project_root=PROJECT_ROOT,
                    output_dir=Path(tmp),
                ),
            )
            config = composition.backend_adapter.config
            self.assertTrue(Path(config.resolved_cell_manifest_path).is_absolute())
            self.assertEqual(
                config.resolved_cell_manifest_hash,
                composition.resolved_cell_manifest.manifest_hash,
            )
            snapshot = composition.genesis_identity_snapshot()
            self.assertEqual(
                snapshot["schema_version"],
                "true_rl_agent.runtime_composition_snapshot.v2",
            )
            self.assertEqual(
                snapshot["resolved_cell"]["manifest_hash"],
                config.resolved_cell_manifest_hash,
            )
            candidate = make_verified_candidate_artifact(
                "p1-matrix-candidate",
                admission_policy_descriptor=(
                    composition.candidate_admission_policy.admission_policy_descriptor()
                ),
            )
            jobs = composition.backend_adapter.materialize_jobs(
                round_id="p1-matrix-job",
                task_spec=composition.task_plugin.task_spec,
                candidate=candidate,
                policy_initialization=PolicyInitialization.random(),
            )
            self.assertEqual(
                jobs.train_spec.training_binding.schema_version,
                TRAINING_BINDING_V1_SCHEMA_VERSION,
            )
            for job in (jobs.train_spec, jobs.eval_spec):
                argv = job.metadata["cli_argv"]
                self.assertEqual(
                    argv[argv.index("--resolved-cell-manifest-path") + 1],
                    config.resolved_cell_manifest_path,
                )
                self.assertEqual(
                    argv[argv.index("--resolved-cell-manifest-hash") + 1],
                    config.resolved_cell_manifest_hash,
                )
                if job is jobs.train_spec:
                    self.assertEqual(argv.count("--base-task-protocol-hash"), 1)
                    self.assertEqual(argv.count("--task-protocol-hash"), 0)
                else:
                    self.assertEqual(argv.count("--base-task-protocol-hash"), 1)
                    self.assertEqual(argv.count("--task-protocol-hash"), 1)

            future_manifest = replace(
                composition.resolved_cell_manifest,
                legacy_training_binding_v1_grant_hash=None,
                manifest_hash=None,
                resolved_cell_task_protocol_hash=None,
            )
            future_path = materialize_resolved_cell_manifest(
                future_manifest,
                output_dir=Path(tmp) / "future-cell",
            )
            future_backend = IsaacLabRslRlBackendAdapter(
                replace(
                    config,
                    resolved_cell_manifest_path=str(future_path),
                    resolved_cell_manifest_hash=future_manifest.manifest_hash,
                )
            )
            future_binding = future_backend._training_binding(
                task_spec=composition.task_plugin.task_spec,
                policy_initialization=PolicyInitialization.random(),
            )
            self.assertEqual(
                future_binding.schema_version,
                TRAINING_BINDING_V2_SCHEMA_VERSION,
            )
            self.assertEqual(
                future_binding.task_protocol_hash,
                future_manifest.resolved_cell_task_protocol_hash,
            )

    def test_runner_rejects_wrong_child_and_checkpoint_cell_identity(self) -> None:
        manifest = resolve_push_cell_manifest(
            cell_variant_id=PUSH_G1_29DOF_LEGACY_CELL_VARIANT_ID,
            config=self.push_config,
            plugin=self.push_plugin,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = materialize_resolved_cell_manifest(
                manifest,
                output_dir=Path(tmp),
            )
            argv = [
                sys.executable,
                "child.py",
                "--resolved-cell-manifest-path",
                str(path),
                "--resolved-cell-manifest-hash",
                str(manifest.manifest_hash),
            ]
            summary = {
                "resolved_cell_manifest_hash": manifest.manifest_hash,
                "resolved_cell_identity": manifest.summary_identity(),
            }
            SubprocessRunner._validate_resolved_cell_summary(
                summary,
                argv=argv,
                job_id="p1-child",
            )
            with self.assertRaisesRegex(ValueError, "summary resolved cell"):
                SubprocessRunner._validate_resolved_cell_summary(
                    {
                        **summary,
                        "resolved_cell_manifest_hash": "0" * 64,
                    },
                    argv=argv,
                    job_id="p1-child",
                )

            checkpoint_path = Path(tmp) / "checkpoint.pt"
            checkpoint_path.write_bytes(b"p1-checkpoint")
            checkpoint_hash = (
                "sha256:" + hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            )
            checkpoint = CheckpointRef(
                uri=str(checkpoint_path),
                checksum=checkpoint_hash,
                role="eval",
                sealed=True,
                metadata={
                    "local_path": str(checkpoint_path),
                    "resolved_cell_manifest_hash": "0" * 64,
                },
            )
            runner = SubprocessRunner(
                allowed_executables=(sys.executable,),
                cwd=PROJECT_ROOT,
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "checkpoint resolved cell manifest hash mismatch",
            ):
                runner._validate_checkpoint_before_launch(
                    checkpoint=checkpoint,
                    metadata={"work_dir": str(PROJECT_ROOT)},
                    job_id="p1-eval",
                    argv=[
                        sys.executable,
                        "child.py",
                        "--checkpoint",
                        str(checkpoint_path),
                        "--checkpoint-sha256",
                        checkpoint_hash,
                        "--resolved-cell-manifest-path",
                        str(path),
                        "--resolved-cell-manifest-hash",
                        str(manifest.manifest_hash),
                    ],
                    checkpoint_path_option="--checkpoint",
                    checkpoint_checksum_option="--checkpoint-sha256",
                )
            runner._validate_checkpoint_before_launch(
                checkpoint=checkpoint,
                metadata={
                    "work_dir": str(PROJECT_ROOT),
                    "training_predecessor_manifest_hash": "0" * 64,
                },
                job_id="p1-evaluation-successor",
                argv=[
                    sys.executable,
                    "child.py",
                    "--checkpoint",
                    str(checkpoint_path),
                    "--checkpoint-sha256",
                    checkpoint_hash,
                    "--resolved-cell-manifest-path",
                    str(path),
                    "--resolved-cell-manifest-hash",
                    str(manifest.manifest_hash),
                ],
                checkpoint_path_option="--checkpoint",
                checkpoint_checksum_option="--checkpoint-sha256",
            )

            future_manifest = replace(
                manifest,
                legacy_training_binding_v1_grant_hash=None,
                manifest_hash=None,
                resolved_cell_task_protocol_hash=None,
            )
            future_path = materialize_resolved_cell_manifest(
                future_manifest,
                output_dir=Path(tmp) / "future-cell",
            )
            unbound_checkpoint = replace(
                checkpoint, metadata={"local_path": str(checkpoint_path)}
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "resolved cell manifest hash is required",
            ):
                runner._validate_checkpoint_before_launch(
                    checkpoint=unbound_checkpoint,
                    metadata={"work_dir": str(PROJECT_ROOT)},
                    job_id="p1-future-eval",
                    argv=[
                        sys.executable,
                        "child.py",
                        "--checkpoint",
                        str(checkpoint_path),
                        "--checkpoint-sha256",
                        checkpoint_hash,
                        "--resolved-cell-manifest-path",
                        str(future_path),
                        "--resolved-cell-manifest-hash",
                        str(future_manifest.manifest_hash),
                    ],
                    checkpoint_path_option="--checkpoint",
                    checkpoint_checksum_option="--checkpoint-sha256",
                )

    def test_children_reload_manifest_before_app_creation(self) -> None:
        scripts_and_launch_markers = {
            "scripts/isaaclab_stair_rsl_rl_train.py": "AppLauncher(args",
            "scripts/isaaclab_stair_rsl_rl_eval.py": "app_launcher_class(args",
            "scripts/isaaclab_rsl_rl_train.py": "app_launcher = AppLauncher(args_cli)",
            "scripts/isaaclab_rsl_rl_eval.py": "app_launcher = AppLauncher(args_cli)",
        }
        for relative, launch_marker in scripts_and_launch_markers.items():
            source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
            reload_index = source.find(
                "load_optional_resolved_cell_manifest_from_args("
            )
            launch_index = source.find(launch_marker)
            self.assertGreaterEqual(reload_index, 0, relative)
            self.assertGreater(launch_index, reload_index, relative)
            if relative.endswith("train.py"):
                self.assertIn("resolved_cell_manifest_hash is not None", source)


def _recursive_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        keys.update(str(key) for key in value)
        for item in value.values():
            keys.update(_recursive_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_recursive_keys(item))
    return keys


if __name__ == "__main__":
    unittest.main()
