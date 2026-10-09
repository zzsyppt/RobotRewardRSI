from __future__ import annotations

import ast
import copy
import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from true_rl_agent.evaluation.binding import canonical_json_hash  # noqa: E402
from true_rl_agent.robots.base import (  # noqa: E402
    SUPPORT_EFFECTOR_ROLE,
    SemanticRole,
)
from true_rl_agent.robots.defaults import (  # noqa: E402
    G1_37DOF_ISAACLAB_GROUPED_ADAPTER_ID,
    G1_37DOF_ISAACLAB_GROUPED_PROFILE_ID,
    G1_37DOF_ISAACLAB_PROFILE_ID,
    build_default_robot_registry,
)
from true_rl_agent.robots.registry import RobotRegistry  # noqa: E402
from true_rl_agent.tasks.stair_climbing.plugin import (  # noqa: E402
    StairClimbingTaskPlugin,
)
from true_rl_agent.tasks.stair_climbing.robot_binding import (  # noqa: E402
    STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID,
    build_stair_cell_catalog,
    compile_stair_cell_manifest,
    resolve_stair_cell_manifest,
)
from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.row_catalog import (  # noqa: E402
    STAIR_G1_37DOF_UNIFIED_V2_CELL_VARIANT_ID,
)
from true_rl_agent.tasks.stair_climbing.runtime_projection import (  # noqa: E402
    STAIR_KINEMATIC_BASE_ARTICULATION_ROOT,
    STAIR_KINEMATIC_BASE_BODY_ROLE,
    resolve_stair_runtime_projection,
    stair_runtime_projection_from_execution,
    stair_runtime_projection_from_manifest,
)


class StairMorphologyProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = StairClimbingTaskConfig.defaults(
            ROOT / "configs/tasks/stair_protocol_v2.json",
            cell_variant_id=STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID,
        )
        cls.plugin = StairClimbingTaskPlugin(cls.config)
        cls.definition = build_stair_cell_catalog(
            config=cls.config,
            plugin=cls.plugin,
        ).resolve(STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID)
        cls.default_registry = build_default_robot_registry()

    def test_stair_requires_every_profile_support_without_fixed_maximum(self) -> None:
        support_requirement = next(
            item
            for item in self.plugin.robot_requirements().roles
            if item.role_id == SUPPORT_EFFECTOR_ROLE
        )
        self.assertEqual(support_requirement.minimum_count, 1)
        self.assertIsNone(support_requirement.maximum_count)
        self.assertTrue(support_requirement.require_all_profile_targets)

        for support_count in (2, 4, 6):
            with self.subTest(support_count=support_count):
                profile, adapter, binding, registry = self._execution_fixture(
                    support_count
                )
                projection = stair_runtime_projection_from_execution(
                    profile=profile,
                    adapter=adapter,
                    binding=binding,
                )
                definition = replace(
                    self.definition,
                    robot_task_binding=binding,
                    contract_hashes={
                        **self.definition.contract_hashes,
                        "reward_input_vnext_schema_hash": (
                            projection.reward_input_schema_hash_vnext
                        ),
                        "reward_capability_vnext_hash": (
                            projection.reward_capability_hash_vnext
                        ),
                    },
                    legacy_training_binding_v1_grant_hash=None,
                )
                manifest = compile_stair_cell_manifest(
                    definition=definition,
                    robot_registry=registry,
                    requirements=self.plugin.robot_requirements(),
                    base_task_protocol_hash=self.config.task_protocol_hash,
                ).require_supported()
                self.assertEqual(
                    len(manifest.resolved_roles[SUPPORT_EFFECTOR_ROLE]),
                    support_count,
                )

                incomplete_binding = replace(
                    binding,
                    role_bindings={
                        **binding.role_bindings,
                        SUPPORT_EFFECTOR_ROLE: binding.role_bindings[
                            SUPPORT_EFFECTOR_ROLE
                        ][:-1],
                    },
                    binding_hash=None,
                )
                incomplete = compile_stair_cell_manifest(
                    definition=replace(
                        definition,
                        robot_task_binding=incomplete_binding,
                    ),
                    robot_registry=registry,
                    requirements=self.plugin.robot_requirements(),
                    base_task_protocol_hash=self.config.task_protocol_hash,
                )
                self.assertEqual(incomplete.status, "unsupported")
                self.assertIn(
                    "role_selection_incomplete:support_effector",
                    incomplete.reasons,
                )

    def test_vnext_schema_tracks_real_morphology_without_fake_g1_fields(self) -> None:
        projections = []
        for support_count in (2, 4, 6):
            profile, adapter, binding, _ = self._execution_fixture(support_count)
            projection = stair_runtime_projection_from_execution(
                profile=profile,
                adapter=adapter,
                binding=binding,
            )
            projections.append(projection)
            self.assertEqual(projection.support_count, support_count)
            self.assertEqual(projection.action_count, projection.joint_count)
            self.assertEqual(
                projection.to_summary_dict()["resolved_roles"][SUPPORT_EFFECTOR_ROLE],
                list(binding.role_bindings[SUPPORT_EFFECTOR_ROLE]),
            )
            field_names = tuple(
                field.name for field in projection.reward_input_fields_vnext
            )
            for morphology_token in ("ankle", "arm", "finger", "torso", "foot"):
                self.assertFalse(
                    any(morphology_token in name for name in field_names),
                    (support_count, morphology_token, field_names),
                )

        self.assertEqual(
            len({item.reward_input_schema_hash_vnext for item in projections}),
            3,
        )
        self.assertEqual(
            len({item.reward_capability_hash_vnext for item in projections}),
            3,
        )

        joint_shape_projections = tuple(
            replace(
                projections[0],
                cell_variant_id=f"stair.synthetic_joint_shape_{joint_count}_v1",
                joint_names=tuple(
                    f"generic_joint_{index}" for index in range(joint_count)
                ),
                action_joint_names=tuple(
                    f"generic_joint_{index}" for index in range(action_count)
                ),
            )
            for joint_count, action_count in ((12, 12), (18, 12), (37, 37))
        )
        self.assertEqual(
            len(
                {
                    item.reward_input_schema_hash_vnext
                    for item in joint_shape_projections
                }
            ),
            3,
        )
        for projection, expected in zip(
            joint_shape_projections,
            ((12, 12), (18, 12), (37, 37)),
            strict=True,
        ):
            fields = {
                field.name: field.trailing_shape
                for field in projection.reward_input_fields_vnext
            }
            self.assertEqual(fields["joint_pos"], (expected[0],))
            self.assertEqual(fields["last_action"], (expected[1],))

    def test_unified_cell_uses_versioned_profile_joint_groups(self) -> None:
        config = StairClimbingTaskConfig.defaults(
            ROOT / "configs/tasks/stair_unified_protocol_v2.json",
            cell_variant_id=STAIR_G1_37DOF_UNIFIED_V2_CELL_VARIANT_ID,
        )
        manifest = resolve_stair_cell_manifest(
            cell_variant_id=STAIR_G1_37DOF_UNIFIED_V2_CELL_VARIANT_ID,
            config=config,
            plugin=StairClimbingTaskPlugin(config),
        )
        self.assertEqual(
            manifest.robot_profile["robot_profile_id"],
            G1_37DOF_ISAACLAB_GROUPED_PROFILE_ID,
        )
        self.assertEqual(
            manifest.runtime_adapter["runtime_adapter_id"],
            G1_37DOF_ISAACLAB_GROUPED_ADAPTER_ID,
        )
        grouped = self.default_registry.profile(
            G1_37DOF_ISAACLAB_GROUPED_PROFILE_ID
        )
        legacy = self.default_registry.profile(G1_37DOF_ISAACLAB_PROFILE_ID)
        self.assertEqual(grouped.runtime_facts, legacy.runtime_facts)
        all_joints = grouped.semantic_roles["joint_group.all"].targets
        groups = {
            role_id: role.targets
            for role_id, role in grouped.semantic_roles.items()
            if role_id.startswith("joint_group.") and role_id != "joint_group.all"
        }
        self.assertEqual(
            groups["joint_group.ankle"],
            tuple(name for name in all_joints if "_ankle_" in name),
        )
        self.assertEqual(
            groups["joint_group.locomotion"],
            tuple(
                name
                for name in all_joints
                if any(token in name for token in ("_hip_", "_knee_", "_ankle_"))
            ),
        )
        self.assertEqual(
            groups["joint_group.hip_knee"],
            tuple(
                name
                for name in all_joints
                if "_hip_" in name or "_knee_" in name
            ),
        )
        projection = stair_runtime_projection_from_manifest(manifest)
        self.assertEqual(
            {role_id: len(names) for role_id, names in projection.joint_groups},
            {
                "joint_group.ankle": 4,
                "joint_group.arm": 10,
                "joint_group.finger": 14,
                "joint_group.hip_knee": 8,
                "joint_group.hip_yaw_roll": 4,
                "joint_group.locomotion": 12,
                "joint_group.torso": 1,
            },
        )
        fields = {
            field.name: field.trailing_shape
            for field in projection.reward_input_fields_vnext
        }
        self.assertEqual(fields["ankle_group_pos"], (4,))
        self.assertEqual(fields["ankle_group_soft_pos_limits"], (4, 2))
        self.assertEqual(fields["locomotion_group_torque"], (12,))
        self.assertEqual(fields["hip_knee_group_acc"], (8,))

    def test_binding_can_request_a_new_profile_group_without_projection_code(self) -> None:
        profile, adapter, binding, _ = self._execution_fixture(4)
        all_joints = profile.semantic_roles["joint_group.all"].targets
        role_id = "joint_group.thigh"
        profile = replace(
            profile,
            semantic_roles={
                **profile.semantic_roles,
                role_id: SemanticRole(role_id, "joint", all_joints[:2]),
            },
            profile_hash=None,
        )
        task_projection = copy.deepcopy(dict(binding.task_projection))
        task_projection["stair_runtime_projection"][
            "reward_joint_group_inputs"
        ] = [{"role_id": role_id, "source": "joint_torque"}]
        binding = replace(
            binding,
            task_projection=task_projection,
            binding_hash=None,
        )
        projection = stair_runtime_projection_from_execution(
            profile=profile,
            adapter=adapter,
            binding=binding,
        )
        fields = {
            field.name: field.trailing_shape
            for field in projection.reward_input_fields_vnext
        }
        self.assertEqual(fields["thigh_group_torque"], (2,))
        self.assertEqual(
            projection.joint_group_input_specs,
            (
                (
                    "thigh_group_torque",
                    "joint_group.thigh",
                    "joint_torque",
                    "N*m",
                    (),
                ),
            ),
        )

    def test_runtime_ids_are_resolved_from_projection_and_reported(self) -> None:
        profile, adapter, binding, _ = self._execution_fixture(4)
        projection = stair_runtime_projection_from_execution(
            profile=profile,
            adapter=adapter,
            binding=binding,
        )
        robot_names = (
            *projection.joint_names,
            *projection.base_body_names,
            *projection.support_body_names,
        )
        contact_names = (
            *projection.support_body_names,
            *projection.forbidden_contact_body_names,
        )
        resolved = resolve_stair_runtime_projection(
            projection,
            robot=_ResolverAsset(robot_names),
            contact_sensor=_ResolverAsset(contact_names),
        )
        summary = resolved.to_summary_dict()
        self.assertEqual(summary["projection_hash"], projection.projection_hash)
        self.assertEqual(len(summary["resolved_runtime"]["support_body_ids"]), 4)
        self.assertEqual(
            len(summary["resolved_runtime"]["contact_support_body_ids"]),
            4,
        )
        self.assertEqual(
            summary["reward_input_schema_hash_vnext"],
            projection.reward_input_schema_hash_vnext,
        )

    def test_projection_identity_is_effective_content_not_manifest_provenance(
        self,
    ) -> None:
        manifest = resolve_stair_cell_manifest(
            cell_variant_id=STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID,
            config=self.config,
            plugin=self.plugin,
        )
        with_manifest = stair_runtime_projection_from_manifest(manifest)
        profile, adapter = self.default_registry.execution(
            robot_profile_id=self.definition.robot_profile_id,
            runtime_adapter_id=self.definition.runtime_adapter_id,
        )
        without_manifest = stair_runtime_projection_from_execution(
            profile=profile,
            adapter=adapter,
            binding=self.definition.robot_task_binding,
        )
        self.assertEqual(with_manifest.projection_hash, without_manifest.projection_hash)
        self.assertEqual(
            with_manifest.resolved_cell_manifest_hash,
            manifest.manifest_hash,
        )
        self.assertIsNone(without_manifest.resolved_cell_manifest_hash)

    def test_kinematic_base_source_is_binding_owned_not_legacy_inferred(self) -> None:
        profile, adapter = self.default_registry.execution(
            robot_profile_id=self.definition.robot_profile_id,
            runtime_adapter_id=self.definition.runtime_adapter_id,
        )
        root_projection = stair_runtime_projection_from_execution(
            profile=profile,
            adapter=adapter,
            binding=self.definition.robot_task_binding,
        )
        self.assertEqual(
            root_projection.kinematic_base_source,
            STAIR_KINEMATIC_BASE_ARTICULATION_ROOT,
        )
        task_projection = copy.deepcopy(
            dict(self.definition.robot_task_binding.task_projection)
        )
        task_projection["stair_runtime_projection"]["kinematic_base"][
            "source"
        ] = STAIR_KINEMATIC_BASE_BODY_ROLE
        body_binding = replace(
            self.definition.robot_task_binding,
            task_projection=task_projection,
            binding_hash=None,
        )
        body_projection = stair_runtime_projection_from_execution(
            profile=profile,
            adapter=adapter,
            binding=body_binding,
        )
        self.assertEqual(
            body_projection.kinematic_base_source,
            STAIR_KINEMATIC_BASE_BODY_ROLE,
        )
        self.assertNotEqual(root_projection.projection_hash, body_projection.projection_hash)

    def test_stair_compiler_requires_vnext_reward_identities(self) -> None:
        for key in (
            "reward_input_vnext_schema_hash",
            "reward_capability_vnext_hash",
        ):
            with self.subTest(key=key):
                contracts = dict(self.definition.contract_hashes)
                contracts.pop(key)
                result = compile_stair_cell_manifest(
                    definition=replace(
                        self.definition,
                        contract_hashes=contracts,
                        legacy_training_binding_v1_grant_hash=None,
                    ),
                    robot_registry=self.default_registry,
                    requirements=self.plugin.robot_requirements(),
                    base_task_protocol_hash=self.config.task_protocol_hash,
                )
                self.assertEqual(result.status, "unsupported")
                self.assertTrue(
                    any("VNext reward" in reason for reason in result.reasons),
                    result.reasons,
                )

    def test_generic_stair_runtime_contains_no_concrete_g1_body_names(self) -> None:
        generic_paths = (
            "runtime_projection.py",
            "isaaclab_reward_runtime.py",
            "isaaclab_runtime_env.py",
            "transition_decoder.py",
            "episode_evidence.py",
            "evaluation_semantics.py",
            "terrain_evidence.py",
            "plugin.py",
            "prompt_adapter.py",
        )
        shared_role_vocabulary = {
            str(role_name)
            for profile_id in self.default_registry.profile_ids()
            for role_name in self.default_registry.profile(profile_id).semantic_roles
        }
        concrete_robot_names = {
            str(name)
            for profile_id in self.default_registry.profile_ids()
            for key in ("body_names", "joint_names")
            for name in self.default_registry.profile(profile_id).runtime_facts.get(
                key, ()
            )
            if str(name) not in shared_role_vocabulary
        }
        forbidden_source_markers = (
            "_FOOT_BODY_NAMES",
            "_BASE_CONTACT_BODY_NAMES",
            "range(2)",
            "maximum_count=2",
            "Unitree G1 Stair Climbing",
            '["policy_compatibility"]["robot_schema"]',
        )
        root = SRC / "true_rl_agent/tasks/stair_climbing"
        for filename in generic_paths:
            source = (root / filename).read_text(encoding="utf-8")
            for marker in forbidden_source_markers:
                self.assertNotIn(marker, source, (filename, marker))
            string_literals = {
                node.value
                for node in ast.walk(ast.parse(source, filename=filename))
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            }
            for name in concrete_robot_names:
                self.assertFalse(
                    any(name in literal for literal in string_literals),
                    (filename, name),
                )

    def _execution_fixture(
        self,
        support_count: int,
    ):
        if support_count not in {2, 4, 6}:
            raise ValueError("test support count must be 2, 4, or 6")
        original = self.default_registry.profile(G1_37DOF_ISAACLAB_PROFILE_ID)
        original_support = tuple(original.semantic_roles[SUPPORT_EFFECTOR_ROLE].targets)
        supports = original_support + tuple(
            f"synthetic_support_{index}_link"
            for index in range(len(original_support), support_count)
        )
        facts = copy.deepcopy(dict(original.runtime_facts))
        facts["body_names"] = list(dict.fromkeys([*facts["body_names"], *supports]))
        facts["support_body_names"] = list(supports)
        facts_without_hash = {
            key: value for key, value in facts.items() if key != "runtime_facts_hash"
        }
        facts["runtime_facts_hash"] = canonical_json_hash(facts_without_hash)
        roles = dict(original.semantic_roles)
        roles[SUPPORT_EFFECTOR_ROLE] = SemanticRole(
            SUPPORT_EFFECTOR_ROLE,
            "body",
            supports,
        )
        profile = replace(
            original,
            runtime_facts=facts,
            semantic_roles=roles,
            profile_hash=None,
        )
        adapter = self.default_registry.runtime_adapter(
            self.definition.runtime_adapter_id
        )
        binding = replace(
            self.definition.robot_task_binding,
            role_bindings={
                **self.definition.robot_task_binding.role_bindings,
                SUPPORT_EFFECTOR_ROLE: supports,
            },
            binding_hash=None,
        )
        registry = RobotRegistry()
        registry.register_profile(profile)
        registry.register_runtime_adapter(adapter)
        return profile, adapter, binding, registry


class _ResolverAsset:
    def __init__(self, names: tuple[str, ...]) -> None:
        self._ids = {name: index for index, name in enumerate(dict.fromkeys(names))}

    def find_joints(self, names: list[str], *, preserve_order: bool):
        return self._resolve(names, preserve_order=preserve_order)

    def find_bodies(self, names: list[str], *, preserve_order: bool):
        return self._resolve(names, preserve_order=preserve_order)

    def _resolve(self, names: list[str], *, preserve_order: bool):
        if not preserve_order:
            raise AssertionError("runtime projection must preserve descriptor order")
        return [self._ids[name] for name in names], list(names)


if __name__ == "__main__":
    unittest.main()
