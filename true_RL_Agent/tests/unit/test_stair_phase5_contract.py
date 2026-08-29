from __future__ import annotations

import ast
import unittest
from pathlib import Path

from true_rl_agent.robots.defaults import (
    GO2_12DOF_ISAACLAB_ADAPTER_ID,
    GO2_12DOF_ISAACLAB_PROFILE_ID,
    build_default_robot_registry,
)
from true_rl_agent.robots.isaaclab_locomotion import (
    isaaclab_locomotion_runtime_spec,
)
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.composition import (
    StairClimbingCompositionConfig,
)
from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
from true_rl_agent.tasks.stair_climbing.reference_program import (
    resolve_official_stair_reference_program,
)
from true_rl_agent.tasks.stair_climbing.robot_binding import (
    STAIR_GO2_12DOF_UNIFIED_CELL_VARIANT_ID,
    resolve_stair_cell_manifest,
)
from true_rl_agent.tasks.stair_climbing.runtime_projection import (
    stair_runtime_projection_from_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
GO2_PROTOCOL = ROOT / "configs/tasks/stair_go2_12dof_unified_protocol_v1.json"
G1_PROTOCOL = ROOT / "configs/tasks/stair_unified_protocol_v1.json"


class StairPhase5ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_default_robot_registry()
        cls.config = StairClimbingTaskConfig.defaults(GO2_PROTOCOL)
        cls.plugin = StairClimbingTaskPlugin(cls.config)
        cls.manifest = resolve_stair_cell_manifest(
            cell_variant_id=cls.config.cell_variant_id,
            config=cls.config,
            plugin=cls.plugin,
        )
        cls.projection = stair_runtime_projection_from_manifest(cls.manifest)

    def test_go2_profile_and_runtime_adapter_preserve_phase0_facts(self) -> None:
        profile, adapter = self.registry.execution(
            robot_profile_id=GO2_12DOF_ISAACLAB_PROFILE_ID,
            runtime_adapter_id=GO2_12DOF_ISAACLAB_ADAPTER_ID,
        )
        facts = profile.runtime_facts
        self.assertEqual(
            profile.runtime_facts_hash,
            "c56fed536354249bec90ce22fa52a10f8b899396758e46f8e03747368d985437",
        )
        self.assertEqual(len(facts["joint_names"]), 12)
        self.assertEqual(len(facts["body_names"]), 19)
        self.assertEqual(len(facts["support_body_names"]), 4)
        self.assertEqual(facts["asset_closure"]["file_count"], 2)
        self.assertEqual(facts["asset_closure"]["unresolved"], [])
        self.assertEqual(facts["asset_closure"]["total_bytes"], 18_810_329)
        actuator = facts["actuators"][0]
        self.assertEqual(actuator["class"], "isaaclab.actuators.actuator_pd.DCMotor")
        self.assertEqual(
            set(actuator["parameters_by_joint"]),
            {
                "armature",
                "damping",
                "dynamic_friction",
                "effort_limit",
                "friction",
                "stiffness",
                "velocity_limit",
            },
        )
        runtime = isaaclab_locomotion_runtime_spec(adapter.runtime_adapter_id)
        self.assertIn("config.go2.rough_env_cfg", runtime.environment_cfg_entrypoint)
        self.assertEqual(runtime.agent_configuration.algorithm["entropy_coef"], 0.01)

    def test_go2_binding_compiles_one_four_support_unified_cell(self) -> None:
        self.assertEqual(
            self.config.cell_variant_id,
            STAIR_GO2_12DOF_UNIFIED_CELL_VARIANT_ID,
        )
        self.assertEqual(
            self.manifest.robot_profile["robot_profile_id"],
            GO2_12DOF_ISAACLAB_PROFILE_ID,
        )
        self.assertEqual(self.projection.action_count, 12)
        self.assertEqual(self.projection.support_count, 4)
        self.assertEqual(self.projection.forbidden_contact_count, 1)
        self.assertEqual(self.projection.kinematic_base_source, "articulation_root")
        self.assertEqual(self.projection.joint_groups, ())
        self.assertTrue(self.projection.support_transition_inputs)
        self.assertAlmostEqual(self.projection.task_vertical_scale, 0.589087)
        self.assertEqual(self.projection.task_horizontal_scale, 1.0)
        input_names = {field.name for field in self.projection.reward_input_fields_vnext}
        self.assertIn("support_last_air_time", input_names)
        self.assertIn("support_first_contact", input_names)

        values = StairClimbingCompositionConfig.defaults(
            project_root=ROOT
        ).to_task_defaults()
        values["cell_variant_id"] = STAIR_GO2_12DOF_UNIFIED_CELL_VARIANT_ID
        selected = StairClimbingCompositionConfig.from_values(
            values
        ).bind_selected_cell_contract(project_root=ROOT)
        self.assertEqual(
            selected.runtime_config,
            "configs/tasks/stair_go2_12dof_unified_protocol_v1.json",
        )

    def test_go2_distribution_keeps_cost_claims_and_cross_robot_anchors(self) -> None:
        profile = self.config.unified_training_profile
        self.assertEqual(profile.terrain["c0"]["riser_range_m"], (0.03, 0.206))
        self.assertEqual(
            profile.terrain["c1"]["riser_base_range_m"],
            (0.047, 0.118),
        )
        self.assertEqual(profile.terrain["c1"]["riser_offset_m"], 0.012)
        suite = self.config.ordered_evaluation_suite
        geometry_by_cohort = {}
        for cohort in ("public_dev", "public_accept", "sealed_validation"):
            cells = suite.cells_for_cohort(cohort)
            self.assertEqual(len(cells), 24)
            geometry_by_cohort[cohort] = {cell.geometry_digest for cell in cells}
            self.assertEqual(len(geometry_by_cohort[cohort]), 24)
        self.assertEqual(len(set().union(*geometry_by_cohort.values())), 72)
        self.assertTrue(
            all(
                not geometry_by_cohort[left] & geometry_by_cohort[right]
                for left, right in (
                    ("public_dev", "public_accept"),
                    ("public_dev", "sealed_validation"),
                    ("public_accept", "sealed_validation"),
                )
            )
        )
        self.assertEqual(
            sum(suite.seed_count(cohort) for cohort in geometry_by_cohort)
            * 24
            * suite.episodes_per_seed_per_cell,
            1080,
        )

        g1_config = StairClimbingTaskConfig.defaults(G1_PROTOCOL)
        g1_anchors = {
            (cell.geometry_digest, cell.stratum_id)
            for cell in g1_config.ordered_evaluation_suite.cells
        }
        go2_anchors = {
            (cell.geometry_digest, cell.stratum_id) for cell in suite.cells
        }
        self.assertEqual(
            {stratum for _, stratum in g1_anchors & go2_anchors},
            {"c0_id", "c1_id"},
        )

    def test_go2_reference_is_the_eight_term_official_effective_manager(self) -> None:
        reference = resolve_official_stair_reference_program(
            self.config,
            resolved_cell_manifest=self.manifest,
        )
        weights = {term.name: term.weight for term in reference.reward_program.reward_terms}
        self.assertEqual(
            weights,
            {
                "track_lin_vel_xy_exp": 1.5,
                "track_ang_vel_z_exp": 0.75,
                "lin_vel_z_l2": -2.0,
                "ang_vel_xy_l2": -0.05,
                "dof_torques_l2": -0.0002,
                "dof_acc_l2": -2.5e-7,
                "action_rate_l2": -0.01,
                "feet_air_time": 0.01,
            },
        )
        equivalence = self.config.vnext_reward_authority.runtime_authority_payload()[
            "reference_equivalence"
        ]
        self.assertEqual(set(equivalence["active_term_names"]), set(weights))
        self.assertEqual(
            equivalence["zero_weight_term_names"],
            ["flat_orientation_l2", "dof_pos_limits"],
        )
        self.assertEqual(equivalence["disabled_term_names"], ["undesired_contacts"])

    def test_task_runtime_has_no_go2_specific_dispatch_or_morphology_names(self) -> None:
        task_root = ROOT / "src/true_rl_agent/tasks/stair_climbing"
        generic_paths = (
            "plugin.py",
            "isaaclab_runtime_env.py",
            "isaaclab_execution.py",
            "isaaclab_unified_terrain.py",
            "isaaclab_evaluation_terrain.py",
            "isaaclab_irregular_terrain.py",
            "isaaclab_rough_terrain.py",
            "rough_surface.py",
            "unified_training.py",
            "unified_evaluation.py",
            "unified_scoring.py",
            "transition_decoder.py",
            "evaluation_semantics.py",
            "prompt_adapter.py",
        )
        profile = self.registry.profile(GO2_12DOF_ISAACLAB_PROFILE_ID)
        role_names = set(profile.semantic_roles)
        concrete_names = {
            str(name)
            for field_name in ("body_names", "joint_names")
            for name in profile.runtime_facts[field_name]
            if str(name) not in role_names
        }
        for relative_path in generic_paths:
            source = (task_root / relative_path).read_text(encoding="utf-8")
            literals = {
                node.value
                for node in ast.walk(ast.parse(source, filename=relative_path))
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            }
            self.assertFalse(
                {
                    name
                    for name in concrete_names
                    if any(name in literal for literal in literals)
                },
                relative_path,
            )
            self.assertNotIn("Unitree/Go2", source, relative_path)
            self.assertNotIn("go2.usd", source, relative_path)
        self.assertNotIn(
            "go2",
            (task_root / "isaaclab_runtime_env.py").read_text(encoding="utf-8").lower(),
        )
        self.assertNotIn(
            "go2",
            (task_root / "isaaclab_execution.py").read_text(encoding="utf-8").lower(),
        )

    def test_existing_g1_unified_identity_and_abi_remain_unchanged(self) -> None:
        g1 = StairClimbingTaskConfig.defaults(G1_PROTOCOL)
        self.assertEqual(
            g1.task_protocol_hash,
            "e65239927cbf98f6cf12b0ddb959c6bd1390bdf7fd1c6a262fbbab7f1f04dcf8",
        )
        self.assertEqual(
            g1.training_agent_configuration().agent_configuration_hash,
            "8a5d64c47ac034695d2eec23ff60a89ecbf1253694edb8c373a4da86872bb450",
        )
        g1_plugin = StairClimbingTaskPlugin(g1)
        g1_manifest = resolve_stair_cell_manifest(
            cell_variant_id=g1.cell_variant_id,
            config=g1,
            plugin=g1_plugin,
        )
        g1_input_names = {
            field.name
            for field in stair_runtime_projection_from_manifest(
                g1_manifest
            ).reward_input_fields_vnext
        }
        self.assertNotIn("support_last_air_time", g1_input_names)
        self.assertNotIn("support_first_contact", g1_input_names)


if __name__ == "__main__":
    unittest.main()
