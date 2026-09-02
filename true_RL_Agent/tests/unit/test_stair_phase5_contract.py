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
from true_rl_agent.tasks.stair_climbing.prompt_adapter import (
    _minimum_core_task_section,
    _success_criteria_prompt_contract,
)
from true_rl_agent.tasks.stair_climbing.public_projection import (
    _validate_environment_source_preflight_pair,
)
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
from true_rl_agent.tasks.stair_climbing.unified_protocol import (
    StairUnifiedProtocol,
)
from true_rl_agent.tasks.stair_climbing.training_distribution import (
    build_stair_execution_source_preflight_summary,
)


ROOT = Path(__file__).resolve().parents[2]
GO2_PROTOCOL = ROOT / "configs/tasks/stair_go2_12dof_unified_protocol_v2.json"
G1_PROTOCOL = ROOT / "configs/tasks/stair_unified_protocol_v2.json"
GO2_PROTOCOL_V1 = ROOT / "configs/tasks/stair_go2_12dof_unified_protocol_v1.json"
G1_PROTOCOL_V1 = ROOT / "configs/tasks/stair_unified_protocol_v1.json"


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
            "configs/tasks/stair_go2_12dof_unified_protocol_v2.json",
        )

    def test_go2_distribution_keeps_cost_claims_and_cross_robot_anchors(self) -> None:
        profile = self.config.unified_training_profile
        self.assertEqual(profile.terrain["c0"]["riser_range_m"], (0.03, 0.206))
        self.assertEqual(
            profile.terrain["c1"]["riser_base_range_m"],
            (0.047, 0.118),
        )
        self.assertEqual(profile.terrain["c1"]["riser_offset_m"], 0.007069)
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
            {"c0_id"},
        )

    def test_go2_reference_declares_two_stair_context_weight_adaptations(self) -> None:
        reference = resolve_official_stair_reference_program(
            self.config,
            resolved_cell_manifest=self.manifest,
        )
        weights = {term.name: term.weight for term in reference.reward_program.reward_terms}
        self.assertEqual(
            weights,
            {
                "track_lin_vel_xy_exp": 3.5,
                "track_ang_vel_z_exp": 0.75,
                "ang_vel_xy_l2": -0.05,
                "flat_orientation_l2": -2.5,
                "dof_torques_l2": -0.0002,
                "dof_acc_l2": -2.5e-7,
                "action_rate_l2": -0.01,
                "feet_air_time": 0.25,
            },
        )
        equivalence = self.config.vnext_reward_authority.runtime_authority_payload()[
            "reference_equivalence"
        ]
        self.assertEqual(set(equivalence["active_term_names"]), set(weights))
        self.assertEqual(
            equivalence["zero_weight_term_names"],
            ["lin_vel_z_l2", "dof_pos_limits"],
        )
        self.assertEqual(equivalence["disabled_term_names"], ["undesired_contacts"])
        self.assertEqual(
            {
                item["term_name"]: (
                    item["upstream_weight"], item["reference_weight"]
                )
                for item in equivalence["weight_adaptations"]
            },
            {
                "track_lin_vel_xy_exp": (1.5, 3.5),
                "lin_vel_z_l2": (-2.0, 0.0),
            },
        )

    def test_go2_prompt_uses_morphology_neutral_vnext_semantics(self) -> None:
        config = StairClimbingTaskConfig.defaults(
            GO2_PROTOCOL,
            resolved_cell_task_protocol_hash=(
                self.manifest.resolved_cell_task_protocol_hash
            ),
        )
        plugin = StairClimbingTaskPlugin(
            config,
            resolved_cell_manifest=self.manifest,
        )
        sections = plugin.prompt_adapter().build_prompt_capacity_preflight_sections()
        contract = sections["reward_inputs_contract"]
        self.assertEqual(
            contract["schema_version"],
            "true_rl_agent.reward_inputs_contract.stair.v13",
        )
        self.assertTrue(contract["input_type"].endswith(".StairRewardInputsVNext"))
        constraints = contract["field_semantics_constraints"]
        self.assertIn(
            "support_contact_force_history",
            constraints["high_risk_axis_semantics"],
        )
        self.assertNotIn(
            "foot_contact_force_history",
            constraints["high_risk_axis_semantics"],
        )
        self.assertEqual(
            constraints["termination_masks"]["equivalent_fields"],
            {"terminated_non_timeout": "forbidden_contact"},
        )
        self.assertIn(
            "3-sample",
            constraints["support_contact"]["semantics"],
        )
        self.assertIn(
            "not current-sample-only",
            constraints["support_contact"]["temporal_behavior"],
        )
        prompt_text = str(sections)
        self.assertNotIn("foot(0 left;1 right)", prompt_text)
        self.assertNotIn("StairRewardInputsV2", prompt_text)

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

    def test_g1_v2_changes_outcome_identity_without_changing_training_abi(self) -> None:
        historical = StairUnifiedProtocol.load(
            ROOT / "configs/tasks/stair_unified_protocol_v1.json"
        )
        self.assertEqual(
            historical.task_protocol_hash,
            "e65239927cbf98f6cf12b0ddb959c6bd1390bdf7fd1c6a262fbbab7f1f04dcf8",
        )
        g1 = StairClimbingTaskConfig.defaults(G1_PROTOCOL)
        self.assertEqual(
            g1.task_protocol_hash,
            "fc84b8eece40372afd6680acf711cb3b754bba660608318e5a334492d9587ded",
        )
        self.assertEqual(
            historical.training_profile.profile_hash,
            g1.unified_training_profile.profile_hash,
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

    def test_v2_prompts_contain_no_pseudo_fall_clearance(self) -> None:
        for protocol_path in (G1_PROTOCOL, GO2_PROTOCOL):
            with self.subTest(protocol_path=protocol_path.name):
                outcomes = StairClimbingTaskConfig.defaults(
                    protocol_path
                ).episode_outcomes()
                self.assertNotIn("low_base_clearance", outcomes["signals"])
                self.assertNotIn("low_base_clearance", outcomes["primary_reasons"])
                self.assertNotIn(
                    "low_base_clearance_detected", outcomes["diagnostic_flags"]
                )
                success_criteria = _success_criteria_prompt_contract(outcomes)
                self.assertNotIn(
                    "minimum_kinematic_base_clearance_above_top_m",
                    success_criteria,
                )
                compact = _minimum_core_task_section(
                    {
                        "fixed_eval_protocol_summary": {
                            "success_criteria": success_criteria
                        }
                    }
                )
                self.assertNotIn(
                    "minimum_kinematic_base_clearance_above_top_m",
                    compact["fixed_eval_protocol_summary"]["success_criteria"],
                )

    def test_v2_success_semantics_reuse_v1_training_identity_for_both_rows(self) -> None:
        for v1_path, v2_path, expected_binding_hash, predecessor_manifest_hash in (
            (
                G1_PROTOCOL_V1,
                G1_PROTOCOL,
                "d7aea85d7742c57eae1753b8b236b32555269d9bcd8c25ed2b3742d2e045e14c",
                "cb9c93c1239d2f20eb58737b684cc74e21b6675a724198c470a58410f9ccdae2",
            ),
            (
                GO2_PROTOCOL_V1,
                GO2_PROTOCOL,
                "c0112a87a7fb5750ef11d55a6bb50cfdeb0be662380c1387b2d8cab8c977de98",
                "54cd180c9f39f65bffd71482ec8a87cda912d0e378e8b434d0c44c839a3ab1c2",
            ),
        ):
            v1 = StairClimbingTaskConfig.defaults(v1_path)
            v2 = StairClimbingTaskConfig.defaults(v2_path)
            self.assertNotEqual(v1.task_protocol_hash, v2.task_protocol_hash)
            self.assertEqual(
                v2.unified_protocol_config.training_lineage["source_cell_variant_id"],
                v1.cell_variant_id,
            )
            self.assertEqual(v1.train_seed, v2.train_seed)
            self.assertIsNone(v1.training_predecessor_manifest_hash)
            self.assertEqual(
                v2.training_predecessor_manifest_hash,
                predecessor_manifest_hash,
            )
            self.assertEqual(
                v1.unified_training_profile.profile_hash,
                v2.unified_training_profile.profile_hash,
            )
            self.assertEqual(
                v1.training_agent_configuration().agent_configuration_hash,
                v2.training_agent_configuration().agent_configuration_hash,
            )
            self.assertEqual(
                v1.train_runtime_profile().runtime_dynamics_hash,
                v2.train_runtime_profile().runtime_dynamics_hash,
            )
            self.assertNotEqual(
                v1.eval_runtime_profile().runtime_dynamics_hash,
                v2.eval_runtime_profile().runtime_dynamics_hash,
            )

            bound_configs = []
            for config, protocol_path in zip(
                (v1, v2),
                (v1_path, v2_path),
                strict=True,
            ):
                plugin = StairClimbingTaskPlugin(config)
                manifest = resolve_stair_cell_manifest(
                    cell_variant_id=config.cell_variant_id,
                    config=config,
                    plugin=plugin,
                )
                bound_configs.append(
                    StairClimbingTaskConfig.defaults(
                        protocol_path,
                        resolved_cell_task_protocol_hash=(
                            manifest.resolved_cell_task_protocol_hash
                        ),
                    )
                )
            v1_binding = bound_configs[0].formal_training_binding()
            v2_binding = bound_configs[1].formal_training_binding()
            self.assertEqual(v1_binding.binding_hash, expected_binding_hash)
            self.assertEqual(v2_binding.binding_hash, expected_binding_hash)
            self.assertEqual(v1_binding.to_private_dict(), v2_binding.to_private_dict())

    def test_go2_v2_source_preflight_accepts_only_its_frozen_predecessor(self) -> None:
        config = StairClimbingTaskConfig.defaults(
            GO2_PROTOCOL,
            resolved_cell_task_protocol_hash=(
                self.manifest.resolved_cell_task_protocol_hash
            ),
        )
        train_preflight = build_stair_execution_source_preflight_summary(
            task_protocol_hash=config.training_binding_task_protocol_hash,
            source_manifest_hash="a" * 64,
        )
        eval_preflight = build_stair_execution_source_preflight_summary(
            task_protocol_hash=config.task_protocol_hash,
            source_manifest_hash="c" * 64,
        )
        lineage = config.unified_protocol.training_lineage
        assert lineage is not None
        train_summary = {
            "environment_source_preflight": train_preflight,
            "resolved_cell_manifest_hash": config.training_predecessor_manifest_hash,
            "resolved_cell_identity": {
                "cell_variant_id": lineage["source_cell_variant_id"],
                "resolved_cell_task_protocol_hash": (
                    config.training_binding_task_protocol_hash
                ),
            },
        }
        self.assertEqual(
            _validate_environment_source_preflight_pair(
                config,
                train_summary=train_summary,
                eval_preflight=eval_preflight,
            ),
            eval_preflight,
        )
        with self.assertRaisesRegex(ValueError, "train/eval source preflight"):
            _validate_environment_source_preflight_pair(
                config,
                train_summary={
                    **train_summary,
                    "environment_source_preflight": {
                        **train_preflight,
                        "task_protocol_hash": "b" * 64,
                    },
                },
                eval_preflight=eval_preflight,
            )
        with self.assertRaisesRegex(ValueError, "train/eval source preflight"):
            _validate_environment_source_preflight_pair(
                config,
                train_summary={
                    **train_summary,
                    "resolved_cell_manifest_hash": "d" * 64,
                },
                eval_preflight=eval_preflight,
            )


if __name__ == "__main__":
    unittest.main()
