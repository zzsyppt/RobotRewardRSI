"""Current unified Stair single-cell contract after Phase 3B activation work."""

from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.episode_evidence import StairEpisodeOutcome
from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
from true_rl_agent.tasks.stair_climbing.prompt_adapter import (
    StairClimbingPromptAdapter,
    _compact_latest_feedback_entry,
    _compact_required_feedback_entry,
)
from true_rl_agent.tasks.stair_climbing.robot_binding import (
    build_stair_cell_catalog,
    resolve_stair_cell_manifest,
)
from true_rl_agent.tasks.stair_climbing.unified_protocol import (
    StairUnifiedProtocol,
)
from true_rl_agent.tasks.stair_climbing.unified_scoring import (
    build_unified_score_projection,
    build_unified_stair_prompt_feedback,
    build_unified_stair_summary,
    validate_unified_stair_summary,
)
from true_rl_agent.tasks.stair_climbing.unified_training import (
    STAIR_TERRAIN_REGIME_C0,
    STAIR_TERRAIN_REGIME_C1,
    STAIR_TERRAIN_REGIME_C2,
    StairUnifiedTrainingProfile,
    unified_assignment_summary,
    unified_initial_terrain_assignment,
)


class StairPhase3BContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        protocol_path = ROOT / "configs/tasks/stair_unified_protocol_v1.json"
        base = StairClimbingTaskConfig.defaults(protocol_path)
        base_plugin = StairClimbingTaskPlugin(base)
        manifest = resolve_stair_cell_manifest(
            cell_variant_id=base.cell_variant_id,
            config=base,
            plugin=base_plugin,
        )
        cls.manifest = manifest
        cls.config = StairClimbingTaskConfig.defaults(
            protocol_path,
            resolved_cell_task_protocol_hash=(
                manifest.resolved_cell_task_protocol_hash
            ),
        )
        cls.plugin = StairClimbingTaskPlugin(
            cls.config,
            resolved_cell_manifest=manifest,
        )

    def test_single_cell_and_vnext_only_binding(self) -> None:
        catalog = build_stair_cell_catalog(
            config=self.config,
            plugin=self.plugin,
        )
        self.assertEqual(
            catalog.ids(),
            ("stair.g1_37dof.unified_v1",),
        )
        definition = catalog.resolve("stair.g1_37dof.unified_v1")
        self.assertEqual(definition.protocol_state, "active")
        self.assertFalse(self.manifest.uses_legacy_training_binding_v1)
        self.assertEqual(
            self.manifest.contract_hashes["protocol_state_hash"],
            definition.protocol_state_hash,
        )
        self.assertEqual(
            catalog.require_active("stair.g1_37dof.unified_v1"),
            definition,
        )

    def test_unified_training_execution_contract_is_deep_json(self) -> None:
        execution = self.config.training_execution_contract()

        self.assertEqual(
            execution["observation_groups"],
            {"policy": ["policy"], "critic": ["policy"]},
        )
        self.assertEqual(execution["checkpoint_selector"], "final_iteration")
        self.assertIsInstance(execution["observation_groups"]["policy"], list)

    def test_training_distribution_is_exactly_half_c0_half_c1(self) -> None:
        profile = self.config.unified_training_profile
        command = self.config.unified_protocol.training["command"]
        evaluation_command = self.config.unified_protocol.runtime_controls[
            "evaluation_command"
        ]
        self.assertEqual(tuple(command["forward_mps_range"]), (0.0, 0.8))
        self.assertEqual(
            command["forward_mps_range"][-1],
            evaluation_command["forward_mps"],
        )
        self.assertNotIn(
            "floor_depth_m",
            self.config.unified_protocol.evaluation_suite.task_projection()[
                "geometry_policy"
            ],
        )
        terrain_types, terrain_levels = unified_initial_terrain_assignment(
            num_envs=self.config.train_num_envs,
            seed=self.config.train_seed,
            profile=profile,
        )
        receipt = unified_assignment_summary(
            types=terrain_types,
            levels=terrain_levels,
            profile=profile,
        )
        self.assertEqual(len(profile.column_descriptors()), 10)
        self.assertEqual(
            receipt["regime_counts"],
            {
                STAIR_TERRAIN_REGIME_C0: 2048,
                STAIR_TERRAIN_REGIME_C1: 2048,
            },
        )
        self.assertNotIn(STAIR_TERRAIN_REGIME_C2, receipt["regime_counts"])
        self.assertEqual(
            terrain_types,
            unified_initial_terrain_assignment(
                num_envs=self.config.train_num_envs,
                seed=self.config.train_seed,
                profile=profile,
            )[0],
        )

    def test_legacy_stair_010_geometry_is_not_an_id_guard_cell(self) -> None:
        for cell in self.config.ordered_evaluation_suite.cells:
            if cell.stratum_id != "c0_id":
                continue
            self.assertFalse(
                all(abs(value - 0.10) <= 1.0e-12 for value in cell.riser_heights_m)
                and all(abs(value - 0.40) <= 1.0e-12 for value in cell.tread_depths_m)
            )

    def test_evaluation_suite_has_equal_cost_strata_and_distinct_claims(self) -> None:
        suite = self.config.ordered_evaluation_suite
        profile = self.config.unified_training_profile
        self.assertEqual(self.config.eval_num_envs, 72)
        geometry_digests = {}
        for split in ("public_dev", "public_accept", "sealed_validation"):
            cells = suite.cells_for_cohort(split)
            self.assertEqual(len(cells), 24)
            geometry_digests[split] = {cell.geometry_digest for cell in cells}
            self.assertEqual(len(geometry_digests[split]), 24)
            self.assertTrue(
                all(
                    cell.floor_depth_m == profile.terrain["floor_depth_m"]
                    for cell in cells
                )
            )
            self.assertTrue(
                all("cell_id" not in cell.geometry_payload() for cell in cells)
            )
            self.assertEqual(
                Counter(cell.stratum_id for cell in cells),
                Counter(
                    {
                        "c0_id": 3,
                        "c1_id": 3,
                        "height_ood": 3,
                        "tread_ood": 3,
                        "joint_ood": 3,
                        "phase_composition_ood": 3,
                        "c1_conditional_frontier": 3,
                        "c2_mechanism_ood": 3,
                    }
                ),
            )
        for index, left in enumerate(geometry_digests):
            for right in tuple(geometry_digests)[index + 1 :]:
                self.assertFalse(geometry_digests[left] & geometry_digests[right])
        self.assertEqual(len(set().union(*geometry_digests.values())), 72)
        self.assertEqual(
            sum(suite.seed_count(split) for split in geometry_digests)
            * 24
            * suite.episodes_per_seed_per_cell,
            1080,
        )
        phase_cells = [
            cell
            for split in geometry_digests
            for cell in suite.cells_for_cohort(split)
            if cell.stratum_id == "phase_composition_ood"
        ]
        self.assertEqual(
            {cell.factor_levels["phase"] for cell in phase_cells},
            {"p1", "p2", "p4", "p6"},
        )
        self.assertTrue(
            all(
                profile.phase_changes_training_sequence(
                    pattern_id=cell.factor_levels["sequence_pattern"],
                    phase_steps=int(cell.factor_levels["phase"][1:]),
                )
                for cell in phase_cells
            )
        )
        self.assertEqual(
            sum(
                cell.terrain_regime_id == STAIR_TERRAIN_REGIME_C2
                for cell in suite.cells_for_cohort("public_dev")
            ),
            3,
        )
        for split in geometry_digests:
            profiles = [
                cell.surface_profile
                for cell in suite.cells_for_cohort(split)
                if cell.terrain_regime_id == STAIR_TERRAIN_REGIME_C2
            ]
            self.assertEqual(
                sorted(float(profile["amplitude_m"]) for profile in profiles),
                [0.004, 0.006, 0.008],
            )

    def test_training_profile_rejects_phase_labels_without_new_geometry(self) -> None:
        payload = copy.deepcopy(self.config.unified_training_profile.to_dict())
        payload["pattern_families"]["alternating"]["held_out_phase_steps"] = [
            1,
            2,
        ]
        payload["profile_hash"] = canonical_json_hash(
            {key: value for key, value in payload.items() if key != "profile_hash"}
        )
        with self.assertRaisesRegex(ValueError, "distinct geometry sequence"):
            StairUnifiedTrainingProfile.from_mapping(payload)

    def test_catalog_still_rejects_a_real_draft_protocol(self) -> None:
        source_dir = ROOT / "configs/tasks"
        payload = json.loads(
            (source_dir / "stair_unified_protocol_v1.json").read_text(
                encoding="utf-8"
            )
        )
        payload["protocol_state"] = "draft"
        payload["protocol_hash"] = canonical_json_hash(
            {key: value for key, value in payload.items() if key != "protocol_hash"}
        )
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            for filename in payload["component_refs"].values():
                shutil.copyfile(source_dir / filename, directory / filename)
            protocol_path = directory / "draft_protocol.json"
            protocol_path.write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )
            config = StairClimbingTaskConfig.defaults(protocol_path)
            plugin = StairClimbingTaskPlugin(config)
            catalog = build_stair_cell_catalog(config=config, plugin=plugin)
            with self.assertRaisesRegex(ValueError, "not active"):
                catalog.require_active(config.cell_variant_id)

    def test_unified_protocol_loader_is_cell_and_filename_parameterized(self) -> None:
        source_dir = ROOT / "configs/tasks"
        source_protocol = source_dir / "stair_unified_protocol_v1.json"
        payload = json.loads(source_protocol.read_text(encoding="utf-8"))
        payload["cell_variant_id"] = "stair.go2.synthetic_unified_v1"
        payload["protocol_hash"] = canonical_json_hash(
            {key: value for key, value in payload.items() if key != "protocol_hash"}
        )
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            for filename in payload["component_refs"].values():
                shutil.copyfile(source_dir / filename, directory / filename)
            protocol_path = directory / "renamed_go2_protocol.json"
            protocol_path.write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )
            protocol = StairUnifiedProtocol.load(
                protocol_path,
                expected_cell_variant_id="stair.go2.synthetic_unified_v1",
            )
            self.assertEqual(
                protocol.cell_variant_id,
                "stair.go2.synthetic_unified_v1",
            )
            config = StairClimbingTaskConfig.defaults(protocol_path)
            self.assertEqual(config.cell_variant_id, protocol.cell_variant_id)

    def test_score_is_ood_primary_with_id_and_ood_guard_vectors(self) -> None:
        suite = self.config.ordered_evaluation_suite
        success_counts = {
            "c0_id": 3,
            "c1_id": 2,
            "height_ood": 1,
            "tread_ood": 2,
            "joint_ood": 1,
            "phase_composition_ood": 2,
            "c1_conditional_frontier": 1,
            "c2_mechanism_ood": 1,
        }
        progress = {
            "c0_id": 0.8,
            "c1_id": 0.7,
            "height_ood": 0.6,
            "tread_ood": 0.5,
            "joint_ood": 0.4,
            "phase_composition_ood": 0.65,
            "c1_conditional_frontier": 0.55,
            "c2_mechanism_ood": 0.45,
        }
        per_seed = []
        for _ in range(self.config.public_seed_count):
            occurrences: Counter[str] = Counter()
            outcomes = []
            for env_index, cell in enumerate(
                suite.private_cell_assignment("public_dev")
            ):
                occurrence = occurrences[cell.cell_id]
                occurrences[cell.cell_id] += 1
                succeeded = occurrence < success_counts[cell.stratum_id]
                fraction = progress[cell.stratum_id]
                c2_diagnostic_failure = (
                    cell.stratum_id == "c2_mechanism_ood" and not succeeded
                )
                outcomes.append(
                    StairEpisodeOutcome(
                        env_index=env_index,
                        bin_id=cell.cell_id,
                        episode_steps=cell.episode_horizon_control_steps,
                        primary_reason=(
                            "success"
                            if succeeded
                            else "slip"
                            if c2_diagnostic_failure
                            else "timeout"
                        ),
                        success=succeeded,
                        diagnostic_flags=(
                            ("first_valid_top_arrival_achieved",)
                            if succeeded
                            else (
                                "slip_detected",
                                "insufficient_clearance_detected",
                            )
                            if c2_diagnostic_failure
                            else ()
                        ),
                        max_progress_m=fraction * cell.progress_goal_m,
                        progress_fraction=fraction,
                        active_terminal_terms=() if succeeded else ("time_out",),
                        evaluation_cell_id=cell.cell_id,
                    )
                )
            per_seed.append(outcomes)
        summary = build_unified_stair_summary(
            suite=suite,
            semantics=self.config.evaluation_semantics,
            split="public_dev",
            per_seed_outcomes=per_seed,
        )
        canonical = validate_unified_stair_summary(
            suite=suite,
            semantics=self.config.evaluation_semantics,
            split="public_dev",
            summary=summary,
        )
        self.assertAlmostEqual(canonical["j_id"], 0.7916666666666666)
        self.assertAlmostEqual(canonical["j_ood"], 0.47129629629629627)
        self.assertAlmostEqual(canonical["j_primary"], 0.5780864197530864)
        projection = build_unified_score_projection(canonical)
        self.assertEqual(
            set(projection),
            {
                "j_primary",
                "j_ood",
                "c0_id",
                "c1_id",
                "aggregation_mode",
                "qualified_ood_families",
                "objective_family_mask_hash",
            },
        )
        self.assertTrue(
            all(
                projection[name]["score_stats"]["count"]
                == self.config.public_seed_count
                for name in ("j_primary", "j_ood", "c0_id", "c1_id")
            )
        )
        feedback = build_unified_stair_prompt_feedback(
            suite=suite,
            semantics=self.config.evaluation_semantics,
            summary=canonical,
        )
        self.assertEqual(
            set(feedback["generalization_summary"]["stratum_diagnostics"]),
            set(success_counts),
        )
        self.assertIn(
            "primary_failure_breakdown",
            feedback["generalization_summary"]["stratum_diagnostics"][
                "c2_mechanism_ood"
            ],
        )
        c2_diagnostics = feedback["generalization_summary"][
            "stratum_diagnostics"
        ]["c2_mechanism_ood"]
        self.assertAlmostEqual(
            c2_diagnostics["primary_failure_breakdown"]["slip"],
            2.0 / 3.0,
        )
        self.assertEqual(
            c2_diagnostics["primary_failure_breakdown"][
                "insufficient_clearance"
            ],
            0.0,
        )
        self.assertAlmostEqual(
            c2_diagnostics["diagnostic_flag_rates"]["slip_detected"],
            2.0 / 3.0,
        )
        self.assertAlmostEqual(
            c2_diagnostics["diagnostic_flag_rates"][
                "insufficient_clearance_detected"
            ],
            2.0 / 3.0,
        )
        self.assertGreater(
            feedback["diagnostic_flag_rates"][
                "insufficient_clearance_detected"
            ],
            0.0,
        )
        compact = _compact_required_feedback_entry(
            {"feedback": feedback},
            evaluation_profile=None,
        )["feedback"]
        stratum_table = compact["generalization_summary"][
            "stratum_diagnostics"
        ]
        rows_by_stratum = {row[0]: row for row in stratum_table["rows"]}
        self.assertIn("c2_mechanism_ood", rows_by_stratum)
        diagnostic_index = stratum_table["diagnostic_flag_rate_columns"].index(
            "insufficient_clearance_detected"
        )
        self.assertAlmostEqual(
            rows_by_stratum["c2_mechanism_ood"][5][diagnostic_index],
            round(2.0 / 3.0, 6),
        )
        self.assertIn("F_c2", compact["generalization_summary"]["family_scores"])

    def test_compact_stratum_table_distinguishes_truncation_from_true_zero(
        self,
    ) -> None:
        suite = self.config.ordered_evaluation_suite
        per_seed = []
        for _ in range(self.config.public_seed_count):
            occurrences: Counter[str] = Counter()
            outcomes = []
            for env_index, cell in enumerate(
                suite.private_cell_assignment("public_dev")
            ):
                occurrence = occurrences[cell.cell_id]
                occurrences[cell.cell_id] += 1
                if cell.stratum_id == "c2_mechanism_ood":
                    reason = (
                        "fall",
                        "slip",
                        "insufficient_clearance",
                    )[occurrence]
                elif cell.stratum_id == "c0_id":
                    reason = "slip"
                else:
                    reason = "timeout"
                outcomes.append(
                    StairEpisodeOutcome(
                        env_index=env_index,
                        bin_id=cell.cell_id,
                        episode_steps=cell.episode_horizon_control_steps,
                        primary_reason=reason,
                        success=False,
                        diagnostic_flags=(),
                        max_progress_m=0.0,
                        progress_fraction=0.0,
                        active_terminal_terms=(
                            ("time_out",) if reason == "timeout" else ()
                        ),
                        evaluation_cell_id=cell.cell_id,
                    )
                )
            per_seed.append(outcomes)

        summary = build_unified_stair_summary(
            suite=suite,
            semantics=self.config.evaluation_semantics,
            split="public_dev",
            per_seed_outcomes=per_seed,
        )
        feedback = build_unified_stair_prompt_feedback(
            suite=suite,
            semantics=self.config.evaluation_semantics,
            summary=summary,
        )
        raw_c2 = feedback["generalization_summary"]["stratum_diagnostics"][
            "c2_mechanism_ood"
        ]["primary_failure_breakdown"]
        compact_entry = _compact_required_feedback_entry(
            {"feedback": feedback},
            evaluation_profile=None,
        )
        table = compact_entry["feedback"]["generalization_summary"][
            "stratum_diagnostics"
        ]
        rows = {row[0]: row for row in table["rows"]}
        failure_columns = table["primary_failure_rate_columns"]
        c2_rates = dict(
            zip(
                failure_columns,
                rows["c2_mechanism_ood"][4],
                strict=True,
            )
        )
        c0_rates = dict(zip(failure_columns, rows["c0_id"][4], strict=True))

        self.assertAlmostEqual(raw_c2["slip"], 1.0 / 3.0)
        self.assertEqual(
            table["schema_version"],
            "true_rl_agent.task.stair_climbing.stratum_feedback_table.v2",
        )
        self.assertTrue(table["truncated_rates_are_null"])
        self.assertIsNone(c2_rates["slip"])
        self.assertEqual(c2_rates["timeout"], 0.0)
        self.assertEqual(c0_rates["fall"], 0.0)
        self.assertEqual(c0_rates["slip"], 1.0)
        self.assertEqual(
            _compact_latest_feedback_entry(compact_entry),
            compact_entry,
        )

    def test_calibration_aggregates_only_families_present_in_its_cohort(
        self,
    ) -> None:
        suite = self.config.ordered_evaluation_suite
        assignment = suite.private_cell_assignment("calibration")
        per_seed = []
        for _ in range(suite.calibration_seed_count):
            per_seed.append(
                [
                    StairEpisodeOutcome(
                        env_index=env_index,
                        bin_id=cell.cell_id,
                        episode_steps=cell.episode_horizon_control_steps,
                        primary_reason="success",
                        success=True,
                        diagnostic_flags=("first_valid_top_arrival_achieved",),
                        max_progress_m=cell.progress_goal_m,
                        progress_fraction=1.0,
                        active_terminal_terms=(),
                        evaluation_cell_id=cell.cell_id,
                    )
                    for env_index, cell in enumerate(assignment)
                ]
            )

        summary = build_unified_stair_summary(
            suite=suite,
            semantics=self.config.evaluation_semantics,
            split="calibration",
            per_seed_outcomes=per_seed,
        )

        self.assertEqual(set(summary["family_scores"]), {"F_scalar", "F_c2"})
        self.assertEqual(summary["j_ood"], 1.0)

    def test_task_projection_and_prompt_do_not_own_robot_morphology(self) -> None:
        task_text = json.dumps(
            self.config.unified_protocol.task_projection(),
            sort_keys=True,
        )
        for marker in (
            "torso_link",
            "pelvis",
            "ankle_roll_link",
            ".usd",
            "37dof",
        ):
            self.assertNotIn(marker, task_text.lower())
        sections = StairClimbingPromptAdapter(
            self.config,
            resolved_cell_manifest=self.manifest,
        ).build_prompt_capacity_preflight_sections()
        prompt_text = json.dumps(sections, sort_keys=True)
        preflight_entries = sections["public_feedback_summary"][
            "feedback_entries"
        ]
        self.assertEqual(len(preflight_entries), 2)
        for entry in preflight_entries:
            preflight_generalization = entry["feedback"][
                "generalization_summary"
            ]
            self.assertEqual(
                set(preflight_generalization),
                {
                    "family_scores",
                    "j_id",
                    "j_ood",
                    "named_guard_scores",
                    "objective_family_mask_hash",
                    "public_dev_objective_score",
                    "qualified_ood_families",
                    "stratum_diagnostics",
                },
            )
            self.assertEqual(
                {row[0] for row in preflight_generalization["stratum_diagnostics"]["rows"]},
                {
                    cell.stratum_id
                    for cell in self.config.ordered_evaluation_suite.cells
                },
            )
        preflight_generalization = preflight_entries[0]["feedback"][
            "generalization_summary"
        ]
        self.assertEqual(
            preflight_generalization["qualified_ood_families"],
            list(
                self.config.ordered_evaluation_suite.score_contract[
                    "families"
                ]
            ),
        )
        for cell in self.config.ordered_evaluation_suite.cells_for_cohort(
            "sealed_validation"
        ):
            self.assertNotIn(cell.cell_id, prompt_text)
        self.assertNotIn("cohort_geometry_overrides", prompt_text)
        self.assertEqual(
            sections["public_eval_contract"]["primary_objective"],
            self.config.ordered_evaluation_suite.score_contract["j_primary"],
        )
        self.assertEqual(
            sections["public_eval_contract"]["objective_structure"][
                "ood_families"
            ],
            {
                name: list(strata)
                for name, strata in self.config.ordered_evaluation_suite.score_contract[
                    "families"
                ].items()
            },
        )
        self.assertEqual(
            sections["public_eval_contract"]["named_non_regression_guards"],
            ["j_ood", "c0_id", "c1_id"],
        )
        public_cells = self.config.ordered_evaluation_suite.cells_for_cohort(
            "public_dev"
        )
        expected_strata = list(
            dict.fromkeys(cell.stratum_id for cell in public_cells)
        )
        expected_claims = {
            stratum: next(
                cell.claim_kind
                for cell in public_cells
                if cell.stratum_id == stratum
            )
            for stratum in expected_strata
        }
        eval_summary = sections["task"]["fixed_eval_protocol_summary"]
        train_summary = sections["task"]["fixed_train_protocol_summary"]
        self.assertEqual(
            train_summary["c2_training_exposure"],
            "forbidden",
        )
        self.assertEqual(
            train_summary["c1_training_phase_steps"],
            0,
        )
        self.assertEqual(eval_summary["strata"], expected_strata)
        self.assertEqual(eval_summary["cells_per_stratum"], 3)
        self.assertIn("c2_mechanism_ood", eval_summary["strata"])
        claims = sections["public_eval_contract"]["generalization_claims"]
        self.assertEqual(claims, expected_claims)
        self.assertEqual(
            sections["public_eval_contract"]["c2_evidence_scope"],
            "current evaluation-only surface/contact mechanism shift; "
            "zero training exposure; height-scan policy salience not claimed",
        )
        self.assertNotIn("future mechanism", prompt_text.lower())

    def test_no_retired_irregular_cell_switch_remains_in_active_code(self) -> None:
        hits = []
        for root in (ROOT / "src", ROOT / "scripts"):
            for path in root.rglob("*.py"):
                if "is_irregular_cell" in path.read_text(encoding="utf-8"):
                    hits.append(str(path.relative_to(ROOT)))
        self.assertEqual(hits, [])

    def test_acceptance_v6_names_exact_non_regression_guards(self) -> None:
        protocol = self.config.acceptance_protocol
        self.assertTrue(protocol.uses_named_guards)
        self.assertEqual(
            tuple(guard.name for guard in protocol.named_guards),
            ("j_ood", "c0_id", "c1_id"),
        )
        self.assertEqual(
            protocol.score_source, "unified_score.j_primary.score_stats.mean"
        )


if __name__ == "__main__":
    unittest.main()
