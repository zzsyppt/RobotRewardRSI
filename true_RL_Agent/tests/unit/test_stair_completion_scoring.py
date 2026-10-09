"""Scientific counterexamples for P3, including honest historical replay."""

import copy
import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from true_rl_agent.robots.defaults import build_default_robot_registry
from true_rl_agent.tasks.stair_climbing.calibration import (
    build_stair_calibration_suite, select_stair_calibration_levels, stair_cell_scale_report,
    calibrate_stair_surface_base, freeze_stair_formal_suite,
    refine_stair_calibration_suite,
)
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.episode_evidence import StairEpisodeOutcome
from true_rl_agent.tasks.stair_climbing.ordered_summary import validate_episode_facts
from true_rl_agent.tasks.stair_climbing.evaluation_semantics import StairEvaluationSemantics
from true_rl_agent.tasks.stair_climbing.unified_scoring import (
    build_unified_stair_summary, validate_unified_stair_summary, build_unified_score_projection,
    build_reference_objective_family_mask, validate_objective_family_mask,
)


class StairCompletionScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = StairClimbingTaskConfig.defaults(ROOT / "configs/tasks/stair_unified_protocol_v2.json")
        registry = build_default_robot_registry()
        cls.suite = build_stair_calibration_suite(
            geometry_policy=cls.config.ordered_evaluation_suite.to_dict()["geometry_policy"],
            training_profile=cls.config.unified_training_profile,
            robot_profile=registry.profile("unitree_g1_37dof_isaaclab_v2"),
        )

    def summary(self, *, mode="mixed", split="calibration", success_overrides=None):
        seeds = []
        for seed_index in range(self.suite.seed_count(split)):
            outcomes = []
            for index, cell in enumerate(self.suite.private_cell_assignment(split)):
                success = mode != "fail" and (mode != "mixed" or index % 3 != 2)
                if mode == "c2_floor" and cell.stratum_id == "c2_mechanism_ood":
                    success = False
                success = (success_overrides or {}).get(cell.cell_id, success)
                steps = 100 if mode == "constant" else 100 + (index % 3) * 20 + seed_index * 10
                outcomes.append(StairEpisodeOutcome(
                    env_index=index, bin_id=cell.cell_id, evaluation_cell_id=cell.cell_id,
                    episode_steps=steps, primary_reason="success" if success else "forbidden_contact",
                    success=success, diagnostic_flags=("slip_detected",) if success else (),
                    max_progress_m=cell.progress_goal_m * (1 if success else 0.8),
                    progress_fraction=1.0 if success else 0.8, active_terminal_terms=(),
                ))
            seeds.append(outcomes)
        return build_unified_stair_summary(suite=self.suite, semantics=self.config.evaluation_semantics,
                                           split=split, per_seed_outcomes=seeds)

    def test_episode_facts_follow_the_declared_success_progress(self):
        def fact(reason, step, progress):
            return dict(primary_reason=reason, episode_steps=step, progress_fraction=progress, diagnostic_flags=[])
        outcomes = self.config.evaluation_semantics.to_dict()["episode_outcomes"]
        outcomes["success"]["minimum_progress_fraction"] = 0.9
        semantics = StairEvaluationSemantics.from_task_outcomes(
            task_protocol_hash="a" * 64, task_id="stair_climbing", episode_outcomes=outcomes)
        validate_episode_facts([fact("success", 100, 0.9), fact("forbidden_contact", 1, 1.0)],
                               horizon_steps=600, semantics=semantics)
        with self.assertRaisesRegex(ValueError, "evaluation semantics"):
            validate_episode_facts([fact("success", 100, 0.89)], horizon_steps=600, semantics=semantics)

    def test_raw_episode_facts_reconstruct_scores_and_diagnostics(self):
        summary = self.summary()
        self.assertEqual(summary["ordered_summary"]["total_episode_count"], 1728)
        validated = validate_unified_stair_summary(suite=self.suite, semantics=self.config.evaluation_semantics,
                                                 split="calibration", summary=summary)
        self.assertEqual(summary, validated)
        projection = build_unified_score_projection(summary)
        self.assertIn("joint_ood", projection)
        bad = copy.deepcopy(summary)
        seed = bad["ordered_summary"]["seed_summaries"][0]
        next(iter(seed["episode_facts_by_cell"].values()))[0]["diagnostic_flags"] = []
        with self.assertRaisesRegex(ValueError, "facts differ"):
            validate_unified_stair_summary(suite=self.suite, semantics=self.config.evaluation_semantics,
                                          split="calibration", summary=bad)
        bad = copy.deepcopy(summary)
        bad["seed_summaries"][0]["cell_scores"][next(iter(bad["seed_summaries"][0]["cell_scores"]))] = 1.0
        with self.assertRaisesRegex(ValueError, "derived score"):
            validate_unified_stair_summary(suite=self.suite, semantics=self.config.evaluation_semantics,
                                          split="calibration", summary=bad)

    def test_measured_headroom_rejects_floor_and_uniform_ceiling(self):
        for mode in ("fail", "constant", "varying_success_times"):
            result = select_stair_calibration_levels(suite=self.suite, summary=self.summary(mode=mode))
            self.assertEqual(result["status"], "NO_GO")
        self.assertEqual(select_stair_calibration_levels(suite=self.suite, summary=self.summary())["status"], "GO")

    def test_time_variation_cannot_manufacture_objective_headroom(self):
        constant = self.summary(mode="constant")
        varying = self.summary(mode="varying_success_times")
        self.assertEqual(build_unified_score_projection(constant), build_unified_score_projection(varying))
        self.assertNotEqual(constant["ordered_summary"]["seed_summaries"][0]["episode_facts_by_cell"],
                            varying["ordered_summary"]["seed_summaries"][0]["episode_facts_by_cell"])
        self.assertNotIn("completion_by_cell", constant["seed_summaries"][0])

    def test_new_reference_cannot_drop_a_difficult_family(self):
        for mode in ("mixed", "c2_floor"):
            summary = self.summary(mode=mode, split="public_accept")
            mask = build_reference_objective_family_mask(suite=self.suite, public_accept_summary=summary)
            validate_objective_family_mask(suite=self.suite, mask=mask,
                                           public_accept_summary=summary, require_nonempty=False)
            self.assertEqual(len(mask["qualified_ood_families"]), 3 if mode == "mixed" else 0)

    def test_same_calibration_method_uses_actual_robot_lengths(self):
        go2 = StairClimbingTaskConfig.defaults(ROOT / "configs/tasks/stair_go2_12dof_unified_protocol_v2.json")
        suite = build_stair_calibration_suite(
            geometry_policy=go2.ordered_evaluation_suite.to_dict()["geometry_policy"], training_profile=go2.unified_training_profile,
            robot_profile=build_default_robot_registry().profile("unitree_go2_12dof_isaaclab_v1"),
        )
        g1_report = stair_cell_scale_report(self.suite, "calibration")
        go2_report = stair_cell_scale_report(suite, "calibration")
        self.assertNotEqual(g1_report["scale_context"]["support_length_m"], go2_report["scale_context"]["support_length_m"])
        cell_id = "calibration.c2_mechanism_ood_level_5"
        for candidate, report in ((self.suite, g1_report), (suite, go2_report)):
            cell = candidate.cell(cell_id)
            amplitude = report["cells"][cell_id]["roughness_amplitude_m"]
            self.assertAlmostEqual(amplitude, 0.42 * min(cell.riser_heights_m))
            self.assertLess(2 * amplitude, min(cell.riser_heights_m))
            self.assertGreater(amplitude, 0.015)
        self.assertEqual(len(suite.cells_for_cohort("calibration")), 48)

    def test_control_scores_do_not_dilute_the_objective(self):
        controls = [c for c in self.suite.cells_for_cohort("calibration")
                    if c.stratum_id in {"c0_id", "c1_id", "phase_composition_ood"}]
        high = self.summary(success_overrides={c.cell_id: True for c in controls})
        low = self.summary(success_overrides={c.cell_id: False for c in controls})
        self.assertEqual(high["j_primary"], low["j_primary"])
        self.assertEqual(high["j_primary"], high["j_ood"])
        self.assertGreater(high["named_guard_scores"]["phase_control"], low["named_guard_scores"]["phase_control"])
        self.assertGreater(high["named_guard_scores"]["c0_id"], low["named_guard_scores"]["c0_id"])

    def test_new_robot_factory_does_not_require_a_legacy_suite(self):
        from true_rl_agent.evaluation.binding import canonical_json_hash
        from true_rl_agent.tasks.stair_climbing.unified_training import StairUnifiedTrainingProfile
        profile = build_default_robot_registry().profile("unitree_g1_37dof_isaaclab_v2")
        source_length = profile.runtime_facts["morphology"]["nominal_root_to_support_body_center_z_m"]
        for length in (0.15, 0.42, 0.72, 1.1, 1.6):
            facts = copy.deepcopy(dict(profile.runtime_facts))
            facts["morphology"]["nominal_root_to_support_body_center_z_m"] = length
            facts["runtime_facts_hash"] = canonical_json_hash({k: v for k, v in facts.items() if k != "runtime_facts_hash"})
            robot = replace(profile, runtime_facts=facts, profile_hash=None)
            payload = self.config.unified_training_profile.to_dict()
            for regime, key in (("c0", "riser_range_m"), ("c1", "riser_base_range_m")):
                payload["terrain"][regime][key] = [x * length / source_length for x in payload["terrain"][regime][key]]
            payload["terrain"]["c1"]["riser_offset_m"] *= length / source_length
            payload["profile_hash"] = canonical_json_hash({k: v for k, v in payload.items() if k != "profile_hash"})
            training = StairUnifiedTrainingProfile.from_mapping(payload)
            suite = build_stair_calibration_suite(training_profile=training, robot_profile=robot,
                geometry_policy={"control_dt_s": 0.02, "anchor_horizon_control_steps": 600})
            deepest = suite.cell("calibration.tread_ood_level_5").tread_depths_m[0]
            d0 = float(training.terrain["c0"]["tread_depth_m"])
            self.assertAlmostEqual(d0 - deepest, min(0.5 * length, 0.75 * d0))
            self.assertGreater(deepest, 0)
            self.assertEqual(len(suite.cells_for_cohort("calibration")), 48)

    def test_formal_suite_retains_a_lower_selected_surface_control(self):
        source = self.suite
        for base_id in ("calibration.c0_id_level_1", "calibration.c1_id_level_1"):
            self.suite = source
            id_cells = [c for c in source.cells_for_cohort("calibration") if c.stratum_id in {"c0_id", "c1_id"}]
            summary = self.summary(success_overrides={c.cell_id: c.cell_id == base_id for c in id_cells})
            self.suite, decision = calibrate_stair_surface_base(suite=source, summary=summary,
                training_profile=self.config.unified_training_profile)
            self.assertEqual(decision["flat_control_cell_id"], base_id)
            fresh = self.summary(success_overrides={c.cell_id: True for c in id_cells})
            formal = freeze_stair_formal_suite(calibration_suite=self.suite, calibration_summary=fresh,
                training_profile=self.config.unified_training_profile)
            for cohort in ("public_dev", "public_accept", "sealed_validation"):
                cells = formal.cells_for_cohort(cohort)
                for rough in (c for c in cells if c.stratum_id == "c2_mechanism_ood"):
                    self.assertEqual(rough.riser_heights_m, source.cell(base_id).riser_heights_m)
                    self.assertTrue(any(c.surface_profile.get("kind") == "flat"
                        and c.riser_heights_m == rough.riser_heights_m and c.tread_depths_m == rough.tread_depths_m for c in cells))
                self.assertEqual({c.factor_levels["sequence_pattern"] for c in cells if c.stratum_id == "c1_id"},
                                  set(self.config.unified_training_profile.pattern_ids))

    def test_current_robot_suites_use_fixed_c2_bases_with_held_out_surfaces(self):
        for stem in ("stair", "stair_go2_12dof"):
            config = StairClimbingTaskConfig.defaults(ROOT / f"configs/tasks/{stem}_unified_protocol_v3.json")
            suite = config.ordered_evaluation_suite
            templates = [c.template_id for c in suite.cells_for_cohort("public_dev") if c.stratum_id == "c2_mechanism_ood"]
            for template in templates:
                cells = [suite.cell(cohort + "." + template) for cohort in
                         ("calibration", "public_dev", "public_accept", "sealed_validation")]
                self.assertEqual(len({c.riser_heights_m for c in cells}), 1)
                self.assertEqual(len({c.tread_depths_m for c in cells}), 1)
                self.assertEqual(len({c.surface_profile["amplitude_m"] for c in cells}), 1)
                self.assertEqual(len({c.geometry_digest for c in cells}), 4)

    def test_public_refinement_keeps_coverage_and_sample_count(self):
        overrides = {c.cell_id: True for c in self.suite.cells_for_cohort("calibration")
                     if c.stratum_id in {"c0_id", "c1_id"}}
        refined, decision = refine_stair_calibration_suite(suite=self.suite,
            summary=self.summary(success_overrides=overrides), training_profile=self.config.unified_training_profile)
        self.assertEqual(refined.episodes_per_seed_per_cell * refined.seed_count("calibration"), 36)
        self.assertEqual(len(refined.cells_for_cohort("calibration")), 48)
        self.assertEqual(set(decision["refinements"]), {"height_ood", "tread_ood", "joint_ood", "c1_conditional_frontier", "c2_mechanism_ood"})

    def test_private_suite_round_trip_preserves_geometry_and_score_origin(self):
        from true_rl_agent.tasks.stair_climbing.study_holdout import StairStudyHoldout, FINAL_HOLDOUT_POLICY
        from true_rl_agent.tasks.stair_climbing.unified_evaluation import StairUnifiedEvaluationSuite
        payload = self.suite.to_dict()
        payload["final_holdout_policy"] = FINAL_HOLDOUT_POLICY
        self.suite = source = StairUnifiedEvaluationSuite.freeze(payload, training_profile=self.config.unified_training_profile)
        mask = build_reference_objective_family_mask(suite=source, public_accept_summary=self.summary(split="public_accept"))
        self.suite = actual = StairStudyHoldout("test", source.suite_hash, "a" * 64).bind_suite(
            source, training_profile=self.config.unified_training_profile)
        restored = StairUnifiedEvaluationSuite.from_mapping(actual.to_dict(), training_profile=self.config.unified_training_profile)
        self.assertEqual(restored.cells, actual.cells)
        self.assertEqual(restored.to_dict(), actual.to_dict())
        self.assertEqual(actual.source_suite_hash, source.suite_hash)
        self.assertNotEqual(actual.suite_hash, source.suite_hash)
        self.assertEqual(actual.to_dict()["cohort_hashes"]["sealed_validation"], actual.cohort_hash("sealed_validation"))
        summary = self.summary(split="sealed_validation")
        self.assertEqual(summary["ordered_suite_hash"], actual.suite_hash)
        self.assertEqual(summary["source_suite_hash"], source.suite_hash)
        build_unified_score_projection(summary, objective_family_mask=mask)

    def test_surface_uses_an_empirically_supported_flat_base(self):
        suite, decision = calibrate_stair_surface_base(
            suite=self.suite, summary=self.summary(mode="constant"),
            training_profile=self.config.unified_training_profile,
        )
        control = suite.cell(decision["flat_control_cell_id"])
        self.assertEqual(control.stratum_id, "c0_id")
        rough = suite.cell("calibration.c2_mechanism_ood_level_5")
        self.assertEqual(rough.riser_heights_m, control.riser_heights_m)
        self.assertEqual(rough.tread_depths_m, control.tread_depths_m)
        self.assertNotEqual(rough.geometry_digest, control.geometry_digest)
        with self.assertRaisesRegex(ValueError, "cannot support"):
            calibrate_stair_surface_base(suite=self.suite, summary=self.summary(mode="fail"),
                                        training_profile=self.config.unified_training_profile)

    def test_formal_neighbours_share_only_the_measured_surface_control(self):
        formal = freeze_stair_formal_suite(
            calibration_suite=self.suite, calibration_summary=self.summary(),
            training_profile=self.config.unified_training_profile,
        )
        rough = next(c for c in formal.cells_for_cohort("public_dev") if c.stratum_id == "c2_mechanism_ood")
        control = next(c for c in formal.cells_for_cohort("public_dev")
                       if c.stratum_id in {"c0_id", "c1_id"} and c.riser_heights_m == rough.riser_heights_m
                       and c.tread_depths_m == rough.tread_depths_m)
        digests = [{cell.geometry_digest for cell in formal.cells_for_cohort(cohort)}
                   for cohort in ("calibration", "public_dev", "public_accept", "sealed_validation")]
        for index, left in enumerate(digests):
            for right in digests[index + 1:]:
                self.assertEqual(left & right, {control.geometry_digest})

        for index, cohort in enumerate(("public_dev", "public_accept", "sealed_validation")):
            for rough in formal.cells_for_cohort(cohort):
                if rough.stratum_id != "c2_mechanism_ood":
                    continue
                original = self.suite.cell("calibration." + rough.template_id)
                self.assertEqual(rough.riser_heights_m, original.riser_heights_m)
                self.assertEqual(rough.tread_depths_m, original.tread_depths_m)
                expected_surface = dict(original.surface_profile)
                expected_surface["phase_index"] = (expected_surface["phase_index"] + 2 * (index + 1)) % expected_surface["phase_count"]
                self.assertEqual(dict(rough.surface_profile), expected_surface)
                self.assertAlmostEqual(
                    rough.surface_profile["amplitude_m"] / min(rough.riser_heights_m),
                    original.surface_profile["amplitude_m"] / min(original.riser_heights_m),
                )

    def test_shared_surface_control_does_not_allow_other_repeated_geometries(self):
        from true_rl_agent.tasks.stair_climbing.unified_evaluation import StairUnifiedEvaluationSuite

        for stratum in ("height_ood", "c2_mechanism_ood", "c0_id"):
            payload = self.suite.to_dict()
            template = next(item for item in payload["cell_templates"] if item["stratum_id"] == stratum)
            payload["cohort_geometry_overrides"]["public_accept"][template["template_id"]] = template["geometry"]
            with self.subTest(stratum=stratum), self.assertRaisesRegex(ValueError, "geometries overlap"):
                StairUnifiedEvaluationSuite.freeze(payload, training_profile=self.config.unified_training_profile)

    def test_formal_results_cannot_drive_public_calibration_revision(self):
        for split in ("public_dev", "public_accept", "sealed_validation"):
            for operation in (refine_stair_calibration_suite, freeze_stair_formal_suite):
                arguments = dict(training_profile=self.config.unified_training_profile)
                if operation is refine_stair_calibration_suite:
                    arguments.update(suite=self.suite, summary=self.summary(split=split))
                else:
                    arguments.update(calibration_suite=self.suite, calibration_summary=self.summary(split=split))
                with self.subTest(split=split, operation=operation.__name__), self.assertRaisesRegex(ValueError, "calibration|public pool"):
                    operation(**arguments)

    def test_missing_selected_flat_control_names_the_rough_template(self):
        from true_rl_agent.tasks.stair_climbing.calibration import _set_formal_instances

        payload = self.suite.to_dict()
        rough = next(item for item in payload["calibration_templates"] if item["stratum_id"] == "c2_mechanism_ood")
        with self.assertRaisesRegex(ValueError, rough["template_id"] + " lacks its selected flat control"):
            _set_formal_instances(payload, [rough], training_profile=self.config.unified_training_profile)

    def test_guard_regression_is_rejected_even_when_the_total_improves(self):
        from tests.unit.test_acceptance_contract import _named_guard_record
        from tests.fakes.decision_records import (
            TEST_EVALUATION_BINDING, TEST_EVALUATION_PROTOCOL, TEST_SUBJECT_MANIFEST_HASH,
            TEST_PROGRAM_SOURCE_CHECKSUM, TEST_TRAINING_BINDING_HASH, TEST_CAPABILITY_PROFILE_HASH,
        )
        from true_rl_agent.agent_core.decision import decide_candidate
        from true_rl_agent.candidates.schema import CandidateSurfaceAssessment
        from true_rl_agent.evaluation.binding import EvaluationBindingV1
        from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
        from true_rl_agent.evaluation.public_result import PublicEvalResult

        config = StairClimbingTaskConfig.defaults(ROOT / "configs/tasks/stair_unified_protocol_v3.json")
        protocol = config.acceptance_protocol
        binding = EvaluationBindingV1(
            eval_protocol_hash=TEST_EVALUATION_PROTOCOL.protocol_hash,
            acceptance_protocol_hash=protocol.protocol_hash,
            task_protocol_hash=TEST_EVALUATION_BINDING.task_protocol_hash,
            seed_cohort_hash=TEST_EVALUATION_BINDING.seed_cohort_hash,
        )
        def record(score, value, guard_name="joint_ood"):
            scores = {guard.name: 0.5 for guard in protocol.named_guards}
            scores[guard_name] = value
            result = _named_guard_record(objective_score=score, guard_scores=scores, protocol=protocol, binding=binding)
            def coverage(value):
                if isinstance(value, dict):
                    for key, item in list(value.items()):
                        if key == "count" and item == 3:
                            value[key] = protocol.min_score_count
                        elif key in {"score_samples", "seed_macro_scores"}:
                            value[key] = [item[0]] * protocol.min_score_count
                        else:
                            coverage(item)
            coverage(result)
            return result["public_result"]
        baseline = CurrentBestEvalStats.from_public_result_mapping(
            round_id="joint-guard", candidate_id="candidate-acceptance", public_result=record(0.5, 0.5),
            acceptance_protocol=protocol, subject_manifest_hash=TEST_SUBJECT_MANIFEST_HASH,
            program_source_checksum=TEST_PROGRAM_SOURCE_CHECKSUM, training_binding_hash=TEST_TRAINING_BINDING_HASH,
            capability_profile_hash=TEST_CAPABILITY_PROFILE_HASH, checkpoint_hash="sha256:" + "d" * 64,
            expected_evaluation_binding=binding,
        )
        for guard_name in ("joint_ood", "c0_id", "c1_id", "phase_control"):
            for value, expected in ((0.3, False), (0.5, True)):
                with self.subTest(guard=guard_name, value=value):
                    decision = decide_candidate(surface=CandidateSurfaceAssessment("code_level", True, "code_level_candidate"),
                        public_result=PublicEvalResult(**record(0.7, value, guard_name)),
                        current_best=baseline, acceptance_protocol=protocol)
                    self.assertEqual(decision.accepted, expected)
                    if not expected:
                        self.assertEqual(decision.reason, "named_guard_score_regression")
                        self.assertFalse(decision.comparison["named_guards"][guard_name]["met"])

    def test_new_entrypoints_resolve_the_current_suite_by_default(self):
        import argparse
        from scripts.create_stair_reference_genesis import _build_parser
        from true_rl_agent.tasks.stair_climbing.composition import StairClimbingCompositionProvider
        from true_rl_agent.tasks.stair_climbing.row_catalog import DEFAULT_STAIR_CELL_VARIANT_ID, stair_robot_row

        provider = StairClimbingCompositionProvider()
        parser = argparse.ArgumentParser()
        provider.add_cli_arguments(parser)
        # The shared session CLI requires an explicit cell; composition and
        # the reference command still expose the task's current default.
        self.assertEqual({parser.parse_args([]).cell_variant_id,
                          _build_parser(execution_device="cuda:0").get_default("cell_variant_id")},
                         {DEFAULT_STAIR_CELL_VARIANT_ID})
        defaults = provider.cli_defaults()
        row = stair_robot_row(DEFAULT_STAIR_CELL_VARIANT_ID)
        task = StairClimbingTaskConfig.defaults(ROOT / row.protocol_relative_path)
        self.assertEqual(defaults["cell_variant_id"], task.cell_variant_id)
        self.assertEqual(defaults["runtime_config"], row.protocol_relative_path)
        self.assertEqual(defaults["eval_num_envs"], 288)
        self.assertEqual(task.ordered_evaluation_suite.score_contract["j_primary"], "J_OOD")


if __name__ == "__main__":
    unittest.main()
