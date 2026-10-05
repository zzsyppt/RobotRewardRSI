"""Stair adapter input gates and array-to-episode provenance checks."""
from __future__ import annotations

import copy
from pathlib import Path
import unittest

import numpy as np

# Reuse only the archive builder, by composition. Inheriting its TestCase would
# silently run the old inspector tests again without exercising this adapter.
from tests.unit import stair_archive_fixture as fixture
from true_rl_agent.runner.numeric_evidence import NumericEvidenceError, load_numeric_evidence_manifest
from true_rl_agent.tasks.stair_climbing.trajectory_adapter import _load_contract, iter_stair_episodes


class StairTrajectoryAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.StairArchiveFixture.setUpClass()

    def setUp(self):
        self.fixture = fixture.StairArchiveFixture()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def _episodes(self, mutate=None):
        return list(iter_stair_episodes(self.fixture._archive(mutate), source_id="fixture"))

    def test_all_first_episodes_retain_values_and_cross_chunk_row_locators(self):
        reference = self.fixture._archive()
        episodes = list(iter_stair_episodes(reference, source_id="fixture"))
        self.assertEqual(len(episodes), len(self.fixture.cells))
        self.assertEqual(sum(len(item.steps) for item in episodes), 861)
        episode = episodes[2]
        np.testing.assert_array_equal(episode.steps, [1, 2, 3])
        np.testing.assert_allclose(episode.arrays["base_pos_w"][:, 0], [0.1, 0.2, 0.3])
        np.testing.assert_array_equal(episode.chunk_indices, [0, 0, 1])
        np.testing.assert_array_equal(episode.chunk_rows, [0, 1, 0])
        self.assertEqual(episode.source_chunks, tuple(load_numeric_evidence_manifest(reference)["chunks"]))
        ref = episode.reference(("base_pos_w",))
        self.assertEqual([part["rows_half_open"] for part in ref["chunks"]], [[0, 2], [0, 1]])
        for item in episodes:
            self.assertTrue(item.arrays["valid"].all())
            self.assertNotIn("terrain_env_origins_w", item.arrays)

    def test_bound_entities_and_task_configuration_keep_original_authority(self):
        episodes = self._episodes()
        episode, cell = episodes[0], self.fixture.cells[0]
        self.assertEqual(episode.entities["root"], ("robot_0.root",))
        self.assertEqual(episode.entities["support"], tuple(
            f"robot_0.support.{name}" for name in self.fixture.projection.support_body_names))
        self.assertEqual(episode.entities["forbidden"], tuple(
            f"robot_0.body.{name}" for name in self.fixture.projection.forbidden_contact_body_names))
        self.assertEqual(episode.entities["joints"], self.fixture.projection.joint_names)
        self.assertEqual(episode.task["origin_world"], list(cell.expected_origin_m))
        self.assertEqual(episode.task["goal_progress_m"], cell.progress_goal_m)
        self.assertEqual(episode.task["deadline_s"], cell.episode_horizon_s)
        self.assertEqual(episode.task["contact_threshold_n"], 1.0)
        self.assertEqual(episode.task["outcome"]["primary_reason"], "forbidden_contact")
        self.assertEqual(episode.task["outcome_authority"]["metadata_path"], "seeds[0].outcomes[0]")
        # These synthetic positions do not reach the goal. No success is recomputed.
        self.assertTrue(episodes[2].task["outcome"]["success"])
        static = episode.task["static_refs"]["terrain_env_origins_w"]
        self.assertEqual(static["array_index"], [0])
        self.assertEqual(static["file"], "chunk_000000.npz")
        self.assertEqual(static["sha256"], episode.source_chunks[0]["sha256"])

    def test_self_consistent_suite_must_match_resolved_binding(self):
        from true_rl_agent.tasks.stair_climbing.unified_evaluation import (
            compute_unified_suite_declarations, resolve_stair_suite_cells,
        )
        from true_rl_agent.tasks.stair_climbing.unified_training import StairUnifiedTrainingProfile

        self.assertEqual(len(_load_contract(self.fixture._archive()).cells), len(self.fixture.cells))
        def mutate(identity, context, chunks):
            training = StairUnifiedTrainingProfile.from_mapping(context["training_profile"])
            suite = context["evaluation_suite"]
            cell = next(c for c in suite["cell_templates"] if c["template_id"] == "c1_conditional_frontier_level_5")
            cell["geometry"]["riser_base_m"] += .001
            suite.update(compute_unified_suite_declarations(suite,
                cells=resolve_stair_suite_cells(suite, training_profile=training)))
        with self.assertRaisesRegex(NumericEvidenceError, "resolved evaluation_suite_hash mismatch"):
            _load_contract(self.fixture._archive(mutate))

    def test_history_and_historical_reference_limitations_are_preserved(self):
        episode = self._episodes()[0]
        history = episode.source["contact_force_history"]
        self.assertEqual(history["history_clock"], "contact_sensor_buffer_update_events")
        self.assertIn("not_recorded", history["per_history_slot_timestamps"])
        self.assertIn("no_per_slot_validity_flags", history["reset_history"])
        self.assertEqual(episode.source["support_reference_points"]["position_velocity_derivative_equivalence"], "unavailable")
        self.assertEqual(episode.source["resolved_cell_manifest_hash"], self.fixture.resolved.manifest_hash)
        self.assertEqual(episode.source["runtime_projection"], self.fixture.projection.to_summary_dict())

    def test_inventory_preserves_raw_channels_and_marks_locator_derivations(self):
        episode = self._episodes()[2]
        inventory = episode.source["input_inventory"]
        self.assertEqual(inventory["raw_temporal_channels"], sorted(episode.arrays))
        self.assertIn("episode_length", inventory["raw_temporal_channels"])
        for derived in ("step_index", "source_chunk_index", "source_chunk_row"):
            self.assertNotIn(derived, inventory["raw_temporal_channels"])
        self.assertEqual(inventory["derived_locators"]["control_step"]["source_array"], "episode_length")
        self.assertEqual(inventory["derived_locators"]["source_chunk"]["identity_fields"], ["file", "sha256"])
        self.assertEqual(inventory["derived_locators"]["source_row"]["indexing"], "zero_based_half_open")
        scope = inventory["contact_force_scope"]
        self.assertEqual(scope["support_entity_ids"], list(episode.entities["support"]))
        self.assertEqual(scope["non_support_entity_ids"], ["robot_0.body.base"])
        self.assertEqual(scope["all_nonfoot_body_contact_forces"], "not_recorded")
        self.assertIn("all_nonfoot_body_contact_forces", inventory["unavailable_quantities"])


    def test_same_episode_label_and_checkpoint_in_distinct_sources_never_share_identity(self):
        first_ref = self.fixture._archive()
        first = next(iter_stair_episodes(first_ref, source_id="one"))
        def mutate(identity, context, chunks):
            chunks[0][1]["base_pos_w"][0, 0, 0] = 0.25
        second = self._episodes(mutate)[0]
        self.assertEqual((first.run_index, first.env_index), (second.run_index, second.env_index))
        self.assertEqual(first.source["trial_id"], second.source["trial_id"])
        self.assertEqual(first.source["checkpoint_hash"], second.source["checkpoint_hash"])
        self.assertNotEqual(first.episode_id, second.episode_id)
        alias = next(iter_stair_episodes(first_ref, source_id="another_display_name"))
        self.assertEqual(first.episode_id, alias.episode_id)

    def test_all_recorded_public_seeds_and_global_chunk_indices_are_kept(self):
        def mutate(identity, context, chunks):
            second_seed = copy.deepcopy(context["seeds"][0])
            second_seed.update(run_index=2, runner_seed=19)
            context["seeds"].append(second_seed)
            second_chunks = copy.deepcopy(chunks)
            for metadata, arrays in second_chunks:
                metadata["run_index"] = 2
            chunks.extend(second_chunks)
        episodes = self._episodes(mutate)
        n = len(self.fixture.cells)
        self.assertEqual(len(episodes), n * 2)
        self.assertEqual([episodes[0].source["runner_seed"], episodes[n].source["runner_seed"]], [12, 19])
        np.testing.assert_array_equal(episodes[n + 2].chunk_indices, [2, 2, 3])
        self.assertEqual(episodes[n].task["static_refs"]["terrain_env_origins_w"]["file"], "chunk_000002.npz")
        self.assertEqual(episodes[n].task["outcome_authority"]["metadata_path"], "seeds[1].outcomes[0]")

    def test_entire_source_is_verified_before_first_episode_is_yielded(self):
        reference = self.fixture._archive()
        path = Path(reference["manifest_path"]).parent / "chunk_000001.npz"
        path.write_bytes(path.read_bytes() + b"corruption")
        with self.assertRaisesRegex(NumericEvidenceError, "chunk hash or size"):
            next(iter_stair_episodes(reference, source_id="fixture"))

    def test_late_seed_public_scope_is_checked_before_first_yield(self):
        def mutate(identity, context, chunks):
            seed = copy.deepcopy(context["seeds"][0])
            seed.update(run_index=2, split="sealed_validation")
            context["seeds"].append(seed)
        reference = self.fixture._archive(mutate)
        with self.assertRaisesRegex(NumericEvidenceError, "public_dev"):
            next(iter_stair_episodes(reference, source_id="fixture"))

    def test_manifest_content_must_match_pinned_hash(self):
        reference = self.fixture._archive()
        path = Path(reference["manifest_path"])
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(NumericEvidenceError, "manifest hash"):
            next(iter_stair_episodes(reference, source_id="fixture"))

    def test_nonfinite_components_and_raw_quaternion_are_not_repaired_or_discarded(self):
        def mutate(identity, context, chunks):
            chunks[0][1]["support_lin_vel_w"][0, 2, 0, 0] = np.nan
            chunks[0][1]["support_pos_w"][1, 2, 1, 2] = np.inf
            chunks[0][1]["base_quat_w"][0, 2, 0] = 2.0
            chunks[0][1]["base_quat_w"][1, 2, 0] = 0.0
        episode = self._episodes(mutate)[2]
        self.assertTrue(np.isnan(episode.arrays["support_lin_vel_w"][0, 0, 0]))
        self.assertTrue(np.isinf(episode.arrays["support_pos_w"][1, 1, 2]))
        np.testing.assert_array_equal(episode.arrays["base_quat_w"][:, 0], [2, 0, 1])
        self.assertEqual(len(episode.steps), 3)

    def test_invalid_padding_never_becomes_observation(self):
        def mutate(identity, context, chunks):
            chunks[1][1]["support_pos_w"][0, 0] = np.nan
            chunks[1][1]["episode_length"][0, 0] = -500
            chunks[1][1]["termination_1"][0, 0] = True
        episode = self._episodes(mutate)[0]
        self.assertEqual(len(episode.steps), 1)
        self.assertTrue(np.isfinite(episode.arrays["support_pos_w"]).all())

    def test_nonpublic_or_v1_profile_with_v2_fields_is_rejected(self):
        for field, value in (("split", ["public_dev", "sealed_validation"]),
                             ("profile", "stair.unified_first_episode.v1")):
            with self.subTest(field=field):
                def mutate(identity, context, chunks):
                    if field == "split":
                        identity["splits"] = value
                    else:
                        context["profile"] = value
                with self.assertRaisesRegex(NumericEvidenceError, "public_dev|v1 profile"):
                    self._episodes(mutate)

    def test_task_source_roles_and_units_cannot_drift(self):
        for field in ("task", "resolved_hash", "projection", "roles", "units"):
            with self.subTest(field=field):
                def mutate(identity, context, chunks):
                    if field == "task":
                        identity["task_id"] = "push_recovery"
                    elif field == "resolved_hash":
                        identity["resolved_cell_manifest_hash"] = "0" * 64
                    elif field == "projection":
                        chunks[1][0]["transition_header"]["runtime_projection_hash"] = "0" * 64
                    elif field == "roles":
                        context["runtime_projection"]["resolved_roles"]["support_effector"].reverse()
                    else:
                        context["coordinates"]["forces"] = "body_N"
                with self.assertRaisesRegex(NumericEvidenceError, "mismatch"):
                    self._episodes(mutate)

    def test_gap_overlap_resets_and_valid_reentry_are_rejected(self):
        for field in ("gap", "overlap", "reset", "reentry", "missing_row"):
            with self.subTest(field=field):
                def mutate(identity, context, chunks):
                    if field in {"gap", "overlap"}:
                        chunks[1][0]["step_start"] = 4 if field == "gap" else 2
                    elif field == "reset":
                        chunks[1][1]["episode_length"][0, 2] = 1
                    elif field == "reentry":
                        chunks[1][1]["valid"][0, 0] = True
                    else:
                        chunks[0][1]["valid"][0, 2] = False
                with self.assertRaisesRegex(NumericEvidenceError, "gap or overlap|reset|valid first-episode prefix"):
                    self._episodes(mutate)

    def test_v2_missing_declared_channels_and_contract_fields_are_rejected(self):
        for field in ("channel", "metadata", "history", "header", "seed", "outcome"):
            with self.subTest(field=field):
                def mutate(identity, context, chunks):
                    if field == "channel":
                        del chunks[1][1]["joint_pos"]
                    elif field == "metadata":
                        del context["coordinates"]
                    elif field == "history":
                        del context["contact_force_history"]["per_history_slot_timestamps"]
                    elif field == "header":
                        del chunks[1][0]["transition_header"]["support_count"]
                    elif field == "seed":
                        del context["seeds"][0]["step_dt"]
                    else:
                        del context["seeds"][0]["outcomes"][0]["success"]
                with self.assertRaisesRegex(NumericEvidenceError, "fields"):
                    self._episodes(mutate)

    def test_clock_cannot_drift_from_seed_history_or_frozen_binding(self):
        for field in ("seed", "history", "coordinated", "outcome"):
            with self.subTest(field=field):
                def mutate(identity, context, chunks):
                    if field == "seed":
                        context["seeds"][0]["step_dt"] = 0.04
                    elif field == "history":
                        context["contact_force_history"]["physics_steps_per_control"] = 3
                    elif field == "outcome":
                        context["seeds"][0]["outcomes"][0]["episode_steps"] = 2
                    else:
                        context["seeds"][0]["step_dt"] = 0.04
                        context["contact_force_history"].update(control_dt_s=0.04, physics_steps_per_control=8)
                        for metadata, arrays in chunks:
                            metadata["transition_header"]["contact_force_history"].update(
                                control_dt_s=0.04, physics_steps_per_control=8)
                with self.assertRaisesRegex(NumericEvidenceError, "clock|time mismatch|episode steps"):
                    self._episodes(mutate)

    def test_history_declaration_cannot_upgrade_buffer_to_physical_samples(self):
        def mutate(identity, context, chunks):
            context["contact_force_history"]["history_clock"] = "physics_steps"
        with self.assertRaisesRegex(NumericEvidenceError, "history_clock mismatch"):
            self._episodes(mutate)

    def test_dtype_changes_cannot_silently_downcast_later_chunk_values(self):
        def mutate(identity, context, chunks):
            chunks[1][1]["base_pos_w"] = chunks[1][1]["base_pos_w"].astype(np.float64)
            chunks[1][1]["base_pos_w"][0, 2, 0] = 1.0000000001
        with self.assertRaisesRegex(NumericEvidenceError, "dtype changed"):
            self._episodes(mutate)

    def test_outcome_assignment_and_static_origin_are_checked(self):
        for field in ("cell", "origin", "nonfinite_origin", "terrain"):
            with self.subTest(field=field):
                def mutate(identity, context, chunks):
                    if field == "cell":
                        context["seeds"][0]["outcomes"][0]["evaluation_cell_id"] = "sealed_validation.fake"
                    elif field == "origin":
                        chunks[0][1]["terrain_env_origins_w"][0, 0] += 1
                    elif field == "nonfinite_origin":
                        chunks[0][1]["terrain_env_origins_w"][0, 0] = np.nan
                    else:
                        chunks[0][1]["terrain_types"][0] = 999
                with self.assertRaisesRegex(NumericEvidenceError, "assignment|environment order"):
                    self._episodes(mutate)

    def test_termination_contract_and_pre_reset_boundary_are_checked(self):
        for field in ("names", "order", "wrapper", "union", "early_done"):
            with self.subTest(field=field):
                def mutate(identity, context, chunks):
                    if field == "names":
                        chunks[1][0]["termination_term_order"][1] = "unbound_contact"
                    elif field == "order":
                        chunks[1][0]["termination_term_order"].reverse()
                    elif field == "wrapper":
                        chunks[0][1]["wrapper_done_mask"][0, 0] = False
                    elif field == "union":
                        chunks[0][1]["termination_1"][0, 0] = False
                    else:
                        for name in ("done_mask", "wrapper_done_mask", "termination_0"):
                            chunks[0][1][name][0, 2] = True
                with self.assertRaisesRegex(NumericEvidenceError, "mapping|order mismatch|wrapper|OR|precedes"):
                    self._episodes(mutate)


if __name__ == "__main__":
    unittest.main()
