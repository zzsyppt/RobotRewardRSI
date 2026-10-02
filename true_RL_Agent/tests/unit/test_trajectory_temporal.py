"""Independent synthetic gates for source-bound readout temporal evidence.

Expected counts, brackets and identities are specified directly here; these
tests do not import the P0 reference generator or derive expected values by
calling the production operators under test.
"""
from __future__ import annotations

import unittest

import numpy as np

from true_rl_agent.evaluation.trajectory_translation.contracts import (
    NumericEpisode,
    PredicateTrack,
    predicate,
    stable_id,
)
from true_rl_agent.evaluation.trajectory_translation.temporal import (
    before_ranges,
    compose_track,
    conjoin_same_entity,
    contiguous_slices,
    interval_rows,
    not_observed,
    pair_onsets,
    relate_states,
)


FL = "robot_0.support.FL"
FR = "robot_0.support.FR"


def episode(steps=(1, 2, 3, 4), *, dt=0.02, source=None,
            chunk_indices=None, chunk_rows=None, **overrides):
    steps = np.asarray(steps)
    n = len(steps)
    fields = {
        "source": source or {
            "acceptance_split": "public_dev", "manifest_sha256": "a" * 64,
            "trial_id": "trial-1", "checkpoint_hash": "b" * 64,
            "source_id": "r4",
        },
        "run_index": 1,
        "env_index": 54,
        "steps": steps,
        "dt_s": dt,
        "arrays": {"force": np.zeros((n, 2, 3)), "velocity": np.zeros((n, 2, 3))},
        "entities": {"support": (FL, FR)},
        "source_chunks": (
            {"file": "chunk_0.npz", "sha256": "c" * 64},
            {"file": "chunk_1.npz", "sha256": "d" * 64},
        ),
        "chunk_indices": np.zeros(n, dtype=np.int64) if chunk_indices is None else np.asarray(chunk_indices),
        "chunk_rows": np.arange(n) if chunk_rows is None else np.asarray(chunk_rows),
    }
    fields.update(overrides)
    return NumericEpisode(**fields)


def track(values, *, entity=FL, key="marker", channels=("force",), config=None, episode_id=None):
    return PredicateTrack(episode_id or episode().episode_id, key, (entity,), channels, np.asarray(values),
                          config or {"operator": "force_norm_gt", "threshold_n": 1.0})


class NumericEpisodeContractTests(unittest.TestCase):
    def test_stable_ids_preserve_source_trial_and_episode_namespaces(self):
        original = episode()
        same = episode(source={**original.source, "source_id": "renamed-display-label"})
        self.assertEqual(original.episode_id, same.episode_id)
        variants = [
            episode(source={**original.source, "manifest_sha256": "e" * 64}),
            episode(source={**original.source, "trial_id": "trial-2"}),
            episode(run_index=2),
            episode(env_index=60),
        ]
        self.assertEqual(len({original.episode_id, *(x.episode_id for x in variants)}), 5)
        self.assertEqual(stable_id("x", {"a": 1, "b": 2}), stable_id("x", {"b": 2, "a": 1}))

    def test_sealed_split_rejected_even_when_archive_is_frozen(self):
        source = {**episode().source, "acceptance_split": "sealed_validation", "archive_frozen": True}
        with self.assertRaisesRegex(ValueError, "public_dev"):
            episode(source=source)

    def test_control_clock_is_explicit_and_has_no_invented_t0(self):
        sample = episode((1, 2, 4), dt=0.03)
        np.testing.assert_allclose(sample.times, [0.03, 0.06, 0.12])
        np.testing.assert_array_equal(sample.adjacent, [True, False])
        for dt in (0, -0.02, float("nan"), float("inf")):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                episode(dt=dt)

    def test_reset_duplicate_or_invalid_steps_are_rejected(self):
        for steps in ([1, 2, 1], [1, 1], [0, 1], [], [1.0, 2.0], np.array([2, 1], dtype=np.uint64)):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                episode(steps)

    def test_locator_shapes_bounds_and_integer_coordinates_are_required(self):
        for kwargs in (
            {"chunk_indices": [0]}, {"chunk_rows": [0]},
            {"chunk_indices": [0, 0, 0, 2]}, {"chunk_rows": [0, 1, 2, -1]},
            {"chunk_indices": [0, 0, 0, 0.5]}, {"chunk_rows": [0, 1, 2, 3.5]},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                episode(**kwargs)
        with self.assertRaisesRegex(ValueError, "channel length"):
            episode(arrays={"force": np.zeros((3, 2, 3))})
        with self.assertRaisesRegex(ValueError, "duplicate entity"):
            episode(entities={"support": (FL, FL)})

    def test_source_reference_locates_exact_chunk_rows_and_environment(self):
        sample = episode(chunk_indices=[0, 0, 1, 1], chunk_rows=[62, 63, 0, 1])
        ref = sample.reference(("force",), 1, 4)
        self.assertEqual(ref["manifest_sha256"], "a" * 64)
        self.assertEqual(ref["run_index"], 1)
        self.assertEqual(ref["steps_half_open"], [2, 5])
        self.assertEqual(ref["channels"], ["force"])
        self.assertEqual(ref["chunks"], [
            {"file": "chunk_0.npz", "sha256": "c" * 64, "rows_half_open": [63, 64], "env_index": 54},
            {"file": "chunk_1.npz", "sha256": "d" * 64, "rows_half_open": [0, 2], "env_index": 54},
        ])
        gapped = episode((1, 2, 4, 5), chunk_rows=[0, 1, 3, 4]).reference(("force",))
        self.assertEqual([x["rows_half_open"] for x in gapped["chunks"]], [[0, 2], [3, 5]])
        for start, stop in ((-1, 2), (0, 5), (2, 2), (3, 1)):
            with self.subTest(start=start, stop=stop), self.assertRaises(ValueError):
                sample.reference(("force",), start, stop)

    def test_record_identity_changes_with_entity_channel_and_configuration(self):
        sample = episode()
        def record(entity=FL, channels=("force",), config=None):
            return sample.record("marker", "measurement", (entity,), channels, {"count": 1},
                                 unit="1", frame="world", config=config or {"threshold_n": 1.0})
        first = record()
        self.assertEqual(first["record_id"], record()["record_id"])
        records = [first, record(entity=FR), record(channels=("velocity",)), record(config={"threshold_n": 2.0})]
        self.assertEqual(len({r["record_id"] for r in records}), 4)
        self.assertEqual(first["episode_id"], sample.episode_id)
        self.assertEqual(first["entity_ids"], [FL])

    def test_predicates_preserve_unknown_and_reject_invalid_values_before_coercion(self):
        np.testing.assert_array_equal(predicate(np.array([True, False, False]), np.array([True, True, False])),
                                      [1, 0, -1])
        for values in ([2], [-2], [0.5], [float("nan")], np.array([256], dtype=np.int64), [[0, 1]]):
            with self.subTest(values=values), self.assertRaises((ValueError, TypeError)):
                track(values)


class TemporalEvidenceTests(unittest.TestCase):
    def test_all_constant_runs_retain_single_sample_and_both_censor_directions(self):
        sample = episode(tuple(range(1, 9)))
        rows = interval_rows(sample, track([1, 1, 0, 1, 0, 1, 1, 1]))
        active = [r for r in rows if r["state"] == "true"]
        self.assertEqual([r["steps_half_open"] for r in active], [[1, 3], [4, 5], [6, 9]])
        left, single, right = active
        self.assertTrue(left["left_censored"])
        self.assertIsNone(left["onset_range_s"])
        np.testing.assert_allclose(left["end_range_s"], [0.04, 0.06])
        self.assertEqual(single["sample_count"], 1)
        self.assertEqual(single["observed_span_s"], 0)
        self.assertEqual(single["sample_cell_duration_proxy_s"], 0.02)
        self.assertTrue(single["complete_run"])
        np.testing.assert_allclose(single["onset_range_s"], [0.06, 0.08])
        np.testing.assert_allclose(single["end_range_s"], [0.08, 0.10])
        self.assertEqual(single["confirmation_time_s"], 0.08)
        self.assertTrue(right["right_censored"])
        self.assertIsNone(right["end_range_s"])
        self.assertAlmostEqual(right["observed_span_s"], 0.04)
        self.assertAlmostEqual(right["sample_cell_duration_proxy_s"], 0.06)
        self.assertTrue(all(r["physical_continuity"] == "unknown" for r in rows))

    def test_same_occupancy_different_order_has_different_runs_and_events(self):
        sample = episode()
        blocked, alternating = track([0, 0, 1, 1]), track([0, 1, 0, 1])
        self.assertEqual(int((blocked.values == 1).sum()), int((alternating.values == 1).sum()))
        a, b = compose_track(sample, blocked), compose_track(sample, alternating)
        self.assertEqual(sum(r["record_type"] == "event" for r in a), 1)
        self.assertEqual(sum(r["record_type"] == "event" for r in b), 3)
        self.assertEqual([r["value"]["sample_count"] for r in a if r["record_type"] == "state_interval"], [2, 2])
        self.assertEqual([r["value"]["sample_count"] for r in b if r["record_type"] == "state_interval"], [1, 1, 1, 1])

    def test_one_sample_contact_interrupts_absence_without_debounce(self):
        sample = episode((1, 2, 3, 4, 5))
        markers = track([0, 0, 1, 0, 0])
        self.assertEqual(not_observed(sample, markers, 0, 5), "false")
        records = compose_track(sample, markers)
        events = [r["value"] for r in records if r["record_type"] == "event"]
        self.assertEqual([(e["step"], e["before"], e["after"]) for e in events], [(3, False, True), (4, True, False)])
        self.assertTrue(all(e["physical_event_range_s"] is None for e in events))
        off = [r["value"] for r in records if r["record_type"] == "state_interval" and r["value"]["state"] == "false"]
        self.assertEqual([r["steps_half_open"] for r in off], [[1, 3], [4, 6]])

    def test_unknown_and_step_gap_split_runs_without_invented_transitions(self):
        sample = episode((1, 2, 3, 5, 6))
        evidence = track([1, -1, 1, 1, 0])
        rows = interval_rows(sample, evidence)
        self.assertEqual([r["state"] for r in rows], ["true", "unknown", "true", "true", "false"])
        self.assertEqual(rows[0]["right_boundary"], "unknown_sample")
        self.assertEqual(rows[2]["left_boundary"], "unknown_sample")
        self.assertEqual(rows[2]["right_boundary"], "step_gap")
        self.assertEqual(rows[3]["left_boundary"], "step_gap")
        self.assertIsNone(rows[2]["onset_range_s"])
        self.assertIsNone(rows[2]["end_range_s"])
        self.assertIsNone(rows[3]["onset_range_s"])
        events = [r["value"]["step"] for r in compose_track(sample, evidence) if r["record_type"] == "event"]
        self.assertEqual(events, [6])
        self.assertEqual(contiguous_slices(sample.steps, evidence.values >= 0), [(0, 1), (2, 3), (3, 5)])

    def test_absence_is_three_valued_and_counterevidence_is_decisive(self):
        sample = episode()
        self.assertEqual(not_observed(sample, track([0, 0, 0, 0]), 0, 4), "true")
        self.assertEqual(not_observed(sample, track([0, -1, 0, 0]), 0, 4), "unknown")
        self.assertEqual(not_observed(sample, track([0, -1, 1, 0]), 0, 4), "false")
        self.assertEqual(not_observed(episode((1, 2, 4, 5)), track([0, 0, 0, 0]), 0, 4), "unknown")
        self.assertEqual(not_observed(episode((1, 2, 4, 5)), track([0, 0, 0, 0]), 0, 2), "true")
        with self.assertRaises(ValueError):
            not_observed(sample, track([0, 0, 0, 0]), 0, 5)

    def test_same_foot_and_alternating_feet_cannot_be_merged(self):
        sample = episode()
        alternating_left = track([1, 0, 1, 0], entity=FL, key="off")
        alternating_right = track([0, 1, 0, 1], entity=FR, key="off")
        with self.assertRaisesRegex(ValueError, "identical entities"):
            conjoin_same_entity(alternating_left, alternating_right, "continuous_off")
        left_true = [r for r in interval_rows(sample, alternating_left) if r["state"] == "true"]
        self.assertEqual([r["sample_count"] for r in left_true], [1, 1])
        same_foot = interval_rows(sample, track([1, 1, 1, 1], entity=FL, key="off"))
        self.assertEqual(same_foot[0]["sample_count"], 4)
        self.assertTrue(same_foot[0]["left_censored"] and same_foot[0]["right_censored"])

    def test_conjunction_uses_three_valued_logic_and_preserves_operands(self):
        left = track([1, 0, -1, 1], key="off")
        right = track([1, 1, 0, -1], key="low_speed", channels=("velocity",))
        result = conjoin_same_entity(left, right, "low_speed_off")
        np.testing.assert_array_equal(result.values, [1, 0, 0, -1])
        self.assertEqual(result.entity_ids, (FL,))
        self.assertEqual(result.channels, ("force", "velocity"))
        self.assertEqual(result.config["operands"], [left.config, right.config])

    def test_event_brackets_overlap_or_touch_without_definite_before(self):
        cases = (
            ([0.02, 0.04], [0.06, 0.08], "true"),
            ([0.06, 0.08], [0.02, 0.04], "false"),
            ([0.02, 0.04], [0.02, 0.04], "unknown"),
            ([0.02, 0.04], [0.04, 0.06], "unknown"),
            ([0.02, 0.06], [0.04, 0.08], "unknown"),
            (None, [0.04, 0.06], "unknown"),
        )
        for left, right, expected in cases:
            with self.subTest(left=left, right=right):
                self.assertEqual(before_ranges(left, right), expected)

    def test_conjunction_retains_window_operands_for_state_and_transition(self):
        sample = episode(tuple(range(1, 33)))
        window = track([-1] * 25 + [1] * 7, config={"lookback_samples": 25})
        other = track([1] * 25 + [0] + [1] * 6, channels=("velocity",))
        combined = conjoin_same_entity(window, other, "window_and_marker")
        nested = conjoin_same_entity(combined, track([1] * 32), "nested")
        records = compose_track(sample, nested)
        active = next(r for r in records if r["record_type"] == "state_interval"
                      and r["value"]["state"] == "true")
        event = next(r for r in records if r["record_type"] == "event")
        self.assertEqual(active["value"]["steps_half_open"], [27, 33])
        # The event compares step 26 with 27: the earlier window needs step 1.
        self.assertEqual(active["evidence_ref"]["steps_half_open"], [1, 33])
        self.assertEqual(event["value"]["step"], 27)
        self.assertEqual(event["evidence_ref"]["steps_half_open"], [1, 28])

    def test_nearest_onset_equal_distance_chooses_earlier_without_deleting_events(self):
        sample = episode(tuple(range(1, 8)))
        left = track([0, 0, 0, 1, 0, 0, 0], entity=FL)
        right = track([0, 1, 0, 0, 0, 1, 0], entity=FR)
        pairs = pair_onsets(sample, left, right)
        self.assertEqual(len(pairs), 1)
        value = pairs[0]["value"]
        self.assertEqual((value["reference_step"], value["target_step"]), (4, 2))
        self.assertAlmostEqual(value["delta_confirmation_s"], -0.04)
        np.testing.assert_allclose(value["delta_range_s"], [-0.06, -0.02])
        self.assertEqual(value["physical_order"], "unknown")
        self.assertEqual(pairs[0]["entity_ids"], [FL, FR])
        right_events = [r["value"]["step"] for r in compose_track(sample, right)
                        if r["record_type"] == "event" and r["value"]["after"]]
        self.assertEqual(right_events, [2, 6])

    def test_onset_pairing_can_reuse_target_and_keep_unpaired_reference(self):
        sample = episode(tuple(range(1, 8)))
        left = track([0, 1, 0, 1, 0, 0, 0], entity=FL)
        right = track([0, 0, 0, 0, 0, 1, 0], entity=FR)
        paired = pair_onsets(sample, left, right)
        self.assertEqual([(r["value"]["reference_step"], r["value"]["target_step"]) for r in paired], [(2, 6), (4, 6)])
        self.assertEqual(len({r["record_id"] for r in paired}), 2)
        unpaired = pair_onsets(sample, left, track([0] * 7, entity=FR))
        self.assertEqual(len(unpaired), 2)
        for item in unpaired:
            value = item["value"]
            self.assertIsNone(value["target_step"])
            self.assertIsNone(value["delta_range_s"])
            self.assertEqual(value["before"], "unknown")
            self.assertEqual(value["status"], "unpaired_no_target_in_joint_coverage")

    def test_onset_pairing_does_not_cross_unknown_or_missing_steps(self):
        for sample, left_values, right_values in (
            (episode((1, 2, 3, 4, 5)), [0, 1, 0, 0, 0], [0, 0, -1, 0, 1]),
            (episode((1, 2, 4, 5)), [0, 1, 0, 0], [0, 0, 0, 1]),
        ):
            with self.subTest(steps=sample.steps.tolist()):
                pairs = pair_onsets(sample, track(left_values, entity=FL), track(right_values, entity=FR))
                self.assertEqual(len(pairs), 1)
                self.assertEqual(pairs[0]["value"]["reference_step"], 2)
                self.assertIsNone(pairs[0]["value"]["target_step"])

    def test_same_step_onsets_do_not_claim_physical_or_strict_readout_order(self):
        sample = episode((1, 2, 3))
        pair = pair_onsets(sample, track([0, 1, 0], entity=FL), track([0, 1, 0], entity=FR))[0]
        self.assertEqual(pair["value"]["delta_confirmation_s"], 0)
        self.assertEqual(pair["value"]["before"], "unknown")
        self.assertEqual(pair["value"]["physical_order"], "unknown")
        np.testing.assert_allclose(pair["value"]["delta_range_s"], [-0.02, 0.02])

    def test_two_sample_state_records_have_unique_ids_despite_shared_brackets(self):
        sample = episode((1, 2))
        records = compose_track(sample, track([0, 1]))
        self.assertEqual(len(records), 3)
        self.assertEqual(len({r["record_id"] for r in records}), 3)
        states = [r for r in records if r["record_type"] == "state_interval"]
        self.assertEqual([r["value"]["steps_half_open"] for r in states], [[1, 2], [2, 3]])
        self.assertEqual([r["evidence_ref"]["steps_half_open"] for r in states], [[1, 3], [1, 3]])
        self.assertEqual(records, compose_track(sample, track([0, 1])))

    def test_track_length_mismatch_is_rejected_by_public_operators(self):
        sample = episode()
        short = track([0, 0])
        with self.assertRaises(ValueError):
            interval_rows(sample, short)
        with self.assertRaises(ValueError):
            not_observed(sample, short, 0, 4)
        with self.assertRaises(ValueError):
            pair_onsets(sample, short, track([0, 0], entity=FR))

    def test_same_named_tracks_from_another_source_cannot_be_composed(self):
        sample = episode()
        other = episode(source={**sample.source, "manifest_sha256": "f" * 64})
        local = track([0, 1, 1, 0], episode_id=sample.episode_id)
        foreign = track([0, 1, 1, 0], episode_id=other.episode_id)
        operations = (
            lambda: interval_rows(sample, foreign),
            lambda: compose_track(sample, foreign),
            lambda: not_observed(sample, foreign, 0, 4),
            lambda: pair_onsets(sample, local, foreign),
            lambda: conjoin_same_entity(local, foreign, "same_foot"),
        )
        for operation in operations:
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                operation()

    def test_state_relations_keep_operand_ids_and_readout_limits(self):
        sample = episode(tuple(range(1, 7)))
        def active(values, key):
            return next(r for r in compose_track(sample, track(values, key=key))
                        if r["record_type"] == "state_interval" and r["value"]["state"] == "true")
        earlier = active([0, 1, 1, 0, 0, 0], "earlier")
        later = active([0, 0, 0, 1, 1, 0], "later")
        outer = active([0, 1, 1, 1, 1, 0], "outer")
        same_boundary = active([0, 1, 0, 0, 0, 0], "same_boundary")
        first = relate_states(sample, earlier, later, "onset_before")
        self.assertEqual(first["value"]["status"], "true")
        self.assertEqual(first["value"]["operand_record_ids"], [earlier["record_id"], later["record_id"]])
        self.assertEqual(first["value"]["physical_relation"], "unknown")
        self.assertEqual(relate_states(sample, earlier, same_boundary, "onset_before")["value"]["status"], "unknown")
        self.assertEqual(relate_states(sample, earlier, later, "overlap")["value"]["status"], "false")
        self.assertEqual(relate_states(sample, earlier, outer, "overlap")["value"]["status"], "true")
        self.assertEqual(relate_states(sample, earlier, outer, "during_observed_records")["value"]["status"], "true")
        self.assertEqual(relate_states(sample, outer, earlier, "during_observed_records")["value"]["status"], "false")
        with self.assertRaisesRegex(ValueError, "unsupported"):
            relate_states(sample, earlier, later, "causes")

    def test_state_relations_reject_foreign_episode_nonstate_or_false_interval(self):
        sample = episode()
        records = compose_track(sample, track([0, 1, 1, 0]))
        active = next(r for r in records if r["record_type"] == "state_interval" and r["value"]["state"] == "true")
        inactive = next(r for r in records if r["record_type"] == "state_interval" and r["value"]["state"] == "false")
        event = next(r for r in records if r["record_type"] == "event")
        other = episode(source={**sample.source, "manifest_sha256": "f" * 64})
        foreign = next(r for r in compose_track(other, track([0, 1, 1, 0], episode_id=other.episode_id))
                       if r["record_type"] == "state_interval" and r["value"]["state"] == "true")
        for invalid in (inactive, event, foreign):
            with self.subTest(record_type=invalid["record_type"]), self.assertRaises(ValueError):
                relate_states(sample, active, invalid, "overlap")


if __name__ == "__main__":
    unittest.main()
