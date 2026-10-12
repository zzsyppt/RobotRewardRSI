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
    contiguous_slices,
    interval_rows,
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


























if __name__ == "__main__":
    unittest.main()
