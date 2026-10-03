"""CPU boundaries: descriptive recovery never substitutes for task success."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from true_rl_agent.evaluation.trajectory_translation.contracts import NumericEpisode
from true_rl_agent.runner.numeric_evidence import NumericEvidenceError, NumericEvidenceWriter, iter_numeric_evidence, load_numeric_evidence_manifest
from true_rl_agent.tasks.push_recovery.trajectory_adapter import iter_push_episodes
from true_rl_agent.tasks.push_recovery.trajectory_measurement import measure_push_episode
from true_rl_agent.tasks.push_recovery.trajectory_card import aggregate_push, build_push_card, project_push_card, json_bytes, build_push_artifacts, load_push_artifacts


def episode(*, force=False, signed=True, env=0, n=10):
    steps = np.arange(1, n + 1)
    mechanism = "world_planar_external_force" if force else "world_planar_velocity_increment"
    params = {"mechanism": mechanism, "resolved_protocol_hash": "protocol",
        "recovery_velocity_mps": .2, "recovery_angular_velocity_radps": 1.,
        "recovery_tilt_rad": .2, "recovery_height_m": .3}
    source = {"manifest_sha256": "a" * 64, "trial_id": "trial", "checkpoint_hash": "policy",
        "acceptance_split": "public_dev", "group_identity": {"mechanism": mechanism, "checkpoint_hash": "policy"},
        "capabilities": {"root_signed_xy": "recorded_world_xy" if signed else "not_recorded"}}
    a = {"pose": np.tile(np.array([.4, 0., 0., .1, 0., .2], dtype=np.float32), (n, 1)),
        "control_step": steps.copy(), "event_steps": np.full((n, 2), -1, dtype=np.int64),
        "event_verified": np.zeros((n, 2), dtype=bool), "failure": np.zeros(n, dtype=bool), "timeout": steps == n}
    a["event_steps"][steps > 2, 0] = 2
    a["event_verified"][steps > 2, 0] = True
    if signed:
        a["planar_motion"] = np.tile(np.array([.1, 0., .2, 0.], dtype=np.float32), (n, 1))
    timing = {"required_dwell_steps": 3, "recovery_window_steps": 5,
              "event_scheduled_steps": [2, 8], "max_episode_steps": n, "step_dt_s": .1}
    event = {"event_index": 0, "injection_verified": True}
    if force:
        a.update(event_ends=np.full((n, 2), -1, dtype=np.int64), actual_substeps=np.zeros((n, 2), dtype=np.int64), unplanned_wrench=np.zeros(n, dtype=bool))
        a["event_ends"][steps >= 4, 0] = 4
        a["actual_substeps"][steps >= 4, 0] = 2
        event.update(start_control_step=2, end_control_step=4, force_world_n=[1., 0., 0.], delivered_impulse_world_ns=[.2, 0., 0.])
    else:
        event.update(control_step=2, observed_delta_v_world_mps=[1., 0.])
    task = {"parameters": params, "time": timing,
        "force": {"expected_substeps": 2, "pulse_steps": 2} if force else None,
        "planned_push_count": 1, "events": [event], "event_metadata_key": "force_events" if force else "push_events",
        "closed": True, "expected_last_step": n, "terminal_label": "success_recovered", "stratum": "low", "control_only": False}
    return NumericEpisode(source, 1, env, steps, .1, a, {"root": ("root",)},
        ({"file": "chunk_000000.npz", "sha256": "b" * 64},), np.zeros(n, dtype=np.int64), np.arange(n), task)


def remove_step(e, step):
    mask = e.steps != step
    e.steps = e.steps[mask]
    e.arrays = {key: value[mask] for key, value in e.arrays.items()}
    e.chunk_indices = e.chunk_indices[mask]
    e.chunk_rows = e.chunk_rows[mask]


class PushTrajectoryMeasurementTests(unittest.TestCase):
    def measured(self, e):
        return measure_push_episode(e)["value"]

    def test_velocity_anchor_row_is_pre_write_and_dwell_equal_is_recovered(self):
        v = self.measured(episode())
        self.assertEqual(v["recoveries"][0]["recovery_step"], 5)
        self.assertEqual(v["events"][0]["windows"]["pre"]["steps_inclusive"], [1, 2])
        self.assertIsNone(v["events"][0]["windows"]["during"])
        self.assertEqual(v["events"][0]["windows"]["post"]["steps_inclusive"], [3, 7])

    def test_force_end_readout_is_during_recovery_starts_next_step(self):
        v = self.measured(episode(force=True))
        self.assertEqual(v["events"][0]["windows"]["during"]["steps_inclusive"], [3, 4])
        self.assertEqual(v["events"][0]["windows"]["post"]["steps_inclusive"], [5, 9])
        self.assertEqual(v["recoveries"][0]["anchor_step"], 4)
        self.assertEqual(v["recoveries"][0]["recovery_step"], 7)

    def test_dwell_one_short_never_counts_and_last_window_step_inclusive(self):
        e = episode()
        e.arrays["pose"][2:5, 3] = 1.
        self.assertEqual(self.measured(e)["recoveries"][0]["status"], "full_window_without_dwell")
        e.arrays["pose"][4, 3] = .1
        self.assertEqual(self.measured(e)["recoveries"][0]["recovery_step"], 7)

    def test_native_dtype_threshold_is_inclusive(self):
        e = episode()
        e.arrays["pose"][:, 3] = np.float32(.2)
        self.assertEqual(self.measured(e)["stability"]["true_samples"], 10)
        e.arrays["pose"][:, 3] = np.nextafter(np.float32(.2), np.float32(1))
        self.assertEqual(self.measured(e)["stability"]["true_samples"], 0)

    def test_latched_recovery_and_terminal_failure_independent(self):
        e = episode()
        e.arrays["pose"][-1, 3] = 1
        e.task["terminal_label"] = "not_recovered"
        v = self.measured(e)
        self.assertTrue(v["any_recovered"])
        self.assertTrue(v["final_unstable_after_recovery"])
        self.assertEqual(v["outcome_class"], "failure")

    def test_transient_instability_after_latch_can_end_stable(self):
        e = episode()
        e.arrays["pose"][6, 3] = 1
        v = self.measured(e)
        self.assertEqual(v["recoveries"][0]["recovery_step"], 5)
        self.assertTrue(v["final"]["stable"])
        self.assertFalse(v["final_unstable_after_recovery"])

    def test_missing_terminal_never_reuses_last_observed_as_task_endpoint(self):
        e = episode()
        remove_step(e, 10)
        v = self.measured(e)
        self.assertIsNone(v["final"]["stable"])
        self.assertTrue(all(x is None for x in v["final"]["pose"].values()))
        self.assertIsNone(v["task_outcome"])
        self.assertIsNone(v["final_unstable_after_recovery"])

    def test_unknown_breaks_dwell_and_false_overrides_unknown(self):
        e = episode()
        e.arrays["pose"][4, :] = np.nan
        v = self.measured(e)
        self.assertEqual(v["recoveries"][0]["status"], "unknown_samples")
        self.assertEqual(v["stability"]["unknown_samples"], 1)
        e.arrays["pose"][4, 3] = 1
        self.assertEqual(self.measured(e)["stability"]["unknown_samples"], 0)

    def test_gap_does_not_join_two_short_dwell_runs(self):
        e = episode()
        remove_step(e, 5)
        self.assertEqual(self.measured(e)["recoveries"][0]["status"], "unknown_samples")
        intervals = self.measured(e)["stability"]["intervals"]
        self.assertEqual(len(intervals), 2)
        self.assertEqual(intervals[0]["right_boundary"], "step_gap")

    def test_window_edge_gap_is_unknown_not_rollout_truncation(self):
        for gap in (3, 7):
            e = episode()
            e.arrays["pose"][:, 3] = 1.
            remove_step(e, gap)
            self.assertEqual(self.measured(e)["recoveries"][0]["status"], "unknown_samples")

    def test_early_task_terminal_and_rollout_truncation_distinct(self):
        e = episode(n=4)
        self.assertEqual(self.measured(e)["recoveries"][0]["status"], "terminal_before_window_end")
        e.task["closed"] = False
        self.assertEqual(self.measured(e)["recoveries"][0]["status"], "rollout_truncated")

    def test_force_completed_but_unverified_is_invalid_not_missing_anchor(self):
        e = episode(force=True)
        e.arrays["event_verified"][:, 0] = False
        self.assertEqual(self.measured(e)["recoveries"][0]["status"], "invalid_injection")

    def test_v1_known_disturbance_cannot_give_root_direction(self):
        v = self.measured(episode(signed=False))
        self.assertIsNone(v["signed_xy"])
        self.assertEqual(v["events"][0]["vector_world"], [1., 0.])
        self.assertIsNone(v["events"][0]["root_along_disturbance"]["mean_velocity_mps"])

    def test_signed_swap_preserves_scalar_norm_but_changes_projection(self):
        a = episode()
        b = deepcopy(a)
        b.arrays["planar_motion"] = b.arrays["planar_motion"][:, [1, 0, 3, 2]]
        x, y = self.measured(a), self.measured(b)
        self.assertEqual(x["pose"], y["pose"])
        self.assertGreater(x["events"][0]["root_along_disturbance"]["mean_velocity_mps"], 0)
        self.assertEqual(y["events"][0]["root_along_disturbance"]["mean_velocity_mps"], 0)

    def test_reference_rows_resolve_after_gap_and_single_sample_span_is_zero(self):
        e = episode(n=1)
        v = self.measured(e)
        self.assertAlmostEqual(v["coverage"]["sample_cell_duration_proxy_s"], .1)
        self.assertEqual(v["coverage"]["observed_span_s"], 0)
        self.assertEqual(v["stability"]["intervals"][0]["sample_count"], 1)


class PushTrajectoryCardTests(unittest.TestCase):
    def test_episode_equal_and_pooled_denominators_differ(self):
        a, b = episode(), episode(env=1)
        for step in range(3, 11):
            remove_step(b, step)
        b.task["expected_last_step"] = 2
        b.arrays["pose"][:, 3] = 1
        c = aggregate_push([measure_push_episode(a), measure_push_episode(b)])["all"]
        self.assertEqual(c["stability"]["episode_equal"]["mean"], .5)
        self.assertEqual(c["stability"]["pooled_true_samples"], 10)
        self.assertEqual(c["stability"]["pooled_known_samples"], 12)

    def test_condition_policy_and_duplicate_source_rejected(self):
        a = measure_push_episode(episode())
        for field in ("mechanism", "checkpoint_hash"):
            b = measure_push_episode(episode(env=1))
            b["value"]["source"]["group_identity"][field] = "different"
            with self.assertRaisesRegex(ValueError, "mismatch"):
                aggregate_push([a, b])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            aggregate_push([a, a])

    def test_zero_success_unknown_final_and_parent_unavailable(self):
        e = episode()
        e.task["terminal_label"] = "not_recovered"
        card = build_push_card([measure_push_episode(e)], "fixture", episodes=[e])
        self.assertEqual(card["cohorts"]["outcome:success"]["total_episodes"], 0)
        self.assertEqual(card["parent_comparison"]["status"], "unavailable")

    def test_public_budget_whole_claims_and_no_private_paths(self):
        e = episode()
        card = build_push_card([measure_push_episode(e)], "fixture", episodes=[e])
        report = project_push_card(card, episodes=[e], source_id="fixture")
        self.assertLessEqual(len(json_bytes(report)), 12288)
        self.assertNotIn("manifest_path", json.dumps(report))
        with self.assertRaisesRegex(ValueError, "mandatory"):
            project_push_card(card, 100, episodes=[e], source_id="fixture")
        self.assertEqual({c["id"] for c in report["claims"]} | {c["id"] for c in card["projection_audit"]["omitted"]}, {c["id"] for c in card["claims"]})

    def test_unrecorded_foot_mechanism_cannot_be_inserted_in_claim(self):
        e = episode(signed=False)
        card = build_push_card([measure_push_episode(e)], "fixture", episodes=[e])
        card["claims"][0]["text"] = "左足持续支撑，力矩奖励造成恢复。"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            project_push_card(card, episodes=[e], source_id="fixture")

    def test_recomputed_sentence_cannot_retain_old_card_identity(self):
        from true_rl_agent.tasks.push_recovery.trajectory_card import _render_claims
        e = episode()
        card = build_push_card([measure_push_episode(e)], "fixture", episodes=[e])
        card["cohorts"]["all"]["pose"]["planar_speed_mps"]["episode_means"]["mean"] = 123456
        card["claims"] = _render_claims(card["domain"], card["cohorts"], card["examples"], card["selection"], card["parameters"], card["timing"])
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            project_push_card(card, episodes=[e], source_id="fixture")

    def test_frozen_witnesses_and_rule_definitions_are_mandatory(self):
        e = episode()
        e.arrays["pose"][:, 3] = 1.
        e.task["terminal_label"] = "not_recovered"
        card = build_push_card([measure_push_episode(e)], "fixture", episodes=[e])
        required = {c["key"] for c in card["claims"] if c["mandatory"]}
        self.assertIn("witness:full_window_without_dwell", required)
        self.assertIn("protocol", required)
        self.assertIn("no_witness", json.dumps(card["public_report"]))

    def test_representatives_are_deterministic_and_source_bound(self):
        episodes = [episode(env=i) for i in range(4)]
        records = [measure_push_episode(e) for e in episodes]
        a = build_push_card(records, "fixture", episodes=episodes)
        b = build_push_card(records[::-1], "fixture", episodes=episodes[::-1])
        self.assertEqual(a["selection"], b["selection"])


class PushTrajectoryArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.unit import test_push_numeric_capture as fixtures
        cls.fixture = fixtures.PushNumericCaptureTests()
        cls.fixture.setUp()
        f = cls.fixture
        capture = f.capture(f.velocity, "adapter-fixture")
        f.velocity_until(f.velocity.env.max_episode_length)
        ref = capture.complete(f.velocity.state.episode_records())
        cls.manifest = load_numeric_evidence_manifest(ref)
        cls.arrays = list(iter_numeric_evidence(ref))
        cls.manifest["identity"].update(job_id="fixture", task_id="unitree_push_recovery", subject_kind="reference")

    @classmethod
    def tearDownClass(cls):
        cls.fixture.doCleanups()

    def archive(self, *, change=None, metadata=None, identity=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        i = deepcopy(self.manifest["identity"])
        m = deepcopy(self.manifest["metadata"])
        if identity:
            identity(i)
        if metadata:
            metadata(m)
        writer = NumericEvidenceWriter(Path(tmp.name) / "archive", i)
        for header, original in self.arrays:
            a = deepcopy(original)
            if change:
                change(a)
            writer.write_chunk(a, header)
        return writer.complete(m), i

    def test_real_capture_profile_velocity_xy(self):
        ref, identity = self.archive()
        episodes = list(iter_push_episodes(ref, expected_identity=identity))
        self.assertEqual(len(episodes), 2)
        self.assertIn("planar_motion", episodes[0].arrays)

    def test_force_v2_signed_xy_uses_same_capability_adapter_and_card(self):
        f = self.fixture
        capture = f.capture(f.force, "force-adapter-fixture")
        f.force.until(f.force.env.max_episode_length)
        old = capture.complete(f.force.state.episode_records())
        m = load_numeric_evidence_manifest(old)
        identity = {**m["identity"], "job_id": "force-fixture", "task_id": "unitree_push_recovery", "subject_kind": "reference"}
        with tempfile.TemporaryDirectory() as tmp:
            writer = NumericEvidenceWriter(Path(tmp) / "archive", identity)
            for header, arrays in iter_numeric_evidence(old):
                writer.write_chunk(arrays, header)
            ref = writer.complete(m["metadata"])
            episodes = list(iter_push_episodes(ref, expected_identity=identity))
            records = [measure_push_episode(e) for e in episodes]
            card = build_push_card(records, "force-v2-fixture", episodes=episodes)
            self.assertEqual(card["domain"]["root_signed_xy"], "recorded_world_xy")
            self.assertEqual(card["domain"]["mechanism"], "world_planar_external_force")
            self.assertIsNotNone(records[0]["value"]["signed_xy"])

    def test_artifact_loader_binds_manifest_source_and_complete_evidence(self):
        import hashlib
        ref, identity = self.archive()
        sources = [{"reference": ref, "identity": identity}]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "card"
            build_push_artifacts(sources, source_id="fixture", output=root)
            manifest = root / "manifest.json"
            checksum = hashlib.sha256(manifest.read_bytes()).hexdigest()
            card = load_push_artifacts(manifest, manifest_sha256=checksum, expected_sources=sources)
            self.assertEqual(card["cohorts"]["all"]["total_episodes"], 2)
            with self.assertRaisesRegex(ValueError, "source binding"):
                load_push_artifacts(manifest, manifest_sha256=checksum, expected_sources=[])
            (root / "public.json").write_text('{}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                load_push_artifacts(manifest, manifest_sha256=checksum, expected_sources=sources)

    def test_condition_mismatch_is_rejected_before_card(self):
        ref, identity = self.archive(change=lambda a: a["conditions"].__setitem__((0, 0, 0), False))
        with self.assertRaisesRegex(NumericEvidenceError, "disagree"):
            list(iter_push_episodes(ref, expected_identity=identity))

    def test_source_identity_sealed_and_protocol_mismatch_rejected(self):
        ref, identity = self.archive()
        other = {**identity, "checkpoint_hash": "other"}
        with self.assertRaisesRegex(NumericEvidenceError, "identity"):
            list(iter_push_episodes(ref, expected_identity=other))
        for mutation in (lambda i: i.update(split="sealed_test"), lambda i: i.update(mechanism="world_planar_external_force")):
            ref, identity = self.archive(identity=mutation)
            with self.assertRaises(NumericEvidenceError):
                list(iter_push_episodes(ref, expected_identity=identity))

    def test_missing_signed_channel_does_not_fall_back(self):
        ref, identity = self.archive(change=lambda a: a.pop("planar_motion"))
        with self.assertRaisesRegex(NumericEvidenceError, "fields or dimensions"):
            list(iter_push_episodes(ref, expected_identity=identity))

    def test_metadata_event_vector_and_source_hash_tamper_rejected(self):
        def wrong(m):
            m["runtime_episode_records"][0]["push_events"][0]["observed_delta_v_world_mps"] = [1, 2, 3]
        ref, identity = self.archive(metadata=wrong)
        with self.assertRaisesRegex(NumericEvidenceError, "velocity XY"):
            list(iter_push_episodes(ref, expected_identity=identity))
        ref["manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(NumericEvidenceError, "hash mismatch"):
            list(iter_push_episodes(ref, expected_identity=identity))


if __name__ == "__main__":
    unittest.main()
