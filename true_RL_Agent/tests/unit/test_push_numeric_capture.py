"""CPU tensor/archive checks; these are not simulator or policy qualification."""
from __future__ import annotations

import importlib.util
import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from true_rl_agent.runner.numeric_evidence import NumericEvidenceError, NumericEvidenceWriter, iter_numeric_evidence, load_numeric_evidence_manifest
from true_rl_agent.tasks.push_recovery.numeric_capture import PushNumericCapture, PushNumericCaptureOptions, recompute_push_numeric_evidence
from true_rl_agent.tasks.push_recovery.numeric_capture import (
    PLANAR_MOTION_FIELDS, PUSH_NUMERIC_SCHEMA, PUSH_NUMERIC_V1_SCHEMA,
    push_numeric_capture_options_from_args,
)


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires CPU torch")
class PushNumericCaptureTests(unittest.TestCase):
    def setUp(self):
        from tests.unit.test_push_force_runtime import ForceRuntimeTests
        from tests.unit.test_push_physical_protocol import PushPhysicalTensorTests
        self.force = ForceRuntimeTests()
        self.force.setUp()
        self.velocity = PushPhysicalTensorTests()
        self.velocity.setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def capture(self, fixture, name, **kwargs):
        options = PushNumericCaptureOptions(Path(self.tmp.name) / name,
            {"fixture": "CPU tensor, not simulator evidence", "checkpoint_hash": "a" * 64}, chunk_steps=17, **kwargs)
        capture = PushNumericCapture(options, fixture.state, split_name="public_dev", run_index=1)
        fixture.state.numeric_capture = capture
        return capture

    def velocity_until(self, stop, changes=None):
        f = self.velocity
        while int(f.env.episode_length_buf[0]) < stop:
            f.state.inject(None)
            f.data.root_lin_vel_w.zero_()  # The fixture controls physics outcomes.
            f.env.episode_length_buf += 1
            if changes:
                changes(int(f.env.episode_length_buf[0]))
            f.state.observe()

    def rewrite_archive(self, reference, name, *, legacy=False, transform=None):
        manifest = load_numeric_evidence_manifest(reference)
        identity, metadata = copy.deepcopy(manifest["identity"]), copy.deepcopy(manifest["metadata"])
        if legacy:
            identity["task_numeric_schema"] = metadata["schema_version"] = PUSH_NUMERIC_V1_SCHEMA
            metadata.pop("planar_motion_fields")
            metadata.pop("direction_status")
            metadata["coordinates"].pop("planar_motion")
        writer = NumericEvidenceWriter(Path(self.tmp.name) / name, identity, 16 * 1024**2)
        for chunk, arrays in iter_numeric_evidence(reference):
            if legacy:
                arrays.pop("planar_motion")
            if transform is not None:
                transform(arrays)
            writer.write_chunk(arrays, chunk)
        return writer.complete(metadata)

    def test_signed_world_motion_distinguishes_equal_norms_and_uses_reset_origin(self):
        import numpy as np
        f = self.velocity
        origin = f.torch.tensor([[4., -3.], [-7., 2.]])
        f.data.root_pos_w[:, :2] = origin
        f.state.reset(None)
        capture = self.capture(f, "signed-motion")
        velocity = f.torch.tensor([[-.3, .4], [.3, -.4]])
        displacement = f.torch.tensor([[-.25, .5], [.25, -.5]])
        def changes(step):
            f.data.root_lin_vel_w[:, :2] = velocity
            f.data.root_pos_w[:, :2] = origin + displacement
        self.velocity_until(1, changes)
        reference = capture.complete(f.state.episode_records())
        manifest = load_numeric_evidence_manifest(reference)
        self.assertEqual(manifest["metadata"]["schema_version"], PUSH_NUMERIC_SCHEMA)
        self.assertEqual(manifest["metadata"]["planar_motion_fields"], list(PLANAR_MOTION_FIELDS))
        _, arrays = next(iter_numeric_evidence(reference))
        self.assertEqual(arrays["planar_motion"].dtype.name, "float32")
        np.testing.assert_array_equal(arrays["planar_motion"][0, :, :2], velocity.numpy())
        np.testing.assert_array_equal(arrays["planar_motion"][0, :, 2:], displacement.numpy())
        np.testing.assert_array_equal(arrays["pose"][0, 0, (3, 5)], arrays["pose"][0, 1, (3, 5)])
        result = recompute_push_numeric_evidence(reference)
        self.assertTrue(result["matches_runtime"], result)
        self.assertTrue(result["checks"]["planar_motion_norms"])
        self.assertEqual(result["direction_status"], "recorded_world_xy")

        legacy = self.rewrite_archive(reference, "legacy-motion", legacy=True)
        old_result = recompute_push_numeric_evidence(legacy)
        self.assertTrue(old_result["matches_runtime"], old_result)
        self.assertEqual(old_result["direction_status"], "not_collected_in_v1")
        self.assertNotIn("planar_motion_norms", old_result["checks"])
        self.assertEqual(old_result["per_episode"], result["per_episode"])
        self.assertNotIn("planar_motion", next(iter_numeric_evidence(legacy))[1])

        def corrupt(arrays):
            arrays["planar_motion"][0, 0, 0] += 1.
        corrupted = self.rewrite_archive(reference, "wrong-motion", transform=corrupt)
        bad = recompute_push_numeric_evidence(corrupted)
        self.assertFalse(bad["checks"]["planar_motion_norms"])
        self.assertFalse(bad["matches_runtime"])

    def test_v2_requires_motion_fields_instead_of_silently_falling_back_to_v1(self):
        capture = self.capture(self.velocity, "missing-motion")
        self.velocity_until(1)
        ref = capture.complete(self.velocity.state.episode_records())
        broken = self.rewrite_archive(ref, "missing-motion-copy", transform=lambda a: a.pop("planar_motion"))
        with self.assertRaisesRegex(NumericEvidenceError, "fields or dimensions"):
            recompute_push_numeric_evidence(broken)

    def test_velocity_events_all_conditions_interruption_and_final_instability(self):
        f = self.velocity
        # Exercise repeated-event identities without changing the task's equations.
        f.state.planned[:] = 2
        capture = self.capture(f, "velocity")
        first = f.state.event_times[0]
        def changes(step):
            f.data.root_ang_vel_w.zero_()
            if step == first + 3:
                f.data.root_lin_vel_w[:, 0] = 1
                f.data.root_ang_vel_w[:, 0] = 10
            if step == f.env.max_episode_length:
                f.data.root_lin_vel_w[0, 0] = 1
        self.velocity_until(f.env.max_episode_length, changes)
        original = f.state.episode_records()
        reference = capture.complete(original)
        result = recompute_push_numeric_evidence(reference)
        self.assertTrue(result["matches_runtime"], result)
        failed, success = result["per_episode"]
        self.assertEqual(failed["terminal_label"], "not_recovered")
        self.assertTrue(failed["final_unstable_after_recovery"])
        self.assertEqual(success["terminal_label"], "success_recovered")
        for record in result["per_episode"]:
            self.assertEqual([r["identity"] for r in record["recoveries"]], ["event_0", "event_1"])
            interruption = record["recoveries"][0]["dwell_interruptions"][0]
            self.assertEqual(interruption["failed_conditions"], ["planar_speed", "angular_speed"])
            self.assertEqual(interruption["after_anchor_steps"], 3)
            self.assertEqual(interruption["preceding_dwell_steps"], 2)
        manifest = load_numeric_evidence_manifest(reference)
        self.assertGreater(len(manifest["chunks"]), 1)
        self.assertEqual(manifest["metadata"]["contact_status"], "not_collected")
        self.assertEqual(manifest["metadata"]["media_status"], "not_collected_by_numeric_profile")
        for _, arrays in iter_numeric_evidence(reference):
            self.assertEqual(arrays["pose"].dtype.name, "float32")
            self.assertEqual(arrays["control_step"].dtype.name, "int64")
        with self.assertRaisesRegex(NumericEvidenceError, "already published"):
            capture.complete(original)

    def test_force_sequence_not_per_pulse_inclusive_window_and_final_frame(self):
        f = self.force
        capture = self.capture(f, "force")
        last_end = f.state.event_times[1] + f.state.pulse_steps
        f.data.root_lin_vel_w[2:, 0] = 1
        f.until(last_end + f.state.window - f.state.required_dwell)
        f.data.root_lin_vel_w[3, 0] = 0
        f.until(last_end + f.state.window)
        f.data.root_lin_vel_w.zero_()
        f.until(f.env.max_episode_length)
        result = recompute_push_numeric_evidence(capture.complete(f.state.episode_records()))
        self.assertTrue(result["matches_runtime"], result)
        missed, recovered = result["per_episode"][2:]
        self.assertEqual(missed["terminal_label"], "not_recovered")
        self.assertEqual(missed["recoveries"][0]["coverage"], "full_window_without_dwell")
        self.assertEqual(recovered["recoveries"][0]["identity"], "sequence")
        self.assertEqual(recovered["recoveries"][0]["anchor_control_step"], last_end)
        self.assertEqual(recovered["recoveries"][0]["recovery_control_step"], last_end + f.state.window)
        self.assertEqual(recovered["valid_steps"], f.env.max_episode_length)

    def test_force_partial_pulse_fall_and_closed_padding_do_not_cross_reset(self):
        f = self.force
        capture = self.capture(f, "force-fall")
        f.until(f.state.event_times[0])
        f.data.root_pos_w[0, 2] = .1
        f.data.root_pos_w[0, :2] = f.state.initial_xy[0] + f.torch.tensor([-.5, .25])
        f.data.root_lin_vel_w[0, :2] = f.torch.tensor([-.4, .3])
        f.step()
        frozen = f.state.episode_records()[0]
        f.data.root_pos_w[0, 2] = .4
        f.state.reset(f.torch.tensor([0]))
        f.until(f.env.max_episode_length)
        self.assertEqual(frozen, f.state.episode_records()[0])
        reference = capture.complete(f.state.episode_records())
        result = recompute_push_numeric_evidence(reference)
        self.assertTrue(result["matches_runtime"], result)
        fallen = result["per_episode"][0]
        self.assertEqual(fallen["terminal_label"], "height_collapse")
        self.assertEqual(fallen["snapshot_metrics"]["invalid_injection_count"], 0)
        self.assertEqual(fallen["valid_steps"], f.state.event_times[0] + 1)
        self.assertEqual(fallen["recoveries"][0]["coverage"], "anchor_not_observed")
        self.assertIsNone(fallen["recoveries"][0]["anchor_control_step"])
        valid_motion = []
        for _, arrays in iter_numeric_evidence(reference):
            valid_motion.extend(arrays["planar_motion"][:, 0][arrays["valid"][:, 0]].tolist())
        self.assertEqual(len(valid_motion), fallen["valid_steps"])
        self.assertEqual(valid_motion[-1], f.torch.tensor([-.4, .3, -.5, .25]).tolist())

    def test_truncated_capture_publishes_data_but_never_terminal_success(self):
        f = self.force
        capture = self.capture(f, "truncated")
        f.until(f.state.event_times[1] + f.state.pulse_steps + f.state.required_dwell)
        result = recompute_push_numeric_evidence(capture.complete(f.state.episode_records()))
        self.assertTrue(result["matches_runtime"], result)
        self.assertEqual(result["episode_coverage"], "truncated")
        self.assertTrue(all(row["terminal_label"] is None for row in result["per_episode"]))
        self.assertTrue(all(row["snapshot_metrics"]["success"] for row in result["per_episode"]))

    def test_rollout_explicit_integration_and_capture_off_are_equivalent(self):
        import torch
        from true_rl_agent.tasks.push_recovery.rollout import run_physical_push_rollout
        f = self.force
        f.env.unwrapped = f.env
        f.env.seed = lambda seed: None
        f.env.action_manager = SimpleNamespace(total_action_dim=1)
        def reset():
            f.env.episode_length_buf.zero_()
            f.state.episode_index.fill_(-1)  # Each paired fixture represents a fresh evaluator.
            f.state.reset(None)
            return torch.zeros((4, 1))
        def step(actions):
            f.step()
            return torch.zeros((4, 1)), torch.zeros(4), f.state.closed, {}
        f.env.reset, f.env.step = reset, step
        args = SimpleNamespace(seed=42, max_steps=None, policy="zero", action_scale=None,
            public_seeds=3, public_accept_seeds=3, sealed_validation_seeds=7,
            evaluation_subject=SimpleNamespace(kind="reference", subject_id="fixture"), task_id="unitree_push_recovery", round="cpu")
        phase_identity = {"job_id": "fixture", "task_id": args.task_id, "subject_kind": "reference", "subject_id": "fixture"}
        capture_identity = {**phase_identity, "evaluation_plan_hash": "b" * 64,
            "checkpoint_hash": "sha256:" + "c" * 64, "runtime_dynamics_hash": "sha256:" + "d" * 64}
        options = PushNumericCaptureOptions(Path(self.tmp.name) / "rollout", capture_identity, chunk_steps=32)
        with patch("true_rl_agent.tasks.push_recovery.rollout.public_runtime_dynamics", return_value={}):
            off = run_physical_push_rollout(f.env, {"push_evaluation_protocol": {}}, args, split_name="public_dev", run_index=1)
            on = run_physical_push_rollout(f.env, {"push_evaluation_protocol": {}}, args, split_name="public_dev", run_index=1, numeric_capture=options)
        self.assertEqual(off["aggregate"], on["aggregate"])
        self.assertEqual(off["secondary_metrics"], on["secondary_metrics"])
        self.assertEqual(off["per_episode"], on["per_episode"])
        self.assertNotIn("numeric_evidence_ref", off)
        self.assertIsNone(f.state.numeric_capture)
        self.assertFalse(hasattr(f.state, "_numeric_pose"))
        self.assertFalse(hasattr(f.state, "_numeric_planar_motion"))
        self.assertTrue(recompute_push_numeric_evidence(on["numeric_evidence_ref"])["matches_runtime"])
        from true_rl_agent.tasks.push_recovery.phased_evaluation import build_push_phase_summary
        from true_rl_agent.tasks.push_recovery.metrics import project_push_split
        from true_rl_agent.evaluation.phases import EvaluationPhase
        kwargs = dict(phase=EvaluationPhase.PUBLIC_DEV, evaluation_plan_hash="b" * 64,
            expected_seed_count=1, wall_time_ns=1, identity=phase_identity, execution_device="cpu",
            checkpoint_safety_report={"checkpoint_hash": capture_identity["checkpoint_hash"]},
            runtime_dynamics_hash=capture_identity["runtime_dynamics_hash"])
        on_phase = build_push_phase_summary(seed_summaries=[on], **kwargs)
        off_phase = build_push_phase_summary(seed_summaries=[off], **kwargs)
        legacy_ref = self.rewrite_archive(on["numeric_evidence_ref"], "rollout-legacy", legacy=True)
        build_push_phase_summary(seed_summaries=[{**on, "numeric_evidence_ref": legacy_ref}], **kwargs)
        self.assertNotEqual(on_phase["task_evidence_hash"], off_phase["task_evidence_hash"])
        self.assertEqual(project_push_split([on], split_name="public_dev", errors=[]),
            project_push_split([off], split_name="public_dev", errors=[]))
        with self.assertRaisesRegex(NumericEvidenceError, "another phase, policy, or seed"):
            build_push_phase_summary(seed_summaries=[on], **{**kwargs, "checkpoint_safety_report": {"checkpoint_hash": "sha256:" + "e" * 64}})

    def test_budget_and_episode_gap_fail_without_valid_manifest(self):
        f = self.velocity
        capture = self.capture(f, "budget", max_buffer_bytes=1)
        with self.assertRaisesRegex(NumericEvidenceError, "buffer budget"):
            self.velocity_until(1)
        self.assertFalse((capture.writer.root / "manifest.json").exists())
        # A declared complete artifact with a task-level coverage gap is rejected.
        f.state.numeric_capture = None
        f.state.begin_evaluation("public_dev", 42)
        f.env.episode_length_buf.zero_()
        f.state.reset(None)
        capture = self.capture(f, "gap")
        self.velocity_until(1)
        f.env.episode_length_buf += 1
        self.velocity_until(3)
        ref = capture.complete(f.state.episode_records())
        with self.assertRaisesRegex(NumericEvidenceError, "episode gap"):
            recompute_push_numeric_evidence(ref)

    def test_cli_capture_is_public_dev_only_and_legacy_is_out_of_scope(self):
        args = SimpleNamespace(numeric_evidence_dir=Path(self.tmp.name), numeric_evidence_max_bytes=100,
            evaluation_phase="public_accept", resolved_cell_manifest=SimpleNamespace(cell_variant_id="push.go2_12dof.recovery_v3"))
        with self.assertRaisesRegex(ValueError, "public_dev"):
            push_numeric_capture_options_from_args(args)
        args.evaluation_phase = "public_dev"
        args.resolved_cell_manifest.cell_variant_id = "push.g1_29dof.legacy_v1"
        with self.assertRaisesRegex(ValueError, "physical G1/Go2"):
            push_numeric_capture_options_from_args(args)
        args.numeric_evidence_dir = None
        self.assertIsNone(push_numeric_capture_options_from_args(args))


class RecoveryClassificationCompatibilityTests(unittest.TestCase):
    def test_velocity_and_force_keep_their_different_failure_priorities(self):
        from true_rl_agent.tasks.push_recovery.failure_classifier import (
            classify_force_recovery_episode, classify_recovery_episode,
        )
        parameters = {"fall_tilt_rad": .7, "fall_height_m": .2, "max_displacement_m": None}
        metrics = {"control_only": False, "planned_push_count": 2,
            "max_pitch_rad": .8, "min_pitch_rad": -.8, "max_abs_roll_rad": .8,
            "min_base_height_m": .1, "max_xy_drift_m": 0., "final_recovery_condition": False,
            "push_events": [{"injection_verified": False, "recovered": False}],
            "force_events": [{"injection_verified": False, "completed": False}],
            "unplanned_wrench": False, "sequence_recovered": False}
        velocity = classify_recovery_episode(metrics, parameters)
        force = classify_force_recovery_episode(metrics, parameters)
        self.assertEqual(velocity["label"], "invalid_push_injection")
        self.assertEqual(force["label"], "fall_forward")
        self.assertEqual(velocity["invalid_injection_count"], 1)
        self.assertEqual(force["invalid_injection_count"], 1)
        # Physical failure closes the planned cohort without treating the missing
        # second push as an additional injection error in either protocol.
        metrics["push_events"][0]["injection_verified"] = True
        metrics["force_events"][0]["injection_verified"] = True
        for classifier in (classify_recovery_episode, classify_force_recovery_episode):
            result = classifier(metrics, parameters)
            self.assertEqual(result["label"], "fall_forward")
            self.assertEqual(result["invalid_injection_count"], 0)

    def test_shared_terminal_classification_retains_strict_fall_boundaries(self):
        from true_rl_agent.tasks.push_recovery.failure_classifier import (
            classify_force_recovery_episode, classify_recovery_episode,
        )
        parameters = {"fall_tilt_rad": .7, "fall_height_m": .2, "max_displacement_m": None}
        metrics = {"control_only": False, "planned_push_count": 1,
            "max_pitch_rad": .7, "min_pitch_rad": -.7, "max_abs_roll_rad": .7,
            "min_base_height_m": .2, "max_xy_drift_m": 0., "final_recovery_condition": True,
            "push_events": [{"injection_verified": True, "recovered": True}],
            "force_events": [{"injection_verified": True, "completed": True}],
            "unplanned_wrench": False, "sequence_recovered": True}
        for classifier in (classify_recovery_episode, classify_force_recovery_episode):
            self.assertEqual(classifier(metrics, parameters)["label"], "success_recovered")
            self.assertEqual(classifier({**metrics, "final_recovery_condition": False}, parameters)["label"], "not_recovered")
            for change, label in (({"max_pitch_rad": .701}, "fall_forward"),
                                  ({"min_pitch_rad": -.701}, "fall_backward"),
                                  ({"max_abs_roll_rad": .701}, "fall_side"),
                                  ({"min_base_height_m": .199}, "height_collapse")):
                self.assertEqual(classifier({**metrics, **change}, parameters)["label"], label)


if __name__ == "__main__":
    unittest.main()
