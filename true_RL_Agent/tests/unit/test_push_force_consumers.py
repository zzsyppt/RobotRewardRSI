"""Force evidence and shared velocity/force pre-reset recording contracts."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from true_rl_agent.evaluation.phases import EvaluationPhase
from true_rl_agent.tasks.push_recovery import metrics
from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
from true_rl_agent.tasks.push_recovery.phased_adapter import PushPhasedEvaluationAdapter, PUSH_PHASE_EVIDENCE_SCHEMA
from true_rl_agent.tasks.push_recovery.prompt_adapter import PushRecoveryPromptAdapter
from true_rl_agent.tasks.push_recovery.rollout import aggregate_recovery_episodes, run_physical_push_rollout


def episode(*, success=True, partial=False, env_id=0):
    events = [{"event_index": i, "start_control_step": 1 + i * 22,
               "end_control_step": 12 + i * 22, "actual_substeps": 8 if partial and i == 1 else 44,
               "expected_substeps": 44, "completed": not (partial and i == 1),
               "injection_verified": True, "force_world_n": [20., 0., 0.],
               "delivered_impulse_world_ns": [4.4, 0., 0.]} for i in range(2)]
    return {"env_id": env_id, "stratum": "repeated", "control_only": False,
            "control_success": False, "success": success, "failure_label": "success_recovered" if success else "fall_side",
            "planned_push_count": 2, "push_event_count": 2, "completed_push_event_count": 1 if partial else 2,
            "verified_push_event_count": 2, "invalid_injection_count": 0, "force_events": events,
            "sequence_recovered": success, "sequence_recovery_control_step": 50 if success else None,
            "final_recovery_condition": success, "episode_steps": 3,
            "min_base_height_m": .12, "max_abs_pitch_rad": .1, "max_abs_roll_rad": .9 if not success else .1,
            "max_xy_drift_m": 100., "final_planar_speed_mps": .1}


def rollout():
    rows = [episode(env_id=0), episode(success=False, partial=True, env_id=1)]
    return {"schema_version": metrics.FORCE_ROLLOUT_SCHEMA, "status": "pass", "per_episode": rows,
            "aggregate": aggregate_recovery_episodes(rows, force_sequence=True),
            "secondary_metrics": {"mean_min_base_height_m": .12, "mean_max_xy_drift_m": 100.}}


class PushForceConsumersTests(unittest.TestCase):
    def test_both_mechanisms_record_numeric_state_before_reset_and_interval_once(self):
        import numpy as np
        import torch
        from true_rl_agent.runner.native_video import NativeVideoOptions, NativeVideoRecorder
        from true_rl_agent.runner.numeric_evidence import iter_numeric_evidence
        from true_rl_agent.tasks.push_recovery.numeric_capture import PushNumericCaptureOptions
        from true_rl_agent.tasks.push_recovery.runtime import capture_push_transition
        from tests.unit.test_push_force_runtime import ForceRuntimeTests
        from tests.unit.test_push_physical_protocol import PushPhysicalTensorTests

        def run(root, *, force, video):
            fixture = ForceRuntimeTests() if force else PushPhysicalTensorTests()
            fixture.setUp()
            base, state, data = fixture.env, fixture.state, fixture.data
            base.unwrapped = base
            base.max_episode_length = 3
            base.action_manager = SimpleNamespace(total_action_dim=1)
            base.seed = lambda seed: None
            # The fake interval event occurs at step 1, after numeric capture.
            if not force:
                state.event_times = (1, 2)
            prior_hook = lambda: None
            base._push_pre_reset_capture = prior_hook
            snapshots, encoded, interval_changes = [], [], []
            def reset():
                base.episode_length_buf.zero_()
                state.reset(None)
                return torch.zeros(base.num_envs, 1)
            base.reset = reset
            def render():
                return np.full((2, 2, 3), int(float(data.root_pos_w[0, 2]) * 100), dtype=np.uint8)
            base.render = render
            def step(action):
                base.episode_length_buf += 1
                step_number = int(base.episode_length_buf[1])
                data.root_lin_vel_w[:, 0] = step_number / 10
                if step_number == 2:
                    data.root_pos_w[0, 2] = .05  # Selected first episode closes early.
                failure = capture_push_transition(base)
                done = failure | (base.episode_length_buf >= base.max_episode_length)
                ids = done.nonzero(as_tuple=False).flatten()
                if len(ids):
                    data.root_pos_w[ids, 2] = .99  # Native reset follows the task observation.
                    base.episode_length_buf[ids] = 0
                    state.reset(ids)
                if not force:
                    before = data.root_lin_vel_w.clone()
                    state.inject(None)  # Mirrors the installed post-reset interval phase.
                    interval_changes.append(not torch.equal(before, data.root_lin_vel_w))
                return torch.zeros(base.num_envs, 1), torch.zeros(base.num_envs), done, {}
            base.step = step
            recorder = None
            class Wrapper:
                unwrapped = base
                def reset(self):
                    return base.reset()
                def render(self):
                    return base.render()
                def step(self, action):
                    result = base.step(action)
                    # Same fallback condition as NativeVideoRecorder.RecordingEnv.
                    if recorder is not None and not recorder._task_records_steps:
                        recorder.record_step()
                    return result
            env = Wrapper()
            if video:
                recorder = object.__new__(NativeVideoRecorder)
                recorder.env, recorder.options = env, NativeVideoOptions(root / "video", resolution=(2, 2))
                recorder.options.output_dir.mkdir(parents=True)
                recorder.control_dt_s, recorder.fps, recorder.records = .02, 50., []
                recorder._recording, recorder._frames = False, []
                native_record = recorder.record_step
                def record():
                    snapshots.append((int(base.episode_length_buf[0]),
                                      float(data.root_pos_w[0, 2]), float(data.root_lin_vel_w[0, 0])))
                    native_record()
                recorder.record_step = record
            args = SimpleNamespace(seed=42, max_steps=3, policy="zero", action_scale=None,
                public_seeds=3, public_accept_seeds=3, sealed_validation_seeds=7,
                evaluation_subject=SimpleNamespace(kind="reference", subject_id="program"),
                task_id="unitree_push_recovery", round="test", _push_video_recorders=() if recorder is None else (recorder,))
            def encode(path, frames, **kwargs):
                encoded.extend(int(frame[0, 0, 0]) for frame in frames)
                path.write_bytes(b"encoded")
            with patch("true_rl_agent.tasks.push_recovery.rollout.public_runtime_dynamics", return_value={}), \
                 patch("true_rl_agent.runner.native_video._encode_frames", side_effect=encode), \
                 patch("torch.cuda.is_available", return_value=False):
                summary = run_physical_push_rollout(env, {"push_evaluation_protocol": {}} if force else {}, args,
                    split_name="public_dev", run_index=1,
                    numeric_capture=PushNumericCaptureOptions(root / "numeric", {"fixture": "CPU"}, chunk_steps=2))
            arrays = list(iter_numeric_evidence(summary["numeric_evidence_ref"]))
            selected = [(int(a["control_step"][row, 0]), float(a["pose"][row, 0, 0]),
                         float(a["planar_motion"][row, 0, 0]))
                        for _, a in arrays for row in range(len(a["valid"])) if a["valid"][row, 0]]
            self.assertIs(base._push_pre_reset_capture, prior_hook)
            self.assertIsNone(state.numeric_capture)
            if video:
                self.assertEqual(snapshots, selected)
                self.assertEqual([row[0] for row in snapshots], [1, 2])
                self.assertEqual(encoded, [40, 5])  # Never the reset state's 99.
                self.assertEqual(recorder.records[0]["frame_count"], 2)
                self.assertTrue(recorder.records[0]["includes_terminal_frame"])
                self.assertEqual(recorder.records[0]["capture_point"],
                    "after_physics_before_task_observe_reset_and_interval_events")
            if not force:
                self.assertTrue(any(interval_changes))
            return summary, arrays

        with tempfile.TemporaryDirectory() as tmp:
            for force in (False, True):
                with self.subTest(force=force):
                    root = Path(tmp) / ("force" if force else "velocity")
                    recorded, numeric = run(root / "recorded", force=force, video=True)
                    baseline, original = run(root / "baseline", force=force, video=False)
                    for key in ("per_episode", "aggregate", "secondary_metrics", "execution_cost"):
                        self.assertEqual(recorded[key], baseline[key])
                    for (_, captured), (_, unrecorded) in zip(numeric, original, strict=True):
                        self.assertEqual(captured.keys(), unrecorded.keys())
                        for key in captured:
                            np.testing.assert_array_equal(captured[key], unrecorded[key])

    def test_sequence_counts_are_separate_from_pulse_completion_and_early_falls(self):
        raw = rollout()
        counts = metrics.recovery_episode_counts(raw)
        self.assertEqual((counts["planned_push_event_count"], counts["push_event_count"],
                          counts["completed_push_event_count"], counts["verified_push_event_count"],
                          counts["sequence_recovered_count"]), (4, 4, 3, 4, 1))
        self.assertNotIn("recovered_event_count", counts)
        self.assertEqual(metrics.push_recovery_evidence_gate([raw])["status"], "pass")
        cfg = PushRecoveryTaskConfig.for_cell("push.go2_12dof.recovery_v3")
        result = metrics.build_push_public_result_from_summaries(public_dev_summaries=[raw] * 3,
            public_accept_summaries=[raw] * 3, budget_summary={"status": "pass"},
            acceptance_protocol=cfg.acceptance_protocol(), evaluation_protocol=cfg.evaluation_protocol())
        self.assertEqual(result.status, "pass")
        self.assertEqual(result.public_score, .5)
        self.assertNotIn("motion_quality_gate", result.metrics)
        self.assertNotIn("behavior_quality_gate", result.metrics)

    def test_pulse_completion_and_labels_must_match_raw_evidence(self):
        for mutate in (
            lambda raw: raw["aggregate"].update(completed_push_event_count=4),
            lambda raw: raw["per_episode"][0]["force_events"][0].update(actual_substeps=43),
            lambda raw: raw["per_episode"][0].update(sequence_recovered=False),
            lambda raw: raw["per_episode"][1].update(failure_label="out_of_bounds"),
        ):
            raw = rollout()
            mutate(raw)
            self.assertEqual(metrics.push_recovery_evidence_gate([raw])["status"], "fail")

    def test_force_phase_rejects_old_schema_and_gates_both_robot_cells(self):
        from tests.unit.test_push_recovery_v2_metrics import rollout as old_rollout
        for cell in ("push.go2_12dof.recovery_v3", "push.g1_37dof.recovery_v3"):
            cfg = PushRecoveryTaskConfig.for_cell(cell)
            adapter = PushPhasedEvaluationAdapter(cfg, "a" * 64)
            phase = SimpleNamespace(task_id="unitree_push_recovery", phase=EvaluationPhase.PUBLIC_DEV,
                eval_protocol_hash=cfg.evaluation_protocol().protocol_hash,
                task_evidence={"schema_version": PUSH_PHASE_EVIDENCE_SCHEMA, "phase": "public_dev",
                               "seed_summaries": [rollout()] * 3})
            self.assertEqual(adapter.project_reference_public_dev_gate(phase_result=phase)["status"], "pass")
            phase.task_evidence["seed_summaries"] = [old_rollout()] * 3
            with self.assertRaisesRegex(ValueError, "requires v3"):
                adapter.project_reference_public_dev_gate(phase_result=phase)

    def test_unknown_or_mixed_schema_never_falls_back_to_legacy_scoring(self):
        raw = rollout()
        raw["schema_version"] = "true_rl_agent.push_recovery.rollout.v99"
        errors = []
        metrics.project_push_split([raw], split_name="public_dev", errors=errors)
        self.assertTrue(errors)
        from tests.unit.test_push_recovery_v2_metrics import rollout as old_rollout
        errors = []
        metrics.project_push_split([rollout(), old_rollout()], split_name="public_dev", errors=errors)
        self.assertIn("mixed_recovery_rollout_schemas", [row["reason"] for row in errors])

    def test_prompt_reports_sequence_evidence_without_changing_training_signal_units(self):
        cfg = PushRecoveryTaskConfig.for_cell("push.go2_12dof.recovery_v3")
        split = metrics.project_push_split([rollout()], split_name="public_dev", errors=[])
        record = {"round_id": "reference", "candidate": {"candidate_id": "program"},
                  "train_result": {"summary": {}},
                  "decision": {"accepted": True}, "public_result": {"metrics": {"public_dev": split}}}
        sections = PushRecoveryPromptAdapter(cfg).build_prompt_sections(public_records=[record], public_context={})
        counts = sections["public_feedback_summary"]["public_dev"]["recovery_counts"]
        self.assertEqual(counts["sequence_recovered_count"], 1)
        self.assertEqual(counts["completed_push_event_count"], 3)
        self.assertNotIn("recovered_event_count", counts)
        self.assertIn("last complete", sections["public_eval_contract"]["recovery_scope"])
        self.assertEqual(sections["reward_inputs_contract"]["fields"]["push_vec_public"]["units"], "m/s")
        self.assertIn("unchanged training", sections["reward_inputs_contract"]["training_scope"])

    def test_native_records_terminal_state_before_reset_once_and_restores_hook(self):
        import numpy as np
        import torch
        from true_rl_agent.runner.native_video import NativeVideoOptions, NativeVideoRecorder

        rows = [episode(env_id=0), episode(env_id=1)]
        rows[0]["episode_steps"] = 2
        state = SimpleNamespace(closed=torch.zeros(2, dtype=torch.bool), control_only=False,
            parameters={"resolved_protocol_hash": "b" * 64, "mechanism": "world_planar_external_force"},
            begin_evaluation=lambda *a: None, episode_records=lambda: copy.deepcopy(rows))
        class Env:
            def __init__(self):
                self.unwrapped = self
                self.num_envs, self.device, self.step_dt, self.max_episode_length = 2, "cpu", .02, 3
                self.action_manager = SimpleNamespace(total_action_dim=1)
                self.step_number, self.render_state = 0, 0
            def seed(self, seed): pass
            def reset(self): return torch.zeros(2, 1)
            def render(self): return np.full((2, 2, 3), self.render_state, dtype=np.uint8)
            def step(self, action):
                self.step_number += 1
                self.render_state = self.step_number
                self._push_pre_reset_capture()
                state.closed[0] = self.step_number >= 2
                state.closed[1] = self.step_number >= 3
                self.render_state = 99  # native autoreset follows the capture point
                return torch.zeros(2, 1), torch.zeros(2), state.closed, {}
        env = Env()
        previous_hook = lambda: None
        env._push_pre_reset_capture = previous_hook
        with tempfile.TemporaryDirectory() as tmp:
            recorder = object.__new__(NativeVideoRecorder)
            recorder.env, recorder.options = env, NativeVideoOptions(Path(tmp), resolution=(2, 2))
            recorder.control_dt_s, recorder.fps, recorder.records = .02, 50., []
            recorder._recording, recorder._frames = False, []
            args = SimpleNamespace(seed=42, max_steps=None, policy="zero", action_scale=None,
                public_seeds=3, public_accept_seeds=3, sealed_validation_seeds=7,
                evaluation_subject=SimpleNamespace(kind="reference", subject_id="program"),
                task_id="unitree_push_recovery", round="test", _push_video_recorders=(recorder,))
            captured = []
            def encode(path, frames, **kwargs):
                captured.extend(int(frame[0, 0, 0]) for frame in frames)
                path.write_bytes(b"encoded")
            with patch("true_rl_agent.tasks.push_recovery.rollout.push_runtime_state", return_value=state), \
                 patch("true_rl_agent.tasks.push_recovery.rollout.public_runtime_dynamics", return_value={}), \
                 patch("true_rl_agent.runner.native_video._encode_frames", side_effect=encode):
                summary = run_physical_push_rollout(env, {"push_evaluation_protocol": {}}, args,
                    split_name="public_dev", run_index=1)
            self.assertEqual(summary["schema_version"], metrics.FORCE_ROLLOUT_SCHEMA)
            self.assertEqual(summary["secondary_metrics"]["mean_max_xy_drift_m"], 100.)
            self.assertNotIn("mean_final_xy_drift_m", summary["secondary_metrics"])
            self.assertEqual(captured, [1, 2])
            self.assertEqual(recorder.records[0]["frame_count"], 2)
            self.assertTrue(recorder.records[0]["includes_terminal_frame"])
            self.assertIs(env._push_pre_reset_capture, previous_hook)

    def test_two_native_snapshot_streams_close_at_each_env_terminal(self):
        import torch
        from tests.unit.test_stair_pre_reset_media import CpuEnv, first_pixel, native_modules
        from true_rl_agent.runner.native_video import NativeVideoOptions, native_video_recorders

        rows = [episode(env_id=0), episode(env_id=1)]
        rows[0]["episode_steps"], rows[1]["episode_steps"] = 2, 3
        state = SimpleNamespace(closed=torch.zeros(2, dtype=torch.bool), control_only=False,
            parameters={"resolved_protocol_hash": "b" * 64, "mechanism": "world_planar_external_force"},
            begin_evaluation=lambda *a: None, episode_records=lambda: copy.deepcopy(rows))
        class Env(CpuEnv):
            def __init__(self):
                super().__init__(legacy=False)
                self.num_envs, self.device, self.max_episode_length = 2, "cpu", 3
                self.action_manager = SimpleNamespace(total_action_dim=1)
            def seed(self, seed): pass
            def reset(self): return torch.zeros(2, 1)
            def step(self, action):
                self.step_number += 1
                self.data.root_pos_w.torch[:, 0] = torch.tensor([self.step_number * 10, self.step_number * 10 + 100])
                self._push_pre_reset_capture()
                state.closed[:] = torch.tensor([self.step_number >= 2, self.step_number >= 3])
                if state.closed[0]: self.data.root_pos_w.torch[0, 0] = 99
                return torch.zeros(2, 1), torch.zeros(2), state.closed, {}
        base = Env()
        with tempfile.TemporaryDirectory() as tmp, patch.dict("sys.modules", native_modules()):
            recorders = native_video_recorders(base, NativeVideoOptions(Path(tmp), resolution=(2, 2),
                snapshot_interval_steps=1, secondary_env_index=1,
                secondary_eye=(-1., 0., 1.), secondary_lookat=(0., 0., 0.)))
            args = SimpleNamespace(seed=42, max_steps=None, policy="zero", action_scale=None,
                public_seeds=3, public_accept_seeds=3, sealed_validation_seeds=7,
                evaluation_subject=SimpleNamespace(kind="reference", subject_id="program"),
                task_id="unitree_push_recovery", round="test", _push_video_recorders=recorders)
            with patch("true_rl_agent.tasks.push_recovery.rollout.push_runtime_state", return_value=state), \
                 patch("true_rl_agent.tasks.push_recovery.rollout.public_runtime_dynamics", return_value={}), \
                 patch("torch.cuda.is_available", return_value=False):
                run_physical_push_rollout(recorders[0].env, {"push_evaluation_protocol": {}}, args,
                                         split_name="public_dev", run_index=1)
            self.assertEqual(base.step_number, 3)
            self.assertEqual([r._control_step for r in recorders], [2, 3])
            self.assertFalse(hasattr(base, "_push_pre_reset_capture"))
            all_paths = []
            for index, recorder in enumerate(recorders):
                record = recorder.records[0]
                self.assertEqual((record["env_index"], record["terminal_control_step"]), (index, index + 2))
                self.assertTrue(record["includes_terminal_frame"])
                self.assertEqual(record["run_index"], 1)
                samples = record["manifest"]["samples"]
                self.assertEqual([s["control_step"] for s in samples], list(range(1, index + 3)))
                for sample in samples:
                    self.assertEqual(sample["state_before"], sample["state_after"])
                    for view in sample["views"]:
                        all_paths.append(view["file"])
                        self.assertEqual(first_pixel(Path(tmp) / view["file"]), sample["control_step"] * 10 + index * 100)
            self.assertEqual(len(all_paths), len(set(all_paths)))
            self.assertEqual(len(all_paths), 10)

    def test_sparse_snapshot_terminal_flag_uses_final_samples(self):
        import torch
        for captured_steps in ([1], [1, 2]):
            with self.subTest(captured_steps=captured_steps):
                rows = [episode(env_id=0)]
                rows[0]["episode_steps"] = 2
                state = SimpleNamespace(closed=torch.zeros(1, dtype=torch.bool), control_only=False,
                    parameters={"resolved_protocol_hash": "b" * 64, "mechanism": "world_planar_external_force"},
                    begin_evaluation=lambda *a: None, episode_records=lambda: copy.deepcopy(rows))
                record = {"frame_count": 2, "capture_mode": "selected_rgb_snapshots",
                          "manifest": {"samples": [{"control_step": step} for step in captured_steps]}}
                recorder = SimpleNamespace(options=SimpleNamespace(env_index=0),
                    start=lambda *a, **kw: None, record_step=lambda: None,
                    finish=lambda **kw: record)
                class Env:
                    def __init__(self):
                        self.unwrapped = self
                        self.num_envs, self.device, self.step_dt, self.max_episode_length = 1, "cpu", .02, 2
                        self.action_manager = SimpleNamespace(total_action_dim=1)
                        self.count = 0
                    def seed(self, seed): pass
                    def reset(self): return torch.zeros(1, 1)
                    def step(self, action):
                        self.count += 1
                        self._push_pre_reset_capture()
                        state.closed[0] = self.count == 2
                        return torch.zeros(1, 1), torch.zeros(1), state.closed, {}
                args = SimpleNamespace(seed=42, max_steps=None, policy="zero", action_scale=None,
                    public_seeds=3, public_accept_seeds=3, sealed_validation_seeds=7,
                    evaluation_subject=SimpleNamespace(kind="reference", subject_id="program"),
                    task_id="unitree_push_recovery", round="test", _push_video_recorders=(recorder,))
                with patch("true_rl_agent.tasks.push_recovery.rollout.push_runtime_state", return_value=state), \
                     patch("true_rl_agent.tasks.push_recovery.rollout.public_runtime_dynamics", return_value={}), \
                     patch("torch.cuda.is_available", return_value=False):
                    run_physical_push_rollout(Env(), {"push_evaluation_protocol": {}}, args,
                                             split_name="public_dev", run_index=1)
                self.assertEqual(record["terminal_control_step"], 2)
                self.assertEqual(record["includes_terminal_frame"], 2 in captured_steps)


if __name__ == "__main__":
    unittest.main()
