"""Force sequence evidence, public projection and pre-reset recording contracts."""
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
                task_id="unitree_push_recovery", round="test", _push_video_recorder=recorder)
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


if __name__ == "__main__":
    unittest.main()
