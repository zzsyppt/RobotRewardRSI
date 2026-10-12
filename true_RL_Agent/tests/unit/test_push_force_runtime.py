"""CPU tensor checks of the production force hooks, not physics qualification."""
import importlib.util
import math
from types import SimpleNamespace
import unittest

from true_rl_agent.tasks.push_recovery.force_protocol import ForceEvaluationProtocol


class ForceRuntimeConfigurationTests(unittest.TestCase):
    def test_manifest_selects_force_only_for_new_evaluation_and_checks_frozen_resolution(self):
        from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
        from true_rl_agent.tasks.push_recovery.plugin import PushRecoveryTaskPlugin
        from true_rl_agent.tasks.push_recovery.robot_binding import resolve_push_cell_manifest
        from true_rl_agent.tasks.push_recovery.runtime import physical_runtime_config_from_manifest
        from true_rl_agent.tasks.push_recovery.protocol import physical_protocol_from_config

        for robot, body in (("g1_37dof", "torso_link"), ("go2_12dof", "base")):
            for version in ("v2", "v3"):
                cell = f"push.{robot}.recovery_{version}"
                config = PushRecoveryTaskConfig.for_cell(cell)
                manifest = resolve_push_cell_manifest(cell_variant_id=cell, config=config, plugin=PushRecoveryTaskPlugin(config))
                train = physical_runtime_config_from_manifest(manifest, "strong_push")
                evaluation = physical_runtime_config_from_manifest(manifest, "strong_push", evaluation=True)
                self.assertNotIn("push_evaluation_protocol", train)
                self.assertNotIn("force_body_name", train)
                self.assertEqual(physical_protocol_from_config(train)[1]["mechanism"], "world_planar_velocity_increment")
                if version == "v2":
                    self.assertEqual(train, evaluation)
                    continue
                self.assertEqual(evaluation["force_body_name"], body)
                self.assertEqual(evaluation["push_protocol"], train["push_protocol"])
                self.assertEqual(physical_protocol_from_config(evaluation)[1]["mechanism"], "world_planar_external_force")
                changed = manifest.to_dict()["robot_task_binding"]
                changed["task_projection"]["push_resolved_evaluation_protocol"]["repeated_gap_s"] += .01
                bad = SimpleNamespace(robot_task_binding=changed, runtime_adapter=manifest.runtime_adapter)
                with self.assertRaisesRegex(ValueError, "force evaluation protocol"):
                    physical_runtime_config_from_manifest(bad, "strong_push", evaluation=True)
                self.assertEqual(physical_runtime_config_from_manifest(bad, "strong_push"), train)


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires CPU torch")
class ForceRuntimeTests(unittest.TestCase):
    def setUp(self):
        import torch
        from true_rl_agent.tasks.push_recovery.force_runtime import ForcePushRuntimeState, _world_vector
        self.torch = torch
        n = 4
        p = ForceEvaluationProtocol().resolve(.4)
        p["evaluation_cohort_counts"] = dict.fromkeys(("low", "medium", "high", "repeated"), 1)
        self.data = SimpleNamespace(root_pos_w=torch.tensor([[0., 0., .4]] * n),
            root_lin_vel_w=torch.zeros((n, 3)), root_ang_vel_w=torch.zeros((n, 3)),
            root_quat_w=torch.tensor([[0., 0., 0., 1.]] * n),
            body_link_quat_w=torch.tensor([[[0., 0., math.sqrt(.5), math.sqrt(.5)]] * 2] * n),
            body_mass=torch.tensor([[2., 3.], [3., 4.], [4., 5.], [5., 6.]]))
        self.applied = []
        self.corrupt = False
        self.skip_apply = False
        composer = SimpleNamespace()
        def set_wrench(**kw):
            self.assertIsNone(kw["positions"])
            self.assertTrue(kw["is_global"])
            self.assertEqual(kw["body_ids"].tolist(), [1])
            composer.force = kw["forces"][:, 0].clone()
        composer.set_forces_and_torques_index = set_wrench
        view = SimpleNamespace(apply_forces_and_torques_at_position=lambda **kw: self.applied.append(kw["force_data"].clone()))
        self.robot = SimpleNamespace(data=self.data, num_bodies=2, find_bodies=lambda name: ([1], [name]),
            instantaneous_wrench_composer=composer, root_view=view)
        def write():
            if self.skip_apply:
                return
            inverse = self.data.body_link_quat_w[:, 1].clone()
            inverse[:, :3] *= -1
            force = torch.zeros((n, 2, 3))
            force[:, 1] = _world_vector(composer.force, inverse)
            if self.corrupt:
                force[0, 1, 0] += 1
            view.apply_forces_and_torques_at_position(force_data=force,
                torque_data=torch.zeros_like(force), position_data=None, is_global=False)
            composer.force.zero_()
        self.robot.write_data_to_sim = write
        class Scene(dict):
            pass
        scene = Scene(robot=self.robot)
        scene.env_origins = torch.zeros((n, 3))
        self.env = SimpleNamespace(num_envs=n, device="cpu", step_dt=.02, physics_dt=.005, scene=scene,
            episode_length_buf=torch.zeros(n, dtype=torch.long), max_episode_length=304,
            cfg=SimpleNamespace(push_protocol_parameters=p, push_evaluation=True, push_control_only=False,
                push_force_evaluation=True, push_runtime_config={"force_body_name": "base"}, decimation=4))
        self.state = ForcePushRuntimeState(self.env)
        self.env._push_runtime_state = self.state
        self.state.begin_evaluation("public_dev", 42)
        self.state.reset(None)
        self.state.install_force_hooks()

    def step(self):
        from true_rl_agent.tasks.push_recovery.runtime import capture_push_transition
        for _ in range(4):
            self.robot.write_data_to_sim()
        self.env.episode_length_buf += 1
        return capture_push_transition(self.env)

    def until(self, step):
        while int(self.env.episode_length_buf[0]) < step:
            self.step()

    def test_actual_mass_world_rotation_substeps_impulse_and_gap_zero(self):
        self.until(self.env.max_episode_length)
        records = self.state.episode_records()
        self.assertEqual([r["push_event_count"] for r in records], [1, 1, 1, 2])
        self.assertEqual(self.state.event_times, (51, 73))
        self.assertTrue(all(r["success"] for r in records))
        for env_id, record in enumerate(records):
            for event in record["force_events"]:
                self.assertEqual(event["actual_substeps"], 44)
                self.assertTrue(event["completed"] and event["injection_verified"])
                impulse = self.torch.tensor(event["delivered_impulse_world_ns"])
                expected = self.torch.tensor(event["force_world_n"]) * .22
                self.torch.testing.assert_close(impulse, expected)
                strength = self.state.parameters["impulse_velocity_mps"][record["stratum"] if env_id < 3 else "high"]
                if env_id == 3:
                    strength *= .8
                self.assertAlmostEqual(float(impulse.norm()) / record["actual_total_mass_kg"], strength, places=5)
        delivered = self.torch.stack(self.applied)
        self.assertEqual(float(delivered[:, :, 0].abs().sum()), 0)
        for before in (0, 50, 62, 72, 84, 303):
            self.assertEqual(float(delivered[before * 4:(before + 1) * 4].abs().sum()), 0)
        self.assertNotIn("recovered_event_count", records[0])
        self.assertNotIn("recovered", records[0]["force_events"][0])

    def test_sequence_requires_fresh_continuous_dwell_after_last_pulse(self):
        self.until(84)
        self.assertEqual(int(self.state.sequence_dwell[3]), 0)
        self.until(99)
        self.assertEqual(int(self.state.sequence_recovery_step[3]), -1)
        self.data.root_lin_vel_w[3, 0] = 1
        self.step()
        self.assertEqual(int(self.state.sequence_dwell[3]), 0)
        self.data.root_lin_vel_w.zero_()
        self.until(115)
        self.assertEqual(int(self.state.sequence_recovery_step[3]), -1)
        self.step()
        self.assertEqual(int(self.state.sequence_recovery_step[3]), 116)
        self.data.root_pos_w[:, 0] = 100  # Displacement remains diagnostic only.
        self.until(304)
        self.assertTrue(self.state.episode_records()[3]["success"])
        self.assertEqual(self.state.episode_records()[3]["max_xy_drift_m"], 100)

    def test_window_boundary_is_inclusive_and_later_dwell_does_not_count(self):
        end = 84
        self.data.root_lin_vel_w[2:, 0] = 1
        self.until(end + self.state.window - self.state.required_dwell)
        self.data.root_lin_vel_w[3, 0] = 0
        self.until(end + self.state.window)
        self.assertEqual(int(self.state.sequence_recovery_step[3]), end + self.state.window)
        self.data.root_lin_vel_w.zero_()
        self.until(304)
        self.assertTrue(self.state.episode_records()[3]["success"])
        self.assertFalse(self.state.episode_records()[2]["success"])

    def test_partial_pulse_early_fall_is_failure_not_bad_injection_and_freezes(self):
        captured = []
        self.env._push_pre_reset_capture = lambda: captured.append((int(self.env.episode_length_buf[0]), bool(self.state.closed[0])))
        self.until(51)
        self.data.root_pos_w[:, 2] = .1
        self.step()
        snapshot = self.state.episode_records()
        self.assertEqual(captured[-1], (52, False))
        for record in snapshot:
            self.assertEqual(record["failure_label"], "height_collapse")
            self.assertEqual(record["invalid_injection_count"], 0)
            self.assertEqual(record["verified_push_event_count"], 1)
            self.assertEqual(record["completed_push_event_count"], 0)
        self.data.root_pos_w[:, 2] = .4
        self.state.reset(None)
        self.step()
        self.assertEqual(self.state.episode_records(), snapshot)
        self.assertEqual(float(self.applied[-1].abs().sum()), 0)
        self.state.begin_evaluation("public_accept", 43)
        self.env.episode_length_buf.zero_()
        self.state.reset(None)
        self.assertEqual(self.state.actual_substeps.sum().item(), 0)
        self.assertEqual(self.state.episode_index.tolist(), [1] * 4)

    def test_bad_actual_wrench_and_missing_callback_are_invalid_even_when_falling(self):
        self.until(51)
        self.corrupt = True
        self.data.root_pos_w[:, 2] = .1
        self.step()
        record = self.state.episode_records()[0]
        self.assertEqual(record["failure_label"], "height_collapse")
        self.assertEqual(record["invalid_injection_count"], 1)
        self.assertFalse(record["force_events"][0]["injection_verified"])
        self.state.begin_evaluation("public_dev", 43)
        self.data.root_pos_w[:, 2] = .4
        self.env.episode_length_buf.zero_()
        self.state.reset(None)
        self.corrupt, self.skip_apply = False, True
        self.until(304)
        self.assertTrue(all(r["invalid_injection_count"] > 0 for r in self.state.episode_records()))

    def test_control_no_forces_and_unplanned_force_is_invalid(self):
        self.state.control_only = True
        self.state.reset(None)
        self.until(304)
        records = self.state.episode_records()
        self.assertTrue(all(r["control_success"] for r in records))
        self.assertTrue(all(r["failure_label"] == "no_push_control" and not r["force_events"] for r in records))
        self.assertEqual(float(self.torch.stack(self.applied).abs().sum()), 0)
        self.state.begin_evaluation("public_dev", 43)
        self.env.episode_length_buf.zero_()
        self.state.reset(None)
        self.corrupt = True
        self.until(304)
        record = self.state.episode_records()[0]
        self.assertEqual(record["failure_label"], "invalid_push_injection")
        self.assertFalse(record["control_success"])


if __name__ == "__main__":
    unittest.main()
