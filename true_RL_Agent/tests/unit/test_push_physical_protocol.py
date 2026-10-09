"""New physical protocol semantics; tensor cases run on target CPU, never a GPU."""
from __future__ import annotations

import importlib.util
import math
from dataclasses import replace
from types import SimpleNamespace
import unittest

from true_rl_agent.tasks.push_recovery.failure_classifier import classify_recovery_episode
from true_rl_agent.tasks.push_recovery.protocol import PushRecoveryProtocol, physical_public_runtime_dynamics
from true_rl_agent.tasks.push_recovery.rollout import aggregate_recovery_episodes


def _episode(**overrides):
    value = {"control_only": False, "stratum": "low", "planned_push_count": 1, "push_events": [],
        "max_pitch_rad": 0.0, "min_pitch_rad": 0.0, "max_abs_roll_rad": 0.0,
        "min_base_height_m": 0.4, "max_xy_drift_m": 0.0, "final_recovery_condition": True}
    value.update(overrides)
    return value


class PushPhysicalProtocolTests(unittest.TestCase):
    def test_dimensional_resolution_and_identity(self):
        protocol = PushRecoveryProtocol.defaults()
        small, large = protocol.resolve(0.4), protocol.resolve(0.9)
        self.assertAlmostEqual(large["velocity_scale_mps"] / small["velocity_scale_mps"], 1.5)
        self.assertAlmostEqual(large["time_scale_s"] / small["time_scale_s"], 1.5)
        self.assertEqual(PushRecoveryProtocol.from_mapping(protocol.to_dict()), protocol)
        self.assertNotEqual(replace(protocol, recovery_tilt_rad=0.19).protocol_hash, protocol.protocol_hash)
        self.assertNotEqual(small["resolved_protocol_hash"], large["resolved_protocol_hash"])
        with self.assertRaises(ValueError):
            protocol.resolve(float("nan"))
        with self.assertRaises(ValueError):
            replace(protocol, second_push_time_scale=10)

    def test_physical_runtime_identity_includes_placement_independent_robot_metadata(self):
        raw = {"push_protocol": PushRecoveryProtocol().to_dict(), "push_robot_scale_m": 0.4,
            "runtime_adapter_id": "go2_adapter_v1", "active_profile": "recovery_v2",
            "task": {"runtime_adapter_id": "go2_adapter_v1", "environment_cfg_entrypoint": "package:environment",
                "runner_cfg_entrypoint": "package:runner", "agent_configuration_hash": "a" * 64}}
        first = physical_public_runtime_dynamics(raw)
        changed = physical_public_runtime_dynamics({**raw, "push_robot_scale_m": 0.41})
        self.assertNotEqual(first["runtime_dynamics_hash"], changed["runtime_dynamics_hash"])
        self.assertEqual(first["environment"]["episode_length_s"], first["push_resolved_protocol"]["episode_length_s"])

    def test_full_eval_resolves_the_manifest_acceptance_domain_before_launch(self):
        import argparse
        import ast
        from pathlib import Path
        from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
        from true_rl_agent.tasks.push_recovery.plugin import PushRecoveryTaskPlugin
        from true_rl_agent.tasks.push_recovery.robot_binding import resolve_push_cell_manifest

        source = (Path(__file__).resolve().parents[2] / "scripts/isaaclab_rsl_rl_eval.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "acceptance_protocol_from_args")
        namespace = {"argparse": argparse, "Any": object}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "push_eval_acceptance", "exec"), namespace)
        for cell_id in ("push.g1_37dof.recovery_v2", "push.go2_12dof.recovery_v2", "push.g1_29dof.legacy_v1"):
            config = PushRecoveryTaskConfig.for_cell(cell_id)
            manifest = resolve_push_cell_manifest(cell_variant_id=cell_id, config=config, plugin=PushRecoveryTaskPlugin(config))
            args = SimpleNamespace(resolved_cell_manifest=manifest, public_seeds=3, public_accept_seeds=3,
                sealed_validation_seeds=7, num_envs=64, eval_protocol_version=config.eval_protocol_version,
                seed=42, acceptance_protocol_hash=config.acceptance_protocol().protocol_hash)
            self.assertEqual(namespace[function.name](args), config.acceptance_protocol())
            args.acceptance_protocol_hash = "0" * 64
            with self.assertRaisesRegex(ValueError, "resolved task configuration"):
                namespace[function.name](args)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in {"build_push_sealed_rollout_evidence", "build_push_public_result_from_summaries"}]
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(any(keyword.arg == "acceptance_protocol" for keyword in node.keywords) for node in calls))
        self.assertLess(source.index("args_cli.acceptance_protocol = acceptance_protocol_from_args(args_cli)"), source.index("AppLauncher(args_cli)"))

    def test_early_fall_remains_policy_failure_and_missing_event_is_invalid(self):
        p = PushRecoveryProtocol().resolve(0.4)
        early = classify_recovery_episode(_episode(min_base_height_m=0.1), p)
        missing = classify_recovery_episode(_episode(), p)
        self.assertEqual(early["label"], "height_collapse")
        self.assertEqual(early["invalid_injection_count"], 0)
        self.assertEqual(missing["label"], "invalid_push_injection")

    def test_every_event_and_terminal_condition_are_required(self):
        p = PushRecoveryProtocol().resolve(0.4)
        events = [{"injection_verified": True, "recovered": True}, {"injection_verified": True, "recovered": False}]
        self.assertEqual(classify_recovery_episode(_episode(planned_push_count=2, push_events=events), p)["label"], "not_recovered")
        events[1]["recovered"] = True
        self.assertTrue(classify_recovery_episode(_episode(planned_push_count=2, push_events=events), p)["success"])
        self.assertFalse(classify_recovery_episode(_episode(planned_push_count=2, push_events=events, final_recovery_condition=False), p)["success"])

    def test_control_is_separate_but_early_fall_is_in_formal_denominator(self):
        p = PushRecoveryProtocol().resolve(0.4)
        records = []
        for raw in (_episode(min_base_height_m=0.1), _episode(push_events=[{"injection_verified": True, "recovered": True}]),
                    _episode(control_only=True, planned_push_count=0)):
            result = {**raw, **classify_recovery_episode(raw, p)}
            result["failure_label"] = result.pop("label")
            records.append(result)
        summary = aggregate_recovery_episodes(records)
        self.assertEqual(summary["episodes"], 3)
        self.assertEqual(summary["eligible_episode_count"], 2)
        self.assertEqual(summary["pushed_episode_count"], 1)
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["control_success_count"], 1)
        self.assertEqual(summary["stratum_results"]["low"], {"episodes": 2, "successes": 1, "success_rate": 0.5})

    def test_disabled_displacement_retains_recovery_and_fall_requirements(self):
        p = PushRecoveryProtocol().resolve(.4)
        raw = _episode(max_xy_drift_m=2.5, push_events=[{"injection_verified": True, "recovered": True}])
        self.assertEqual(classify_recovery_episode(raw, p)["label"], "out_of_bounds")
        p["max_displacement_m"] = None
        self.assertTrue(classify_recovery_episode(raw, p)["success"])
        self.assertEqual(classify_recovery_episode({**raw, "final_recovery_condition": False}, p)["label"], "not_recovered")
        self.assertEqual(classify_recovery_episode({**raw, "min_base_height_m": .1}, p)["label"], "height_collapse")


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires target CPU torch; not a physics acceptance test")
class PushPhysicalTensorTests(unittest.TestCase):
    def setUp(self):
        import torch
        from true_rl_agent.tasks.push_recovery.runtime import PushRuntimeState
        self.torch = torch
        n = 2
        self.data = SimpleNamespace(root_pos_w=torch.tensor([[0., 0., .4], [2., 0., .4]]),
            root_lin_vel_w=torch.zeros((n, 3)), root_ang_vel_w=torch.zeros((n, 3)),
            root_quat_w=torch.tensor([[0., 0., 0., 1.]] * n))
        data = self.data
        def write(value, env_ids):
            data.root_lin_vel_w[env_ids] = value[:, :3]
            data.root_ang_vel_w[env_ids] = value[:, 3:]
        class Scene(dict):
            pass
        scene = Scene(robot=SimpleNamespace(data=data, write_root_velocity_to_sim=write))
        scene.env_origins = torch.zeros((n, 3))
        self.env = SimpleNamespace(num_envs=n, device="cpu", step_dt=.02, scene=scene,
            episode_length_buf=torch.zeros(n, dtype=torch.long), max_episode_length=304,
            cfg=SimpleNamespace(push_protocol_parameters=PushRecoveryProtocol().resolve(.4), push_evaluation=True, push_control_only=False))
        self.state = PushRuntimeState(self.env)
        self.env._push_runtime_state = self.state
        self.state.begin_evaluation("public_dev", 42)
        self.state.reset(None)

    def test_increment_preserves_velocity_components_and_has_event_only_public_signal(self):
        from true_rl_agent.tasks.push_recovery.runtime_adapter import public_push_signal_from_env
        self.data.root_lin_vel_w[:] = self.torch.tensor([.7, -.2, .3])
        self.data.root_ang_vel_w[:] = self.torch.tensor([.1, .2, .3])
        before = self.data.root_lin_vel_w.clone()
        self.env.episode_length_buf[:] = self.state.event_times[0]
        self.state.inject(None)
        self.assertTrue(bool(self.state.verified[:, 0].all()))
        self.torch.testing.assert_close(self.data.root_lin_vel_w[:, :2] - before[:, :2], self.state.requested[:, 0])
        self.torch.testing.assert_close(self.data.root_lin_vel_w[:, 2], before[:, 2])
        self.torch.testing.assert_close(self.data.root_ang_vel_w, self.torch.tensor([[.1, .2, .3]] * 2))
        self.state.inject(None)
        active, delta, observed = public_push_signal_from_env(env=self.env, step_extras=None, rewards=self.torch.zeros(2),
            current_base_lin_vel_xy=self.torch.full((2, 2), 100.), previous_base_lin_vel_xy=self.torch.zeros((2, 2)))
        self.assertTrue(observed)
        self.assertEqual(float(active.sum()), 0)
        self.assertEqual(float(delta.abs().sum()), 0)

    def test_disabled_displacement_continues_episode_and_records_drift(self):
        self.state.parameters["max_displacement_m"] = None
        self.data.root_pos_w[:, 0] += 2.5
        self.env.episode_length_buf[:] = 100
        self.assertFalse(bool(self.state.observe().any()))
        self.assertFalse(bool(self.state.closed.any()))
        self.torch.testing.assert_close(self.state.max_drift, self.torch.full((2,), 2.5))
        self.data.root_pos_w[0, 2] = .1
        self.env.episode_length_buf += 1
        self.assertEqual(self.state.observe().tolist(), [True, False])
        self.assertEqual(self.state.closed.tolist(), [True, False])
        self.env.episode_length_buf[:] = self.env.max_episode_length
        self.state.observe()
        self.assertTrue(bool(self.state.closed.all()))

    def test_dwell_must_be_continuous_and_closed_episode_survives_reset(self):
        self.env.episode_length_buf[:] = self.state.event_times[0]
        self.state.inject(None)
        self.data.root_lin_vel_w.zero_()
        start = self.state.event_times[0]
        for step in range(1, self.state.required_dwell):
            self.env.episode_length_buf[:] = start + step
            self.state.observe()
        self.assertTrue(bool((self.state.recovered_step < 0).all()))
        self.data.root_lin_vel_w[:, 0] = 2
        self.env.episode_length_buf += 1
        self.state.observe()
        self.assertEqual(int(self.state.dwell.sum()), 0)
        self.data.root_lin_vel_w.zero_()
        for _ in range(self.state.required_dwell):
            self.env.episode_length_buf += 1
            self.state.observe()
        self.assertTrue(bool((self.state.recovered_step[:, 0] >= 0).all()))
        self.data.root_pos_w[0, 2] = .1
        self.env.episode_length_buf += 1
        self.state.observe()
        self.assertTrue(bool(self.state.closed[0]))
        snapshot = self.state.episode_records()[0]
        self.data.root_pos_w[0, 2] = .4
        self.state.reset(self.torch.tensor([0]))
        self.assertEqual(self.state.episode_records()[0], snapshot)
        self.assertEqual(snapshot["failure_label"], "height_collapse")

    def test_actual_exposure_counts_survive_reset_and_distinguish_invalid_write(self):
        self.state.evaluation = False
        self.state.reset(None)
        self.env.episode_length_buf[:] = self.state.event_times[0]
        self.state.inject(None)
        self.assertEqual(self.state.exposure_summary()["injection_attempt_count"], 2)
        self.assertEqual(self.state.exposure_summary()["verified_event_count"], 2)
        self.state.inject(None)  # Already injected: no duplicate event.
        self.state.reset(None)
        self.env.scene["robot"].write_root_velocity_to_sim = lambda value, env_ids: None
        self.state.inject(None)
        report = self.state.exposure_summary()
        self.assertEqual(report["injection_attempt_count"], 4)
        self.assertEqual(report["verified_event_count"], 2)
        self.assertFalse(report["evaluation"])
        self.state.control_only = True
        self.state.reset(None)
        self.state.inject(None)
        self.assertEqual(self.state.exposure_summary()["injection_attempt_count"], 4)

    def test_no_push_control_writes_no_events(self):
        self.state.control_only = True
        self.state.reset(None)
        self.env.episode_length_buf[:] = self.env.max_episode_length
        self.state.inject(None)
        self.state.observe()
        self.assertEqual(int((self.state.event_steps >= 0).sum()), 0)
        result = aggregate_recovery_episodes(self.state.episode_records())
        self.assertEqual(result["eligible_episode_count"], 0)
        self.assertIsNone(result["success_rate"])

    def test_runtime_consumes_resolved_cohort_and_direction_parameters(self):
        self.state.parameters["evaluation_cohort_counts"] = dict.fromkeys(("low", "medium", "high", "repeated"), 1)
        self.state.parameters["evaluation_direction_policy"].update(count=1, second_event_offset_rad=0.7)
        self.state.reset(None)
        self.assertEqual(self.state.stratum.tolist(), [1, 2])
        self.torch.testing.assert_close(self.state.angle[:, 0], self.torch.full((2,), self.state.phase_angle))
        self.torch.testing.assert_close(self.state.angle[:, 1] - self.state.angle[:, 0], self.torch.full((2,), 0.7))

    def test_native_xyzw_rotation_is_converted_before_failure_classification(self):
        angle = 1.0
        self.data.root_quat_w[0] = self.torch.tensor([0., math.sin(angle / 2), 0., math.cos(angle / 2)])
        self.env.episode_length_buf += 1
        self.state.observe()
        self.assertEqual(self.state.episode_records()[0]["failure_label"], "fall_forward")


if __name__ == "__main__":
    unittest.main()
