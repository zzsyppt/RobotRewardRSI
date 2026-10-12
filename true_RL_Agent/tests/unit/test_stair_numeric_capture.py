from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock

import torch

from true_rl_agent.runner.numeric_evidence import NumericEvidenceWriter, NumericEvidenceError, iter_numeric_evidence, load_numeric_evidence_manifest
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.evaluation_runtime import run_stair_split_execution
from true_rl_agent.tasks.stair_climbing.numeric_capture import StairNumericCapture, recompute_stair_numeric_evidence, STAIR_NUMERIC_PROFILE_V1, STAIR_NUMERIC_PROFILE
from true_rl_agent.tasks.stair_climbing.legacy_transition_reader import STAIR_CONTACT_HISTORY_TENSOR_FIELDS
from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
from true_rl_agent.tasks.stair_climbing.robot_binding import resolve_stair_cell_manifest
from true_rl_agent.tasks.stair_climbing.runtime_profile import STAIR_EVAL_TRANSITION_EXTRAS_KEY
from true_rl_agent.tasks.stair_climbing.runtime_projection import STAIR_TRANSITION_VNEXT_SCHEMA_VERSION, stair_runtime_projection_from_manifest, stair_contact_force_history_snapshot
from true_rl_agent.tasks.stair_climbing.seed_policy import STAIR_FORMAL_EVAL_NAMESPACE
from true_rl_agent.tasks.stair_climbing.unified_scoring import build_unified_stair_summary

ROOT = Path(__file__).resolve().parents[2]


class StairNumericCaptureTests(unittest.TestCase):
    def test_runtime_limit_snapshot_preserves_environment_joint_and_bound_axes(self):
        from true_rl_agent.tasks.stair_climbing.numeric_capture import runtime_joint_position_limits

        hard = torch.tensor([[[-1., 1.], [-2., 2.], [-3., 3.]],
                             [[-4., 4.], [-5., 5.], [-6., 6.]]])
        robot = SimpleNamespace(joint_names=["unused", "calf", "hip"],
            data=SimpleNamespace(joint_pos_limits=hard, soft_joint_pos_limits=hard * .5))
        result = runtime_joint_position_limits(robot, (2, 1))
        self.assertEqual(result["joint_names"], ["hip", "calf"])
        self.assertEqual(result["values"]["joint_pos_limits"],
                         [[[-3., 3.], [-2., 2.]], [[-6., 6.], [-5., 5.]]])
        self.assertEqual(result["values"]["soft_joint_pos_limits"],
                         [[[-1.5, 1.5], [-1., 1.]], [[-3., 3.], [-2.5, 2.5]]])
        hard.zero_()
        self.assertEqual(result["values"]["joint_pos_limits"][0][0], [-3., 3.])

    def test_actual_recorder_skips_history_snapshot_when_capture_is_off(self):
        # Run the actual method body with tensor fixtures, without importing Kit.
        # This proves the collection gate, not a live sensor/GPU execution.
        source = ROOT / "src/true_rl_agent/tasks/stair_climbing/isaaclab_runtime_env.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "StairEvalTransitionRecorder")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "record_post_step")
        snapshot = Mock(return_value={"contact_force_history": {"fixture": True}})
        namespace = {
            "capture_current_termination_masks": lambda env: SimpleNamespace(term_order=("base_contact",), term_masks={}, done_mask=torch.zeros(1, dtype=torch.bool)),
            "stair_kinematic_base_state": lambda *a, **k: SimpleNamespace(pos_w=torch.zeros(1, 3), quat_xyzw=torch.zeros(1, 4)),
            "quaternion_wxyz": lambda value: value,
            "STAIR_TRANSITION_VNEXT_SCHEMA_VERSION": STAIR_TRANSITION_VNEXT_SCHEMA_VERSION,
            "STAIR_EVAL_TRANSITION_EXTRAS_KEY": STAIR_EVAL_TRANSITION_EXTRAS_KEY,
            "stair_contact_force_history_snapshot": snapshot,
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        projection = SimpleNamespace(contact_sensor_name="contact_forces", legacy_reward_adapter_id=None,
            projection_hash="fixture", reward_input_schema_hash_vnext="fixture", support_count=2, forbidden_contact_count=1,
            joint_names=("hip","calf"))
        env = SimpleNamespace(scene={
            "robot": SimpleNamespace(joint_names=["unused","calf","unused2","hip"],
                data=SimpleNamespace(joint_pos=torch.tensor([[10.,20.,30.,40.]]), body_pos_w=torch.zeros(1, 3, 3), body_lin_vel_w=torch.zeros(1, 3, 3))),
            "contact_forces": SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.zeros(1, 3, 3)))},
            episode_length_buf=torch.ones(1), extras={}, physics_dt=.005, step_dt=.02,
            cfg=SimpleNamespace(scene=SimpleNamespace(lazy_sensor_update=True)))
        recorder = SimpleNamespace(_env=env, runtime_projection=projection, _capture_contact_history=False,
            _resolved_projection=SimpleNamespace(joint_ids=(3, 1), support_body_ids=(1, 2), contact_support_body_ids=(1, 2), contact_forbidden_body_ids=(0,)),
            _terrain_env_origins_w=torch.zeros(1, 3), _terrain_types=torch.zeros(1))
        namespace["record_post_step"](recorder)
        snapshot.assert_not_called()
        self.assertNotIn("contact_force_history", env.extras[STAIR_EVAL_TRANSITION_EXTRAS_KEY])
        recorder._capture_contact_history = True
        namespace["record_post_step"](recorder)
        snapshot.assert_called_once()
        self.assertIn("contact_force_history", env.extras[STAIR_EVAL_TRANSITION_EXTRAS_KEY])
        self.assertEqual(env.extras[STAIR_EVAL_TRANSITION_EXTRAS_KEY]["joint_pos"].tolist(),[[40.,20.]])
        # Execute the production contract expression against the same native
        # order. This is CPU mapping evidence, not an Isaac rollout.
        from true_rl_agent.tasks.stair_climbing.trajectory_capabilities import (
            CONTRACT_VERSION, cell_bindings, geometry_sources, validate_measurement_contract)
        from true_rl_agent.runner.numeric_evidence import file_sha256
        execution = source.with_name("isaaclab_execution.py")
        tree = ast.parse(execution.read_text("utf-8"))
        assignment = next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=="measurement_contract" for t in n.targets))
        namespace.update(CONTRACT_VERSION=CONTRACT_VERSION,cell_bindings=cell_bindings,
            geometry_sources=geometry_sources,Path=Path,file_sha256=file_sha256,__file__=str(execution),
            runtime_projection=projection,robot=env.scene["robot"],resolved=recorder._resolved_projection,
            config=SimpleNamespace(ordered_evaluation_suite=SimpleNamespace(private_cell_assignment=lambda split:[])))
        exec(compile(ast.Module(body=[assignment],type_ignores=[]),str(execution),"exec"),namespace)
        contract = namespace["measurement_contract"]
        validate_measurement_contract(contract,projection)
        self.assertEqual(contract["joint_position"]["resolved_joint_ids"],[3,1])
        self.assertEqual(contract["joint_position"]["joint_names"],["hip","calf"])

    def test_live_to_archive_recomputes_all_outcomes_and_ignores_reset_padding(self):
        self._assert_capture("stair_go2_12dof_unified_protocol_v3.json")

    def test_g1_history_uses_bound_two_support_roles(self):
        self._assert_capture("stair_unified_protocol_v3.json")

    def _assert_capture(self, protocol_filename):
        config = StairClimbingTaskConfig.defaults(ROOT / "configs/tasks" / protocol_filename)
        manifest = resolve_stair_cell_manifest(cell_variant_id=config.cell_variant_id, config=config, plugin=StairClimbingTaskPlugin(config))
        config = replace(config, resolved_cell_task_protocol_hash=manifest.resolved_cell_task_protocol_hash)
        projection = stair_runtime_projection_from_manifest(manifest)
        cells = config.ordered_evaluation_suite.private_cell_assignment("public_dev")
        env = _TwoStepEnv(cells, projection, config)
        kwargs = dict(env=env, inference_policy=lambda observation: observation, protocol=None,
                      semantics=config.evaluation_semantics, seed_resolver=config.seed_resolver(),
                      split="public_dev", seed_count=config.public_seed_count,
                      seed_namespace=STAIR_FORMAL_EVAL_NAMESPACE, runtime_projection=projection)
        off = run_stair_split_execution(**kwargs)
        with tempfile.TemporaryDirectory() as temporary:
            capture = StairNumericCapture(Path(temporary) / "capture", identity={"checkpoint_hash": "fixture"},
                                          config=config, resolved_cell_manifest=manifest)
            on = run_stair_split_execution(**kwargs, numeric_capture=capture)
            self.assertEqual(on, off)
            summary = build_unified_stair_summary(suite=config.ordered_evaluation_suite,
                semantics=config.evaluation_semantics, split="public_dev", per_seed_outcomes=on.per_seed_outcomes)
            ref = capture.complete(summary)
            restored = recompute_stair_numeric_evidence(ref)
            self.assertEqual(restored["summary"], summary)
            self.assertEqual(restored["episode_count"], len(cells) * config.public_seed_count)
            self.assertEqual(restored["profile"], STAIR_NUMERIC_PROFILE)
            self.assertEqual(restored["contact_force_history"]["status"], "recorded")
            metadata, arrays = next(iter_numeric_evidence(ref))
            self.assertEqual(metadata["step_start"], 1)
            self.assertTrue(arrays["valid"][0].all())
            self.assertFalse(arrays["valid"][1, :len(cells)//2].any())
            self.assertTrue(arrays["valid"][1, len(cells)//2:].all())
            self.assertEqual(arrays["base_pos_w"].dtype.name, "float32")
            self.assertEqual(int(arrays["valid"].sum()), sum(o.episode_steps for o in on.per_seed_outcomes[0]))
            self.assertEqual(arrays["forbidden_contact_forces_w_history"].shape, (2, len(cells), 3, 1, 3))
            self.assertFalse(arrays["forbidden_contact_forces_w"].any())
            self.assertTrue((arrays["forbidden_contact_forces_w_history"][0, :len(cells)//2, 1, 0, 0] == -100.).all())
            self.assertFalse(arrays["forbidden_contact_forces_w_history"][1, :len(cells)//2].any())
            # Lossless history survives even though the final instantaneous
            # contact is zero; it does not redefine the existing decoder.
            self.assertEqual(restored["contact_force_history"]["physics_steps_per_control"], 4)
            self.assertEqual(restored["contact_force_history"]["history_order"], "newest_to_oldest")

            original = load_numeric_evidence_manifest(ref)
            legacy_writer = NumericEvidenceWriter(Path(temporary) / "legacy", original["identity"])
            for chunk_metadata, chunk_arrays in iter_numeric_evidence(ref):
                header = dict(chunk_metadata["transition_header"])
                header.pop("contact_force_history")
                legacy_writer.write_chunk({k: v for k, v in chunk_arrays.items() if k not in STAIR_CONTACT_HISTORY_TENSOR_FIELDS},
                    {**chunk_metadata, "transition_header": header})
            legacy_context = {**original["metadata"], "profile": STAIR_NUMERIC_PROFILE_V1}
            legacy_context.pop("contact_force_history")
            legacy = recompute_stair_numeric_evidence(legacy_writer.complete(legacy_context))
            self.assertEqual(legacy["summary"], summary)
            self.assertEqual(legacy["contact_force_history"], {"status": "not_recorded"})

            missing_writer = NumericEvidenceWriter(Path(temporary) / "missing", original["identity"])
            for chunk_metadata, chunk_arrays in iter_numeric_evidence(ref):
                missing_writer.write_chunk({k: v for k, v in chunk_arrays.items() if k != "forbidden_contact_forces_w_history"}, chunk_metadata)
            with self.assertRaisesRegex(NumericEvidenceError, "missing or malformed"):
                recompute_stair_numeric_evidence(missing_writer.complete(original["metadata"]))

    def test_sensor_history_snapshot_preserves_role_order_vectors_clock_and_copies(self):
        history = torch.arange(2 * 3 * 5 * 3, dtype=torch.float32).reshape(2, 3, 5, 3)
        sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w_history=history),
                                 cfg=SimpleNamespace(update_period=0.005))
        resolved = SimpleNamespace(contact_support_body_ids=(3, 0, 4, 2), contact_forbidden_body_ids=(1,))
        result = stair_contact_force_history_snapshot(contact_sensor=sensor, resolved_projection=resolved,
            runtime_projection=SimpleNamespace(contact_history_length=3), physics_dt=0.005, control_dt=0.02,
            lazy_sensor_update=True)
        self.assertTrue(torch.equal(result["support_contact_forces_w_history"], history[:, :, (3, 0, 4, 2), :]))
        expected = result["forbidden_contact_forces_w_history"].clone()
        history[:] = 0
        self.assertTrue(torch.equal(result["forbidden_contact_forces_w_history"], expected))
        self.assertEqual(result["contact_force_history"]["history_clock"], "contact_sensor_buffer_update_events")
        self.assertIn("not_recorded", result["contact_force_history"]["per_history_slot_timestamps"])
        self.assertNotIn("sample_end_offsets_from_control_step_end_s", result["contact_force_history"])
        sensor.cfg.update_period = 0.01
        with self.assertRaisesRegex(RuntimeError, "configured physics-step sensor update period"):
            stair_contact_force_history_snapshot(contact_sensor=sensor, resolved_projection=resolved,
                runtime_projection=SimpleNamespace(contact_history_length=3), physics_dt=0.005, control_dt=0.02,
                lazy_sensor_update=True)


class _TwoStepEnv:
    step_dt = 0.02

    def __init__(self, cells, projection, config):
        self.unwrapped = self
        self.cells, self.projection = cells, projection
        self.term = config.evaluation_semantics.outcomes["signals"]["forbidden_contact"]["manager_term"]

    def reset(self, seed):
        self.step_index = 0
        return torch.zeros((len(self.cells), 1)), {}

    def step(self, actions):
        self.step_index += 1
        n, support = len(self.cells), self.projection.support_count
        origin = torch.tensor([c.expected_origin_m for c in self.cells], dtype=torch.float32)
        base = origin + torch.tensor([0., 0., 0.35])
        positions = origin[:, None, :].repeat(1, support, 1)
        forces = torch.zeros((n, 1, 3))
        done = torch.arange(n) < n//2 if self.step_index == 1 else torch.arange(n) >= n//2
        # Synthetic earlier buffer entry is nonzero; current contact is zero.
        history = torch.zeros((n, 3, support + 1, 3))
        history[done, 1, support, 0] = -100.
        history[:, 1, :support, 2] = 100.
        history_fields = stair_contact_force_history_snapshot(
            contact_sensor=SimpleNamespace(data=SimpleNamespace(net_forces_w_history=history),
                                           cfg=SimpleNamespace(update_period=0.005)),
            resolved_projection=SimpleNamespace(contact_support_body_ids=tuple(range(support)),
                                                 contact_forbidden_body_ids=(support,)),
            runtime_projection=self.projection, physics_dt=0.005, control_dt=self.step_dt, lazy_sensor_update=True)
        raw = dict(schema_version=STAIR_TRANSITION_VNEXT_SCHEMA_VERSION,
            base_pos_w=base, base_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1),
            joint_pos=torch.zeros((n, self.projection.joint_count)), support_pos_w=positions,
            support_lin_vel_w=torch.zeros((n, support, 3)), support_contact_forces_w=torch.zeros((n, support, 3)),
            forbidden_contact_forces_w=forces, episode_length=torch.full((n,), self.step_index, dtype=torch.int64),
            terrain_env_origins_w=origin, terrain_types=torch.tensor([c.terrain_type_index for c in self.cells]),
            termination_term_order=("time_out", self.term), termination_masks={"time_out": torch.zeros(n, dtype=torch.bool), self.term: done},
            done_mask=done, runtime_projection_hash=self.projection.projection_hash,
            reward_input_schema_hash_vnext=self.projection.reward_input_schema_hash_vnext,
            support_count=support, forbidden_contact_count=1, **history_fields)
        return actions, torch.zeros(n), done, {STAIR_EVAL_TRANSITION_EXTRAS_KEY: raw}


if __name__ == "__main__":
    unittest.main()
