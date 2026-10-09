"""CPU execution of production Stair recorder method and native PNG pipeline.

Kit imports are isolated by compiling the actual record_post_step AST unchanged;
only Isaac data/projection dependencies are replaced. No mirror recorder or
simulation is implemented here, and no GPU or encoder is invoked.
"""
from __future__ import annotations

import ast
import json
import tempfile
import types
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from true_rl_agent.backends.isaac_lab_arrays import quaternion_wxyz
from true_rl_agent.runner.native_video import NativeVideoOptions, NativeVideoRecorder, native_video_recorders
from true_rl_agent.tasks.stair_climbing.isaaclab_execution import (
    _finish_stair_seed_video, _stair_pre_reset_video_capture,
)
from true_rl_agent.tasks.stair_climbing.runtime_profile import (
    STAIR_EVAL_TRANSITION_EXTRAS_KEY, STAIR_TRANSITION_RECORDER_SCHEMA_VERSION,
)
from true_rl_agent.tasks.stair_climbing.runtime_projection import STAIR_TRANSITION_VNEXT_SCHEMA_VERSION


ROOT = Path(__file__).resolve().parents[2]


def production_post_step():
    path = ROOT / "src/true_rl_agent/tasks/stair_climbing/isaaclab_runtime_env.py"
    cls = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
               if isinstance(n, ast.ClassDef) and n.name == "StairEvalTransitionRecorder")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "record_post_step")
    ns = {"capture_current_termination_masks": lambda env: env.termination,
          "stair_kinematic_base_state": lambda data, **kw: SimpleNamespace(
              pos_w=data.root_pos_w.torch, quat_xyzw=data.root_quat_w),
          "quaternion_wxyz": quaternion_wxyz,
          "STAIR_EVAL_TRANSITION_EXTRAS_KEY": STAIR_EVAL_TRANSITION_EXTRAS_KEY,
          "STAIR_TRANSITION_RECORDER_SCHEMA_VERSION": STAIR_TRANSITION_RECORDER_SCHEMA_VERSION,
          "STAIR_TRANSITION_VNEXT_SCHEMA_VERSION": STAIR_TRANSITION_VNEXT_SCHEMA_VERSION}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), ns)
    return ns["record_post_step"]


class Array:
    def __init__(self, tensor): self.torch = tensor
    def __getitem__(self, key): return self.torch[key]
    def __getattr__(self, name): return getattr(self.torch, name)


class Scene(dict):
    env_prim_paths = ("env0", "env1")


class Imageable:
    def __init__(self): self.visible = True
    def MakeVisible(self): self.visible = True
    def MakeInvisible(self): self.visible = False


class CpuEnv:
    def __init__(self, *, legacy, gym_terminal=2):
        self.unwrapped = self
        self.step_dt = .02
        self.step_number, self.gym_terminal = 0, gym_terminal
        self.extras, self.numeric = {}, []
        self.episode_length_buf = torch.zeros(2, dtype=torch.int64)
        self.data = SimpleNamespace(**{name: Array(torch.tensor(value, dtype=torch.float32)) for name, value in {
            "root_pos_w": [[0, 0, 1], [100, 0, 1]],
            "root_quat_w": [[0, 0, 0, 1], [0, 0, 0, 1]],
            "root_lin_vel_w": [[0, 0, 0], [0, 0, 0]],
            "root_ang_vel_w": [[0, 0, 0], [0, 0, 0]],
            "joint_pos": [[0, 1], [0, 1]], "joint_vel": [[0, 0], [0, 0]],
            "body_pos_w": np.zeros((2, 3, 3)), "body_lin_vel_w": np.zeros((2, 3, 3)),
            "body_link_pos_w": np.zeros((2, 3, 3)),
            "body_link_quat_w": np.tile([0, 0, 0, 1], (2, 3, 1)),
        }.items()})
        self.scene = Scene(robot=SimpleNamespace(data=self.data, joint_names=["j0", "j1"], body_names=["b0", "b1", "b2"]),
                           contact=SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.zeros(2, 3, 3))))
        self.imageables = {name: Imageable() for name in self.scene.env_prim_paths}
        self.view_env_index = 0
        self.camera_eye, self.camera_events = (0., 0., 0.), []
        self.video_recorder = SimpleNamespace(_capture=SimpleNamespace(_render_product=None))
        self.viewport_camera_controller = SimpleNamespace(
            set_view_env_index=lambda index: setattr(self, "view_env_index", index),
            update_view_to_asset_root=lambda *a: None, update_view_location=self.move_camera)
        self._render_pixels = np.zeros((2, 2, 3), dtype=np.uint8)
        self.sim = SimpleNamespace(stage=SimpleNamespace(GetPrimAtPath=self.imageables.__getitem__),
            get_physics_step_count=lambda: self.step_number * 4,
            physics_manager=SimpleNamespace(get_simulation_time=lambda: self.step_number * .02))
        self.transition = SimpleNamespace(_env=self,
            runtime_projection=SimpleNamespace(contact_sensor_name="contact", legacy_reward_adapter_id="legacy" if legacy else None,
                projection_hash="projection", reward_input_schema_hash_vnext="inputs", support_count=2, forbidden_contact_count=1),
            _resolved_projection=SimpleNamespace(joint_ids=[1, 0], support_body_ids=[0, 1],
                contact_support_body_ids=[0, 1], contact_forbidden_body_ids=[2]),
            _terrain_env_origins_w=torch.zeros(2, 3), _terrain_types=torch.zeros(2, dtype=torch.int64),
            _capture_contact_history=False)
        self.post_step = production_post_step()

    def move_camera(self, *, eye, lookat):
        self.camera_eye = tuple(eye)
        self.camera_events.append(("view", self.view_env_index, self.camera_eye))

    def render(self):
        assert [image.visible for image in self.imageables.values()] == [self.view_env_index == i for i in range(2)]
        capture = self.video_recorder._capture
        if capture._render_product is None:
            capture._render_product = SimpleNamespace(path="/Render/CpuFixture", env=self)
        self.camera_events.append(("render", self.view_env_index, self.camera_eye))
        # Asymmetric sentinels test transport and view ownership, not camera geometry.
        self.rendered_camera_params = {
            "cameraModel": "pinhole", "renderProductResolution": np.array([2, 2]),
            "cameraViewTransform": np.array([1, 2, 3, self.view_env_index, 5, 6, 7, self.camera_eye[0],
                9, 10, 11, float(self.data.root_pos_w.torch[self.view_env_index, 0]), 13, 14, 15, 16]),
            "cameraProjection": np.arange(20, 36), "cameraAperture": np.array([12., 8.]),
            "cameraApertureOffset": np.array([.25, -.5]), "cameraNearFar": np.array([.1, 1000.]),
            "cameraFocalLength": np.array(35.), "metersPerSceneUnit": np.array(.01),
        }
        previous = self._render_pixels
        self._render_pixels = np.full((2, 2, 3), int(self.data.root_pos_w.torch[self.view_env_index, 0]), dtype=np.uint8)
        return previous  # Kit-like one-render camera/state cache; production must discard it.

    def step(self, action):
        self.step_number += 1
        self.episode_length_buf += 1
        self.data.root_pos_w.torch[:, 0] = torch.tensor([self.step_number * 10, self.step_number * 10 + 100])
        done = torch.tensor([self.step_number == self.gym_terminal, self.step_number == 3])
        self.termination = SimpleNamespace(term_order=("timeout",), term_masks={"timeout": done}, done_mask=done)
        self.post_step(self.transition)  # The unchanged production method clones and invokes the hook.
        payload = self.extras[STAIR_EVAL_TRANSITION_EXTRAS_KEY]
        self.numeric.append(payload)
        if done[0]:
            self.data.root_pos_w.torch[0, 0] = 99  # Native reset after recorder post-step.
            self.episode_length_buf[0] = 0
        return None, None, done, self.extras


class Wrapper:
    """Only Gym delegation; recording behavior is the real nested RecordingEnv."""
    def __init__(self, env): self.env, self.unwrapped = env, env.unwrapped
    def render(self): return self.env.render()
    def __getattr__(self, name): return getattr(self.env, name)


class CameraParamsAnnotator:
    @staticmethod
    def get_annotator(name):
        assert name == "camera_params"
        return CameraParamsAnnotator()

    def attach(self, products):
        self.product, = products
        self.product.env.camera_events.append(("attach",))

    def get_data(self):
        env = self.product.env
        env.camera_events.append(("params", env.view_env_index, env.camera_eye))
        return env.rendered_camera_params


def native_modules():
    camera = types.ModuleType("isaaclab.envs.ui.viewport_camera_controller")
    camera.ViewportCameraController = object
    gym = types.ModuleType("gymnasium")
    gym.Wrapper = Wrapper
    pxr = types.ModuleType("pxr")
    pxr.UsdGeom = SimpleNamespace(Imageable=lambda prim: prim)
    omni = types.ModuleType("omni")
    omni.replicator = types.ModuleType("omni.replicator")
    omni.replicator.core = types.ModuleType("omni.replicator.core")
    omni.replicator.core.AnnotatorRegistry = CameraParamsAnnotator
    return {"gymnasium": gym, "pxr": pxr, "isaaclab.envs.ui.viewport_camera_controller": camera,
            "omni": omni, "omni.replicator": omni.replicator, "omni.replicator.core": omni.replicator.core}


def first_pixel(path):
    png, offset, chunks = path.read_bytes(), 8, []
    while offset < len(png):
        length = int.from_bytes(png[offset:offset + 4], "big")
        if png[offset + 4:offset + 8] == b"IDAT":
            chunks.append(png[offset + 8:offset + 8 + length])
        offset += length + 12
    return zlib.decompress(b"".join(chunks))[1]


class StairPreResetMediaTests(unittest.TestCase):
    def test_two_envs_share_physics_but_keep_distinct_pixels_files_and_terminal_steps(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("sys.modules", native_modules()):
            base = CpuEnv(legacy=False)
            options = NativeVideoOptions(Path(tmp), resolution=(2, 2), snapshot_interval_steps=1,
                secondary_env_index=1, secondary_eye=(-1., 0., 1.), secondary_lookat=(0., 0., 0.))
            recorders = native_video_recorders(base, options)
            self.assertEqual([r.env.env is base for r in recorders], [True, True])
            self.assertEqual([x.visible for x in base.imageables.values()], [False, True])
            with _stair_pre_reset_video_capture(base, *recorders):
                for recorder in recorders:
                    recorder.start(f"seed_env_{recorder.options.env_index}", 17, task_records_steps=True)
                for _ in range(3): recorders[0].env.step(None)
            outcomes = tuple(SimpleNamespace(episode_steps=t, to_public_dict=lambda t=t: {"episode_steps": t})
                             for t in (2, 3))
            records = [_finish_stair_seed_video(recorder, outcomes, split="public_dev", ordered_suite=None)
                       for recorder in recorders]
            self.assertEqual(base.step_number, 3)
            self.assertEqual([r._control_step for r in recorders], [3, 3])
            self.assertEqual([r["frame_count"] for r in records], [2, 3])
            for recorder in recorders:
                self.assertIs(recorder._camera_params_annotator.product, base.video_recorder._capture._render_product)
            events = base.camera_events
            attachments = [i for i, event in enumerate(events) if event[0] == "attach"]
            self.assertEqual(len(attachments), 2)
            for i in attachments:
                self.assertGreater(i, 0)
                self.assertEqual(events[i - 1][0], "render")
            for i, event in enumerate(events):
                if event[0] == "params":
                    self.assertEqual(events[i - 3:i], [(kind, *event[1:]) for kind in ("view", "render", "render")])
            self.assertEqual(sum(event[0] == "params" for event in events), 12)
            paths = []
            for env_index, record in enumerate(records):
                self.assertTrue(record["includes_terminal_frame"])
                self.assertEqual(record["env_index"], env_index)
                self.assertEqual(record["seed"], 17)
                saved_manifest = json.loads((Path(tmp) / record["file"]).read_text(encoding="utf-8"))
                self.assertEqual(saved_manifest, record["manifest"])
                samples = saved_manifest["samples"]
                self.assertEqual([s["control_step"] for s in samples], list(range(1, 3 + env_index)))
                for sample in samples:
                    expected = sample["control_step"] * 10 + env_index * 100
                    self.assertEqual(sample["state_before"], sample["state_after"])
                    self.assertEqual(sample["state_before"]["root_pos_w"][0], expected)
                    self.assertEqual(sample["state_before"]["physics_step_count"], sample["control_step"] * 4)
                    for view in sample["views"]:
                        paths.append(view["file"])
                        self.assertEqual(first_pixel(Path(tmp) / view["file"]), expected)
                        eye = options.eye if view["view_id"] == "v0" else options.secondary_eye
                        self.assertEqual(view["camera_render_product_path"], "/Render/CpuFixture")
                        self.assertEqual(view["camera_params"], {
                            "cameraModel": "pinhole", "renderProductResolution": [2, 2],
                            "cameraViewTransform": [[1, 2, 3, env_index], [5, 6, 7, eye[0]],
                                                    [9, 10, 11, expected], [13, 14, 15, 16]],
                            "cameraProjection": [[20, 21, 22, 23], [24, 25, 26, 27],
                                                 [28, 29, 30, 31], [32, 33, 34, 35]],
                            "cameraAperture": [12., 8.], "cameraApertureOffset": [.25, -.5],
                            "cameraNearFar": [.1, 1000.], "cameraFocalLength": 35., "metersPerSceneUnit": .01,
                        })
            self.assertEqual(len(paths), len(set(paths)))
            self.assertEqual(len(list(Path(tmp).glob("*.png"))), 10)
            self.assertFalse(list(Path(tmp).glob("seed_env_0.step000003.*")))

    def test_real_post_step_and_native_wrapper_keep_terminal_pixels_and_numeric_values(self):
        for legacy in (False, True):
            for gym_terminal in (2, 3):
                with self.subTest(legacy=legacy, gym_terminal=gym_terminal), tempfile.TemporaryDirectory() as tmp, \
                     patch.dict("sys.modules", native_modules()):
                    base = CpuEnv(legacy=legacy, gym_terminal=gym_terminal)
                    options = NativeVideoOptions(Path(tmp), resolution=(2, 2), snapshot_interval_steps=1,
                        secondary_eye=(-1., 0., 1.), secondary_lookat=(0., 0., 0.))
                    recorder = NativeVideoRecorder(base, options)
                    original = recorder.record_step
                    observed_payloads = []
                    field = "root_pos_w" if legacy else "base_pos_w"
                    def assert_then_record():
                        payload = base.extras[STAIR_EVAL_TRANSITION_EXTRAS_KEY]
                        self.assertTrue(torch.equal(payload[field], base.data.root_pos_w.torch))
                        observed_payloads.append(int(payload["episode_length"][0]))
                        original()
                    recorder.record_step = assert_then_record
                    prior = lambda: None
                    base._stair_pre_reset_capture = prior
                    with _stair_pre_reset_video_capture(base, recorder):
                        recorder.start("seed", 17, task_records_steps=True)
                        for _ in range(3): recorder.env.step(None)
                    self.assertIs(base._stair_pre_reset_capture, prior)
                    self.assertEqual(recorder._control_step, 3)  # Real wrapper did not double-record.
                    self.assertEqual(len(observed_payloads), 3)
                    self.assertEqual(int(base.numeric[1][field][0, 0]), 20)  # clone survived native reset
                    outcome = SimpleNamespace(episode_steps=2, to_public_dict=lambda: {"episode_steps": 2})
                    record = _finish_stair_seed_video(recorder, (outcome,), split="public_dev", ordered_suite=None)
                    self.assertEqual([x["control_step"] for x in record["manifest"]["samples"]], [1, 2])
                    self.assertEqual(record["terminal_control_step"], 2)
                    self.assertTrue(record["includes_terminal_frame"])
                    self.assertEqual(record["capture_point"], "after_stair_numeric_clone_before_reset_and_interval_events")
                    terminal = record["manifest"]["samples"][-1]
                    self.assertEqual([first_pixel(Path(tmp) / v["file"]) for v in terminal["views"]], [20, 20])
                    self.assertEqual(terminal["state_before"], terminal["state_after"])
                    self.assertFalse(list(Path(tmp).glob("*.step000003.*")))
                    baseline = CpuEnv(legacy=legacy, gym_terminal=gym_terminal)
                    for _ in range(3): baseline.step(None)
                    for captured, plain in zip(base.numeric, baseline.numeric, strict=True):
                        for key, value in captured.items():
                            if isinstance(value, torch.Tensor): self.assertTrue(torch.equal(value, plain[key]), key)

    def test_sparse_fixed_steps_do_not_claim_a_terminal_image(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("sys.modules", native_modules()):
            base = CpuEnv(legacy=False)
            recorder = NativeVideoRecorder(base, NativeVideoOptions(Path(tmp), resolution=(2, 2), snapshot_steps=(1,)))
            with _stair_pre_reset_video_capture(base, recorder):
                recorder.start("sparse", 17, task_records_steps=True)
                for _ in range(3): recorder.env.step(None)
            outcome = SimpleNamespace(episode_steps=2, to_public_dict=lambda: {"episode_steps": 2})
            record = _finish_stair_seed_video(recorder, (outcome,), split="public_dev", ordered_suite=None)
            self.assertFalse(record["includes_terminal_frame"])
            self.assertEqual(record["terminal_control_step"], 2)
            self.assertEqual([s["control_step"] for s in record["manifest"]["samples"]], [1])

    def test_hook_restores_on_capture_failure_or_removes_new_binding(self):
        for existing in (False, True):
            base = SimpleNamespace()
            prior = lambda: None
            if existing: base._stair_pre_reset_capture = prior
            def fail(): raise RuntimeError("render failed")
            with self.assertRaisesRegex(RuntimeError, "render failed"):
                with _stair_pre_reset_video_capture(base, SimpleNamespace(record_step=fail)):
                    base._stair_pre_reset_capture()
            if existing: self.assertIs(base._stair_pre_reset_capture, prior)
            else: self.assertFalse(hasattr(base, "_stair_pre_reset_capture"))
